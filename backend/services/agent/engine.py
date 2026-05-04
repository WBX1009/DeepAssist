from typing import Any, Callable, Dict, Iterator, List, Optional

from backend.common.config import settings
from backend.common.logger import get_logger
from backend.domain.entities.agent_run import AgentRunConfig, AgentTerminationReason
from backend.domain.entities.tooling import ToolCall, ToolResult
from backend.domain.interfaces.llm import BaseLLM
from backend.services.agent.lifecycle import AgentLifecycle
from backend.services.agent.tooling import ToolRegistry

logger = get_logger(__name__)


class AgentEngine:
    """ReAct-style agent loop with canonical ToolCall/ToolResult flow."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[Callable[..., Any]]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        run_config: Optional[AgentRunConfig] = None,
    ):
        self.llm = llm
        self.tool_registry = tool_registry or ToolRegistry.from_callables(tools or [])
        self.openai_tools = self.tool_registry.openai_schemas()
        self.run_config = run_config or AgentRunConfig(
            max_iterations=settings.MAX_AGENT_STEPS,
            max_tool_errors=3,
            repeated_tool_call_limit=3,
        )

    def stream_run(
        self,
        messages: List[Dict[str, Any]],
        model_options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        current_messages = messages.copy()
        lifecycle = AgentLifecycle(self.run_config)
        model_kwargs = {
            key: value for key, value in (model_options or {}).items() if value is not None
        }

        while lifecycle.start_iteration():
            yield {
                "type": "status",
                "content": (
                    f"Starting reasoning step {lifecycle.state.iterations}/"
                    f"{self.run_config.max_iterations}"
                ),
                "state": lifecycle.state.terminal_payload(),
            }

            try:
                response_msg = self.llm.chat(
                    messages=current_messages,
                    tools=self.openai_tools,
                    **model_kwargs,
                )
            except Exception as exc:
                logger.error("LLM call failed during agent run: %s", exc)
                lifecycle.fail(AgentTerminationReason.LLM_ERROR, str(exc))
                yield self._terminal_error_event(lifecycle)
                yield self._finish_event(messages, current_messages, lifecycle)
                return

            safe_content = response_msg.content or ""
            if getattr(response_msg, "reasoning_content", None):
                yield {"type": "reasoning", "content": response_msg.reasoning_content}

            tool_calls = [
                ToolCall.from_llm_tool_call(tool_call)
                for tool_call in (getattr(response_msg, "tool_calls", None) or [])
            ]
            current_messages.append(self._assistant_message(safe_content, tool_calls))

            if not tool_calls:
                lifecycle.complete(safe_content)
                yield {
                    "type": "final_answer",
                    "content": safe_content,
                    "state": lifecycle.state.terminal_payload(),
                }
                yield self._finish_event(messages, current_messages, lifecycle)
                return

            if safe_content.strip():
                yield {"type": "status", "content": safe_content}

            if lifecycle_error := lifecycle.guard_tool_phase():
                for pending_call in tool_calls:
                    result = ToolResult.lifecycle_error(
                        pending_call,
                        lifecycle_error,
                        metadata=lifecycle.state.terminal_payload(),
                    )
                    yield self._tool_result_event(result)
                    current_messages.append(result.to_tool_message())

                yield self._terminal_error_event(lifecycle)
                yield self._finish_event(messages, current_messages, lifecycle)
                return

            for tool_call in tool_calls:
                yield {
                    "type": "tool_call",
                    "name": tool_call.name,
                    "args": tool_call.args,
                    "tool_call_id": tool_call.id,
                    "data": tool_call.to_stream_data(),
                }

                tool_result = self._execute_tool_call(tool_call, lifecycle)
                yield self._tool_result_event(tool_result)
                current_messages.append(tool_result.to_tool_message())

                if lifecycle.state.is_terminal:
                    yield self._terminal_error_event(lifecycle)
                    yield self._finish_event(messages, current_messages, lifecycle)
                    return

                if lifecycle.record_tool_result(tool_result):
                    yield self._terminal_error_event(lifecycle)
                    yield self._finish_event(messages, current_messages, lifecycle)
                    return

                if not tool_result.success:
                    yield self._self_correction_event(tool_result, lifecycle)

        yield self._terminal_error_event(lifecycle)
        yield self._finish_event(messages, current_messages, lifecycle)

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
        lifecycle: AgentLifecycle,
    ) -> ToolResult:
        if tool_call.validation_error:
            return ToolResult.invalid_arguments(tool_call, tool_call.validation_error)

        repeated_call_error = lifecycle.record_tool_call(tool_call)
        if repeated_call_error:
            return repeated_call_error

        return self.tool_registry.execute(tool_call)

    def _assistant_message(
        self,
        safe_content: str,
        tool_calls: List[ToolCall],
    ) -> Dict[str, Any]:
        msg_dict: Dict[str, Any] = {"role": "assistant", "content": safe_content}
        if tool_calls:
            msg_dict["tool_calls"] = [
                tool_call.to_openai_tool_call() for tool_call in tool_calls
            ]
        return msg_dict

    def _tool_result_event(self, tool_result: ToolResult) -> Dict[str, Any]:
        payload = tool_result.to_stream_event().to_payload()
        return {
            "type": payload["event"],
            "name": payload.get("name", tool_result.name),
            "content": payload.get("content", tool_result.to_observation()),
            "success": tool_result.success,
            "error": tool_result.error,
            "tool_call_id": tool_result.tool_call_id,
            "metadata": tool_result.metadata,
            "data": payload.get("data", {}),
        }

    def _terminal_error_event(self, lifecycle: AgentLifecycle) -> Dict[str, Any]:
        payload = lifecycle.state.terminal_payload()
        return {
            "type": "error",
            "content": lifecycle.state.error or "Agent run stopped.",
            "status": payload["status"],
            "termination_reason": payload["termination_reason"],
            "state": payload,
        }

    def _self_correction_event(
        self,
        tool_result: ToolResult,
        lifecycle: AgentLifecycle,
    ) -> Dict[str, Any]:
        return {
            "type": "self_correction",
            "content": (
                "Tool call failed; the error observation was returned to the model "
                "for bounded self-correction."
            ),
            "tool_call_id": tool_result.tool_call_id,
            "name": tool_result.name,
            "error": tool_result.error,
            "state": lifecycle.state.terminal_payload(),
        }

    def _finish_event(
        self,
        initial_messages: List[Dict[str, Any]],
        current_messages: List[Dict[str, Any]],
        lifecycle: AgentLifecycle,
    ) -> Dict[str, Any]:
        return {
            "type": "finish",
            "new_messages": current_messages[len(initial_messages) :],
            "status": lifecycle.state.status.value,
            "termination_reason": lifecycle.state.termination_reason.value
            if lifecycle.state.termination_reason
            else None,
            "state": lifecycle.state.terminal_payload(),
        }
