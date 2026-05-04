import asyncio
from typing import Any, Dict, Iterator, Optional

from backend.common.logger import get_logger
from backend.domain.entities.stream_event import StreamEvent
from backend.services.agent.engine import AgentEngine
from backend.services.agent.supervisor import AgentSupervisor
from backend.services.context_engine import ContextEngine
from backend.services.profile_extractor import ProfileExtractor
from backend.services.session.manager import SessionManager
from backend.services.streaming.sse_manager import SSEManager

logger = get_logger(__name__)


class AgentApplication:
    """Agent workflow orchestrator."""

    def __init__(
        self,
        agent_engine: AgentEngine,
        session_manager: SessionManager,
        context_engine: ContextEngine,
        profile_extractor: ProfileExtractor,
        supervisor: AgentSupervisor | None = None,
    ):
        self.engine = agent_engine
        self.session_mgr = session_manager
        self.context_engine = context_engine
        self.profile_extractor = profile_extractor
        self.supervisor = supervisor

    def stream_agent_task(
        self,
        session_id: str,
        query: str,
        use_user_memory: bool = False,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        history_rounds: int = 10,
    ) -> Iterator[str]:
        final_answer_accumulated = ""
        persist_messages: list[dict] | None = None
        should_persist_partial = False
        history_budget = max(1, history_rounds)
        try:
            logger.info("Entering agent mode for session %s", session_id)

            history = self.session_mgr.get_chat_context(
                session_id,
                max_rounds=history_budget,
            )
            user_profile = (
                self.profile_extractor.render_profile() if use_user_memory else None
            )
            model_options: Dict[str, Any] = {
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
            }
            event_stream = (
                self.supervisor.stream(
                    query,
                    history,
                    user_profile=user_profile,
                    model_options=model_options,
                )
                if self.supervisor
                else self._stream_legacy_agent(
                    query,
                    history,
                    use_user_memory,
                    user_profile,
                    model_options,
                )
            )

            for event in event_stream:
                event_type = event.get("type")

                if event_type == "supervisor_route":
                    yield SSEManager.format_event(
                        StreamEvent(
                            event="supervisor_route",
                            data={
                                "worker": event.get("worker"),
                                "worker_kind": event.get("worker_kind"),
                                "intent": event.get("intent"),
                                "confidence": event.get("confidence"),
                                "reason": event.get("reason"),
                                "signals": event.get("signals", []),
                            },
                        )
                    )

                elif event_type == "status":
                    yield SSEManager.format_event(StreamEvent.status(event.get("content", "")))

                elif event_type == "reasoning":
                    yield SSEManager.format_event(StreamEvent.reasoning(event.get("content", "")))

                elif event_type == "message_delta":
                    delta = event.get("content", "")
                    final_answer_accumulated += delta
                    should_persist_partial = True
                    yield SSEManager.format_event(StreamEvent.message_delta(delta))

                elif event_type == "retrieval_trace":
                    yield SSEManager.format_event(
                        StreamEvent.retrieval_trace(event.get("data", {}))
                    )

                elif event_type == "citation_trace":
                    yield SSEManager.format_event(
                        StreamEvent.citation_trace(event.get("data", {}))
                    )

                elif event_type == "answer_guard":
                    yield SSEManager.format_event(
                        StreamEvent.answer_guard(event.get("data", {}))
                    )

                elif event_type == "self_correction":
                    yield SSEManager.format_event(
                        StreamEvent.self_correction(
                            message=event.get("content", ""),
                            data={
                                "tool_call_id": event.get("tool_call_id"),
                                "name": event.get("name"),
                                "error": event.get("error"),
                                "state": event.get("state", {}),
                            },
                        )
                    )

                elif event_type == "tool_call":
                    yield SSEManager.format_event(
                        StreamEvent.tool_call(
                            name=event.get("name", ""),
                            args=event.get("args", {}),
                            tool_call_id=event.get("tool_call_id"),
                        )
                    )

                elif event_type == "tool_result":
                    yield SSEManager.format_event(
                        StreamEvent.tool_result(
                            name=event.get("name", ""),
                            content=str(event.get("content", "")),
                            tool_call_id=event.get("tool_call_id"),
                            success=event.get("success"),
                            error=event.get("error"),
                            metadata=event.get("metadata", {}),
                        )
                    )

                elif event_type == "final_answer":
                    final_answer_accumulated = event.get("content", "")
                    should_persist_partial = True
                    yield SSEManager.format_event(
                        StreamEvent(
                            event="final_answer",
                            content=final_answer_accumulated,
                            data=event.get("state", {}),
                        )
                    )

                elif event_type == "error":
                    yield SSEManager.format_event(
                        StreamEvent(
                            event="error",
                            message=event.get("content", ""),
                            data=event.get("state", {}),
                        )
                    )

                elif event_type == "finish":
                    new_messages = event.get("new_messages", [])
                    persist_messages = [{"role": "user", "content": query}] + new_messages
                    yield SSEManager.format_end()
                    return

        except (GeneratorExit, asyncio.CancelledError):
            logger.warning("Agent stream closed early for session %s", session_id)
            raise
        except Exception as exc:
            logger.error("Agent workflow failed: %s", exc)
            yield SSEManager.format_error(str(exc))
            yield SSEManager.format_end()
        finally:
            if persist_messages:
                self.session_mgr.add_messages(session_id, persist_messages)
                logger.info("Persisted %s agent trace messages", len(persist_messages))
            elif should_persist_partial and (query.strip() or final_answer_accumulated.strip()):
                self.session_mgr.save_interaction(
                    session_id,
                    query,
                    final_answer_accumulated,
                )

    def _stream_legacy_agent(
        self,
        query: str,
        history,
        use_user_memory: bool,
        user_profile: str | None,
        model_options: Optional[Dict[str, Any]] = None,
    ):
        context = self.context_engine.build_agent_context(
            query=query,
            history=history,
            use_user_memory=use_user_memory,
            user_profile=user_profile,
        )
        yield from self.engine.stream_run(context.messages, model_options=model_options)
