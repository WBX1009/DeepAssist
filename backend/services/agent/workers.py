import re
from typing import Any, Dict, Iterator, List, Optional

from backend.common.logger import get_logger
from backend.domain.entities.agent_plan import MultiAgentPlan
from backend.domain.entities.agent_worker import AgentWorkerType
from backend.domain.interfaces.llm import BaseLLM
from backend.services.agent.engine import AgentEngine
from backend.services.agent.task_decomposer import TaskDecomposer
from backend.services.context_engine import ContextEngine
from backend.services.rag.pipeline import RAGPipeline

logger = get_logger(__name__)

class BaseAgentWorker:
    worker_type: AgentWorkerType

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def _model_options(self, model_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not model_options:
            return {}
        return {key: value for key, value in model_options.items() if value is not None}

class ChatWorker(BaseAgentWorker):
    worker_type = AgentWorkerType.CHAT

    def __init__(self, llm: BaseLLM, context_engine: ContextEngine):
        self.llm = llm
        self.context_engine = context_engine

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        ChatWorker 不支持恢复中断任务，因此即便 recovery_state 被传入，也会被忽略。
        """
        yield {"type": "status", "content": "Supervisor selected ChatWorker"}
        context = self.context_engine.build_quick_context(
            query=query,
            history=history,
            user_profile=user_profile,
        )
        final_answer = ""
        for chunk in self.llm.chat_stream(context.messages, **self._model_options(model_options)):
            if isinstance(chunk, dict):
                if chunk.get("type") == "reasoning":
                    yield {"type": "reasoning", "content": chunk["content"]}
                else:
                    final_answer += chunk["content"]
                    yield {"type": "message_delta", "content": chunk["content"]}
            else:
                final_answer += chunk
                yield {"type": "message_delta", "content": chunk}
        yield {
            "type": "finish",
            "new_messages": [
                {"role": "assistant", "content": final_answer},
            ],
            "worker": self.worker_type.value,
        }

class RAGWorker(BaseAgentWorker):
    worker_type = AgentWorkerType.RAG
    _EXPLICIT_KB_QUERY_PATTERN = re.compile(
        r"(根据|基于).*(知识库|文档|资料)|"
        r"(知识库|文档|资料).*(回答|作答|引用|依据|检索|查找)|"
        r"(请).*(引用|标注)\s*",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        llm: BaseLLM,
        context_engine: ContextEngine,
        rag_pipeline: RAGPipeline,
        collection_name: str = "__all__",
    ):
        self.llm = llm
        self.context_engine = context_engine
        self.rag_pipeline = rag_pipeline
        self.collection_name = collection_name

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        yield {"type": "status", "content": "Supervisor selected RAGWorker"}
        pipeline_result = self.rag_pipeline.build_context(query, self.collection_name)
        context = self.context_engine.build_rag_context_from_pipeline(
            history=history,
            pipeline_result=pipeline_result,
            user_profile=user_profile,
        )

        yield {
            "type": "retrieval_trace",
            "data": pipeline_result.retrieval_result.to_stream_data(),
        }
        yield {
            "type": "citation_trace",
            "data": pipeline_result.context_pack.to_stream_data(),
        }

        rag_decision = self._decide_rag_fallback(query, pipeline_result)
        if rag_decision["action"] == "direct_kb_miss":
            yield {
                "type": "status",
                "content": rag_decision["status"],
            }
            answer = rag_decision["answer"]
            yield {"type": "message_delta", "content": answer}
            yield {
                "type": "answer_guard",
                "data": {
                    "grounded": False,
                    "recommended_action": "fallback_without_kb_claims",
                    "reason": "explicit_kb_query_without_reliable_support",
                    "warnings": ["retrieval_insufficient_for_kb_answer"],
                },
            }
            yield {
                "type": "finish",
                "new_messages": [
                    {"role": "assistant", "content": answer},
                ],
                "worker": self.worker_type.value,
            }
            return

        if rag_decision["action"] == "fallback_to_chat":
            yield {
                "type": "status",
                "content": rag_decision["status"],
            }
            context = self.context_engine.build_quick_context(
                query=query,
                history=history,
                user_profile=user_profile,
            )
        elif rag_decision["action"] == "proceed_with_warning":
            yield {
                "type": "status",
                "content": rag_decision["status"],
            }

        final_answer_chunks = []
        for chunk in self.llm.chat_stream(context.messages, **self._model_options(model_options)):
            if isinstance(chunk, dict):
                if chunk.get("type") == "reasoning":
                    yield {"type": "reasoning", "content": chunk["content"]}
                else:
                    final_answer_chunks.append(chunk["content"])
            else:
                final_answer_chunks.append(chunk)

        final_answer = "".join(final_answer_chunks)

        guard_report = self.rag_pipeline.check_answer(final_answer, pipeline_result)
        final_answer = self._apply_rag_guard_action(
            answer=final_answer,
            guard_report=guard_report,
            query=query,
            pipeline_result=pipeline_result,
        )
        for chunk in self._chunk_text(final_answer):
            yield {"type": "message_delta", "content": chunk}
        yield {"type": "answer_guard", "data": guard_report.to_stream_data()}
        yield {
            "type": "finish",
            "new_messages": [
                {"role": "assistant", "content": final_answer},
            ],
            "worker": self.worker_type.value,
        }

    def _decide_rag_fallback(self, query: str, pipeline_result) -> Dict[str, str]:
        diagnostics = (
            pipeline_result.retrieval_result.metadata.get("diagnostics", {})
            if pipeline_result.retrieval_result.metadata
            else {}
        )
        reason_code = diagnostics.get("reason_code", "ok")
        reason_message = diagnostics.get("reason_message", "retrieval_ready")
        explicit_kb_query = self._is_explicit_kb_query(query)

        if reason_code == "all_channels_failed":
            return {
                "action": "fallback_to_chat",
                "status": "Knowledge-base retrieval is currently degraded; falling back to chat behavior.",
            }

        if reason_code in {"no_hits", "low_relevance"}:
            if explicit_kb_query:
                return {
                    "action": "direct_kb_miss",
                    "status": "The knowledge-base retrieval result is insufficient for a grounded answer.",
                    "answer": self._render_kb_miss_answer(reason_message),
                }
            return {
                "action": "fallback_to_chat",
                "status": "No sufficiently relevant KB hits found; falling back to ChatWorker behavior",
            }

        if reason_code in {"partial_channel_failure", "single_channel_recall"}:
            return {
                "action": "proceed_with_warning",
                "status": "Retrieval completed with partial signal quality; grounding checks remain enabled.",
            }

        return {"action": "proceed", "status": ""}

    def _is_explicit_kb_query(self, query: str) -> bool:
        normalized = " ".join((query or "").split())
        return bool(normalized and self._EXPLICIT_KB_QUERY_PATTERN.search(normalized))

    def _render_kb_miss_answer(self, reason_message: str) -> str:
        return (
            "根据当前知识库检索，暂时没有找到足够相关、可直接支撑答案的资料。"
            f"原因：{reason_message}。"
            "你可以换一个更具体的关键词，或者让智能体先用通用能力解释背景。"
        )

    def _apply_rag_guard_action(
        self,
        answer: str,
        guard_report,
        query: str,
        pipeline_result,
    ) -> str:
        if guard_report.grounded:
            return answer

        if guard_report.recommended_action == "fallback_without_kb_claims":
            if self._is_explicit_kb_query(query):
                return self._render_kb_miss_answer(guard_report.reason)
            return (
                f"{answer}\n\n"
                "Note: this answer is not backed by reliable retrieved KB evidence and should be treated as a general response."
            ).strip()

        if guard_report.recommended_action in {
            "regenerate_with_citations",
            "regenerate_with_known_citations",
        }:
            refs = ", ".join(
                citation.ref_id for citation in pipeline_result.context_pack.citations[:3]
            ) or "none"
            return (
                "The system retrieved knowledge-base snippets, but it could not produce a citation-grounded answer safely. "
                f"Retrieved snippet ids: {refs}. Please ask a narrower question or request a source-grounded answer."
            )

        return answer

    def _chunk_text(self, text: str, chunk_size: int = 160) -> List[str]:
        if not text:
            return []
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

class ToolAgentWorker(BaseAgentWorker):
    worker_type = AgentWorkerType.TOOL

    def __init__(self, agent_engine: AgentEngine, context_engine: ContextEngine):
        self.agent_engine = agent_engine
        self.context_engine = context_engine

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        yield {"type": "status", "content": "Supervisor selected ToolAgentWorker"}
        if recovery_state:
            yield {"type": "status", "content": "Resuming interrupted tool-agent task."}
            messages = list(recovery_state.get("messages", []))
            resume_state = recovery_state.get("lifecycle_state", {})
        else:
            context = self.context_engine.build_agent_context(
                query=query,
                history=history,
                use_user_memory=bool(user_profile),
                user_profile=user_profile,
            )
            messages = list(context.messages)
            tool_inventory = self.agent_engine.tool_registry.describe_tools()
            inventory_message = {
                "role": "system",
                "content": (
                    "Registered tools are listed below. Do not invent tools that are not in this list.\n"
                    "For questions about which knowledge bases are connected, use "
                    "`list_knowledge_base_collections` or `list_knowledge_base_files`.\n"
                    "For questions that need evidence from indexed content, use "
                    "`search_knowledge_base`.\n\n"
                    f"{tool_inventory}"
                ),
            }
            if messages and messages[0].get("role") == "system":
                messages.insert(1, inventory_message)
            else:
                messages.insert(0, inventory_message)
            resume_state = None
        yield from self.agent_engine.stream_run(
            messages,
            model_options=model_options,
            resume_state=resume_state,
        )

class OrchestratorWorker(BaseAgentWorker):
    worker_type = AgentWorkerType.ORCHESTRATOR

    def __init__(
        self,
        llm: BaseLLM,
        chat_worker: BaseAgentWorker,
        rag_worker: Optional[BaseAgentWorker],
        tool_worker: BaseAgentWorker,
        task_decomposer: Optional[TaskDecomposer] = None,
    ):
        self.llm = llm
        self.chat_worker = chat_worker
        self.rag_worker = rag_worker
        self.tool_worker = tool_worker
        self.task_decomposer = task_decomposer or TaskDecomposer()

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        recovery_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        yield {"type": "status", "content": "Supervisor selected OrchestratorWorker"}
        if recovery_state:
            yield {"type": "status", "content": "Resuming interrupted orchestration task."}
            plan = MultiAgentPlan.model_validate(recovery_state.get("plan", {}))
            collaborator_results: List[Dict[str, Any]] = list(
                recovery_state.get("collaborator_results", [])
            )
            start_index = int(recovery_state.get("next_task_index", 0))
        else:
            plan = self.task_decomposer.preview(
                query,
                rag_available=self.rag_worker is not None,
            )
            collaborator_results = []
            start_index = 0
        yield {"type": "multi_agent_plan", "data": plan.to_trace_data()}
        yield {
            "type": "task_snapshot",
            "data": {
                "route_worker": self.worker_type.value,
                "status": "running",
                "plan": plan.model_dump(exclude_none=True),
                "collaborator_results": collaborator_results,
                "next_task_index": start_index,
            },
        }

        for index, task in enumerate(plan.tasks[start_index:], start=start_index):
            worker = self._worker_for(task.worker)
            task_query = self._build_task_query(query, task.query, collaborator_results)
            yield {
                "type": "collaborator_trace",
                "data": {
                    "phase": "start",
                    "task_id": task.task_id,
                    "title": task.title,
                    "worker": task.worker.value,
                    "rationale": task.rationale,
                },
            }
            answer_text = ""
            for event in worker.stream(
                query=task_query,
                history=history,
                user_profile=user_profile,
                model_options=model_options,
                recovery_state=None,
            ):
                event_type = event.get("type")
                if event_type == "message_delta":
                    answer_text += event.get("content", "")
                    continue
                if event_type == "final_answer":
                    answer_text = event.get("content", "") or answer_text
                    continue
                if event_type == "finish":
                    finish_messages = event.get("new_messages", [])
                    if not answer_text:
                        answer_text = self._extract_finish_answer(finish_messages)
                    break
                yield event

            collaborator_results.append(
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "worker": task.worker.value,
                    "answer": answer_text.strip(),
                }
            )
            yield {
                "type": "collaborator_trace",
                "data": {
                    "phase": "finish",
                    "task_id": task.task_id,
                    "title": task.title,
                    "worker": task.worker.value,
                    "output_preview": (answer_text or "")[:240],
                },
            }
            yield {
                "type": "task_snapshot",
                "data": {
                    "route_worker": self.worker_type.value,
                    "status": "running",
                    "plan": plan.model_dump(exclude_none=True),
                    "collaborator_results": collaborator_results,
                    "next_task_index": index + 1,
                },
            }

        yield {
            "type": "task_snapshot",
            "data": {
                "route_worker": self.worker_type.value,
                "status": "running",
                "plan": plan.model_dump(exclude_none=True),
                "collaborator_results": collaborator_results,
                "next_task_index": len(plan.tasks),
            },
        }
        synthesis_messages = self._build_synthesis_messages(
            query=query,
            collaborator_results=collaborator_results,
            user_profile=user_profile,
        )
        final_answer = ""
        for chunk in self.llm.chat_stream(synthesis_messages, **self._model_options(model_options)):
            if isinstance(chunk, dict):
                if chunk.get("type") == "reasoning":
                    yield {"type": "reasoning", "content": chunk["content"]}
                else:
                    final_answer += chunk["content"]
                    yield {"type": "message_delta", "content": chunk["content"]}
            else:
                final_answer += chunk
                yield {"type": "message_delta", "content": chunk}

        yield {
            "type": "finish",
            "new_messages": [
                {"role": "assistant", "content": final_answer},
            ],
            "worker": self.worker_type.value,
        }

    def _worker_for(self, worker_type: AgentWorkerType) -> BaseAgentWorker:
        if worker_type == AgentWorkerType.RAG and self.rag_worker is not None:
            return self.rag_worker
        if worker_type == AgentWorkerType.TOOL:
            return self.tool_worker
        return self.chat_worker

    def _build_task_query(
        self,
        original_query: str,
        task_query: str,
        collaborator_results: List[Dict[str, Any]],
    ) -> str:
        if not collaborator_results:
            return task_query or original_query

        findings = "\n".join(
            f"- {item['title']} ({item['worker']}): {item['answer']}"
            for item in collaborator_results
            if item.get("answer")
        )
        if not findings:
            return task_query or original_query

        return (
            f"Original task:\n{original_query}\n\n"
            f"Current subtask:\n{task_query or original_query}\n\n"
            f"Previous collaborator findings:\n{findings}"
        )

    def _extract_finish_answer(self, messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages or []):
            if message.get("role") == "assistant" and message.get("content"):
                return str(message.get("content"))
        return ""

    def _build_synthesis_messages(
        self,
        query: str,
        collaborator_results: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are the final synthesis coordinator for DeepAssist. "
            "Merge collaborator findings into one clear, direct answer. "
            "If collaborators disagree or some evidence is weak, say so explicitly."
        )
        if user_profile:
            system_prompt = f"{system_prompt}\n\n[User Profile]\n{user_profile}"

        findings = "\n\n".join(
            (
                f"[{item['task_id']}] {item['title']} | worker={item['worker']}\n"
                f"{item['answer'] or '(no useful result)'}"
            )
            for item in collaborator_results
        )
        user_prompt = (
            f"Original user task:\n{query}\n\n"
            f"Collaborator findings:\n{findings}\n\n"
            "Produce the final answer for the user. "
            "Do not mention internal worker names unless it improves clarity."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]