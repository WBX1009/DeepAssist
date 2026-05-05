import re
from typing import Any, Dict, Iterator, List, Optional

from backend.common.logger import get_logger
from backend.domain.entities.agent_worker import AgentWorkerType
from backend.domain.interfaces.llm import BaseLLM
from backend.services.agent.engine import AgentEngine
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
    ) -> Iterator[Dict[str, Any]]:
        yield {"type": "status", "content": "Supervisor selected ChatWorker"}
        context = self.context_engine.build_quick_context(
            query=query,
            history=history,
            user_profile=user_profile,
        )
        final_answer = ""
        for chunk in self.llm.chat_stream(context.messages, **self._model_options(model_options)):
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
    _RAG_RELEVANCE_THRESHOLD = 0.35
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

        final_answer = "".join(
            self.llm.chat_stream(context.messages, **self._model_options(model_options))
        )

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

    def _should_fallback_to_chat(self, pipeline_result) -> bool:
        documents = pipeline_result.retrieval_result.documents
        if not documents:
            return True
        return (documents[0].score or 0.0) < self._RAG_RELEVANCE_THRESHOLD

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
    ) -> Iterator[Dict[str, Any]]:
        yield {"type": "status", "content": "Supervisor selected ToolAgentWorker"}
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
        yield from self.agent_engine.stream_run(
            messages,
            model_options=model_options,
        )
