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

        if self._should_fallback_to_chat(pipeline_result):
            yield {
                "type": "status",
                "content": "No sufficiently relevant KB hits found; falling back to ChatWorker behavior",
            }
            context = self.context_engine.build_quick_context(
                query=query,
                history=history,
                user_profile=user_profile,
            )

        final_answer = ""
        for chunk in self.llm.chat_stream(context.messages, **self._model_options(model_options)):
            final_answer += chunk
            yield {"type": "message_delta", "content": chunk}

        guard_report = self.rag_pipeline.check_answer(final_answer, pipeline_result)
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
        yield from self.agent_engine.stream_run(
            context.messages,
            model_options=model_options,
        )
