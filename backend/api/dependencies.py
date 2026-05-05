from functools import lru_cache
from typing import Any, Callable, List, Optional

from backend.application.agent_app import AgentApplication
from backend.application.chat_app import ChatApplication
from backend.application.kb_app import KnowledgeBaseApp
from backend.application.runtime_app import RuntimeApplication
from backend.common.event_bus import event_bus
from backend.common.logger import get_logger
from backend.domain.interfaces.embedding import BaseEmbedding
from backend.domain.interfaces.keyword_db import BaseKeywordDB
from backend.domain.interfaces.llm import BaseLLM
from backend.domain.interfaces.memory_db import BaseMemoryStore
from backend.domain.interfaces.vector_db import BaseVectorDB
from backend.infrastructure.databases.chroma_store import ChromaStore
from backend.infrastructure.databases.sqlite_memory import SQLiteMemoryStore
from backend.infrastructure.databases.vector_index_health import VectorIndexHealthInspector
from backend.infrastructure.databases.whoosh_store import WhooshStore
from backend.infrastructure.embeddings.bge_m3_local import BGEM3Local
from backend.infrastructure.llms.deepseek_client import DeepSeekClient
from backend.infrastructure.tools.file_ops import (
    list_sandbox_files,
    read_local_file,
    write_local_file,
)
from backend.infrastructure.tools.kb_catalog_tool import KnowledgeBaseCatalogTool
from backend.infrastructure.tools.python_ops import execute_python_code
from backend.infrastructure.tools.rag_tool import KnowledgeBaseTool
from backend.infrastructure.tools.sql_ops import query_business_database
from backend.infrastructure.tools.weather_ops import get_weather
from backend.services.agent.engine import AgentEngine
from backend.services.agent.intent_router import IntentRouter
from backend.services.agent.supervisor import AgentSupervisor
from backend.services.agent.tooling import ToolPolicy, ToolRegistry
from backend.services.agent.task_decomposer import TaskDecomposer
from backend.services.agent.workers import (
    ChatWorker,
    OrchestratorWorker,
    RAGWorker,
    ToolAgentWorker,
)
from backend.services.context_engine import ContextEngine
from backend.services.profile_extractor import ProfileExtractor
from backend.services.rag.answer_guard import SourceAwareResponseGuard
from backend.services.rag.chunking import DocumentChunker
from backend.services.rag.context_packer import ContextPacker
from backend.services.rag.fusion import HybridRetriever
from backend.services.rag.pipeline import RAGPipeline
from backend.services.rag.query_planner import QueryPlanner
from backend.services.rag.query_rewriter import QueryRewriteService
from backend.services.rag.reranker import LexicalOverlapReranker
from backend.services.session.context_window_manager import PriorityContextWindowManager
from backend.services.session.long_term_memory_recall import LongTermMemoryRecallService
from backend.services.session.manager import SessionManager
from backend.services.session.summary_compressor import ConversationSummaryCompressor
from backend.common.config import settings

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Infrastructure adapters
# -----------------------------------------------------------------------------


@lru_cache()
def get_llm() -> BaseLLM:
    return DeepSeekClient()


@lru_cache()
def get_embedding_model() -> Optional[BaseEmbedding]:
    try:
        return BGEM3Local()
    except Exception as exc:
        logger.warning("Embedding model initialization failed: %s", exc)
        return None


@lru_cache()
def get_vector_db() -> Optional[BaseVectorDB]:
    try:
        return ChromaStore()
    except Exception as exc:
        logger.warning("Vector DB initialization failed: %s", exc)
        return None


@lru_cache()
def get_keyword_db() -> Optional[BaseKeywordDB]:
    try:
        return WhooshStore()
    except Exception as exc:
        logger.warning("Keyword DB initialization failed: %s", exc)
        return None


@lru_cache()
def get_memory_store() -> BaseMemoryStore:
    return SQLiteMemoryStore()


@lru_cache()
def get_vector_index_health_inspector() -> VectorIndexHealthInspector:
    return VectorIndexHealthInspector(
        vector_db_path=settings.VECTOR_DB_PATH,
        keyword_db_path=settings.KEYWORD_DB_PATH,
        report_path=settings.VECTOR_HEALTH_REPORT_PATH,
        embedding_model=get_embedding_model(),
    )


# -----------------------------------------------------------------------------
# RAG service graph
# -----------------------------------------------------------------------------


@lru_cache()
def get_reranker() -> LexicalOverlapReranker:
    return LexicalOverlapReranker()


@lru_cache()
def get_context_packer() -> ContextPacker:
    return ContextPacker()


@lru_cache()
def get_answer_guard() -> SourceAwareResponseGuard:
    return SourceAwareResponseGuard()


@lru_cache()
def get_query_rewriter() -> QueryRewriteService:
    return QueryRewriteService()


@lru_cache()
def get_query_planner() -> QueryPlanner:
    return QueryPlanner(query_rewriter=get_query_rewriter())


@lru_cache()
def get_retriever() -> Optional[HybridRetriever]:
    embedding_model = get_embedding_model()
    vector_db = get_vector_db()
    keyword_db = get_keyword_db()

    if not embedding_model or not vector_db or not keyword_db:
        logger.warning(
            "RAG retriever is unavailable because one or more adapters failed to initialize."
        )
        return None

    return HybridRetriever(
        vector_db=vector_db,
        keyword_db=keyword_db,
        embedding_model=embedding_model,
        query_planner=get_query_planner(),
        reranker=get_reranker(),
    )


@lru_cache()
def get_rag_pipeline() -> Optional[RAGPipeline]:
    retriever = get_retriever()
    if retriever is None:
        return None

    return RAGPipeline(
        retriever=retriever,
        context_packer=get_context_packer(),
        answer_guard=get_answer_guard(),
    )


# -----------------------------------------------------------------------------
# Core services
# -----------------------------------------------------------------------------


@lru_cache()
def get_session_manager() -> SessionManager:
    return SessionManager(
        memory_store=get_memory_store(),
        context_window_manager=get_context_window_manager(),
        summary_compressor=get_summary_compressor(),
        memory_recall=get_long_term_memory_recall_service(),
    )


@lru_cache()
def get_context_window_manager() -> PriorityContextWindowManager:
    return PriorityContextWindowManager()


@lru_cache()
def get_summary_compressor() -> ConversationSummaryCompressor:
    return ConversationSummaryCompressor()


@lru_cache()
def get_long_term_memory_recall_service() -> LongTermMemoryRecallService:
    return LongTermMemoryRecallService(get_memory_store())


@lru_cache()
def get_context_engine() -> ContextEngine:
    return ContextEngine()


@lru_cache()
def get_intent_router() -> IntentRouter:
    return IntentRouter()


@lru_cache()
def get_task_decomposer() -> TaskDecomposer:
    return TaskDecomposer()


@lru_cache()
def get_profile_extractor() -> ProfileExtractor:
    extractor = ProfileExtractor(memory_store=get_memory_store())
    event_bus.subscribe(
        "conversation.completed",
        extractor.handle_conversation_completed,
    )
    return extractor


# -----------------------------------------------------------------------------
# Agent assembly
# -----------------------------------------------------------------------------


def _build_agent_tools(
    retriever: Optional[HybridRetriever],
    rag_pipeline: Optional[RAGPipeline],
    kb_app: KnowledgeBaseApp,
) -> List[Callable[..., Any]]:
    tools: List[Callable[..., Any]] = [
        read_local_file,
        write_local_file,
        list_sandbox_files,
        get_weather,
        execute_python_code,
        query_business_database,
    ]
    kb_catalog_tool = KnowledgeBaseCatalogTool(kb_app)
    tools.extend(
        [
            kb_catalog_tool.list_knowledge_base_collections,
            kb_catalog_tool.list_knowledge_base_files,
        ]
    )

    if retriever is not None:
        tools.append(
            KnowledgeBaseTool(
                retriever=retriever,
                rag_pipeline=rag_pipeline,
                collection_name="__all__",
            ).search_knowledge_base
        )

    return tools


@lru_cache()
def get_tool_registry() -> ToolRegistry:
    return ToolRegistry.from_callables(
        tools=_build_agent_tools(get_retriever(), get_rag_pipeline(), get_kb_app()),
        policy=ToolPolicy(max_result_chars=4000),
    )


@lru_cache()
def get_agent_engine() -> AgentEngine:
    return AgentEngine(
        llm=get_llm(),
        tool_registry=get_tool_registry(),
    )


@lru_cache()
def get_agent_supervisor() -> AgentSupervisor:
    llm = get_llm()
    context_engine = get_context_engine()
    rag_pipeline = get_rag_pipeline()
    chat_worker = ChatWorker(llm=llm, context_engine=context_engine)
    rag_worker = (
        RAGWorker(
            llm=llm,
            context_engine=context_engine,
            rag_pipeline=rag_pipeline,
            collection_name="__all__",
        )
        if rag_pipeline is not None
        else None
    )
    tool_worker = ToolAgentWorker(
        agent_engine=get_agent_engine(),
        context_engine=context_engine,
    )

    return AgentSupervisor(
        intent_router=get_intent_router(),
        chat_worker=chat_worker,
        rag_worker=rag_worker,
        tool_worker=tool_worker,
        orchestrator_worker=OrchestratorWorker(
            llm=llm,
            chat_worker=chat_worker,
            rag_worker=rag_worker,
            tool_worker=tool_worker,
            task_decomposer=get_task_decomposer(),
        ),
        task_decomposer=get_task_decomposer(),
    )


# -----------------------------------------------------------------------------
# Application workflows
# -----------------------------------------------------------------------------


@lru_cache()
def get_chat_app() -> ChatApplication:
    get_profile_extractor()
    return ChatApplication(
        llm=get_llm(),
        session_manager=get_session_manager(),
        context_engine=get_context_engine(),
        intent_router=get_intent_router(),
        profile_extractor=get_profile_extractor(),
        retriever=get_retriever(),
        rag_pipeline=get_rag_pipeline(),
    )


@lru_cache()
def get_agent_app() -> AgentApplication:
    profile_extractor = get_profile_extractor()
    return AgentApplication(
        agent_engine=get_agent_engine(),
        session_manager=get_session_manager(),
        context_engine=get_context_engine(),
        profile_extractor=profile_extractor,
        supervisor=get_agent_supervisor(),
    )


@lru_cache()
def get_kb_app() -> KnowledgeBaseApp:
    return KnowledgeBaseApp(
        chunker=DocumentChunker(),
        embedding_model=get_embedding_model(),
        vector_db=get_vector_db(),
        keyword_db=get_keyword_db(),
        health_inspector=get_vector_index_health_inspector(),
    )


@lru_cache()
def get_runtime_app() -> RuntimeApplication:
    return RuntimeApplication(
        kb_app=get_kb_app(),
        tool_registry=get_tool_registry(),
    )
