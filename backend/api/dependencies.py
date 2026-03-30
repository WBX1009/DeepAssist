from functools import lru_cache
import os

from backend.infrastructure.llms.deepseek_client import DeepSeekClient
from backend.infrastructure.embeddings.bge_m3_local import BGEM3Local
from backend.infrastructure.databases.chroma_store import ChromaStore
from backend.infrastructure.databases.whoosh_store import WhooshStore
from backend.infrastructure.databases.sqlite_memory import SQLiteMemoryStore

from backend.infrastructure.tools.rag_tool import KnowledgeBaseTool
from backend.infrastructure.tools.file_ops import read_local_file, write_local_file
from backend.infrastructure.tools.weather_ops import get_weather
from backend.infrastructure.tools.email_ops import send_email

from backend.services.session.manager import SessionManager
from backend.services.rag.fusion import HybridRetriever
from backend.services.rag.chunking import DocumentChunker
from backend.services.agent.engine import AgentEngine

from backend.application.chat_app import ChatApplication
from backend.application.agent_app import AgentApplication
from backend.application.kb_app import KnowledgeBaseApp
from backend.core.logger import get_logger

logger = get_logger(__name__)

@lru_cache()
def get_llm():
    return DeepSeekClient()

@lru_cache()
def get_embedding_model():
    try:
        return BGEM3Local()
    except Exception as e:
        logger.warning(f"⚠️ Embedding 模型加载失败: {e}")
        return None

@lru_cache()
def get_vector_db():
    try:
        return ChromaStore()
    except Exception as e:
        logger.warning(f"⚠️ ChromaDB 加载失败: {e}")
        return None

@lru_cache()
def get_keyword_db():
    try:
        return WhooshStore()
    except Exception as e:
        logger.warning(f"⚠️ Whoosh 加载失败: {e}")
        return None

@lru_cache()
def get_memory_store():
    return SQLiteMemoryStore()

@lru_cache()
def get_retriever():
    embed_model = get_embedding_model()
    v_db = get_vector_db()
    k_db = get_keyword_db()
    
    if not embed_model or not v_db or not k_db:
        logger.warning("⚠️ 检索核心组件缺失，RAG 功能底层实例初始化为空。")
        return None
        
    return HybridRetriever(
        vector_db=v_db,
        keyword_db=k_db,
        embedding_model=embed_model
    )

@lru_cache()
def get_session_manager():
    return SessionManager(memory_store=get_memory_store())

def get_chat_app() -> ChatApplication:
    return ChatApplication(
        llm=get_llm(),
        session_manager=get_session_manager(),
        retriever=get_retriever()
    )

def get_agent_app() -> AgentApplication:
    tools =[read_local_file, write_local_file, get_weather, send_email]
    
    # 动态装载：如果 Retriever 存在，才给 Agent 装备知识库工具
    retriever = get_retriever()
    if retriever:
        rag_tool_instance = KnowledgeBaseTool(retriever=retriever)
        tools.append(rag_tool_instance.search)
        
    engine = AgentEngine(llm=get_llm(), tools=tools)
    return AgentApplication(agent_engine=engine, session_manager=get_session_manager())

def get_kb_app() -> KnowledgeBaseApp:
    return KnowledgeBaseApp(
        chunker=DocumentChunker(),
        embedding_model=get_embedding_model(),
        vector_db=get_vector_db(),
        keyword_db=get_keyword_db()
    )