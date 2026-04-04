from typing import List, Dict, Any
from backend.domain.interfaces.memory_db import BaseMemoryStore
from backend.domain.entities.message import Message, AIMessage
from backend.core.logger import get_logger

logger = get_logger(__name__)

class SessionManager:
    """会话上下文管家"""
    def __init__(self, memory_store: BaseMemoryStore):
        self.store = memory_store

    def get_chat_context(self, session_id: str, max_rounds: int = 5) -> List[Dict[str, Any]]:
        """给 LLM 喂上下文时，将强类型实体转回 OpenAI SDK 需要的 Dict"""
        history_entities = self.store.get_history(session_id, limit=max_rounds)
        # 排除掉 None 值，防止 OpenAI 库严格校验报错
        return[msg.model_dump(exclude_none=True) for msg in history_entities]

    def save_interaction(self, session_id: str, user_query: str, ai_response: str):
        if user_query:
            self.store.add_message(session_id, Message(role="user", content=user_query))
        if ai_response:
            self.store.add_message(session_id, Message(role="assistant", content=ai_response))

    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        """将 engine 返回的 Dict 转换为强类型 Message 并入库"""
        for msg in messages:
            # 动态判断是普通消息还是携带 tool_calls 的复杂消息
            if "tool_calls" in msg and msg["tool_calls"]:
                entity = AIMessage(**msg)
            else:
                entity = Message(**msg)
            self.store.add_message(session_id, entity)
            
        logger.info(f"✅ 已持久化 {len(messages)} 条对话及工具调用轨迹至数据库。")

    def delete_session(self, session_id: str) -> bool:
        """
        生命周期管理：销毁整个会话及其所有关联轨迹
        """
        success = self.store.clear_history(session_id)
        if success:
            logger.info(f"🗑️[Lifecycle] 会话 {session_id} 及其所有对话轨迹已被永久销毁。")
        return success

    def list_sessions(self) -> List[Dict[str, Any]]:
        return self.store.get_all_sessions()