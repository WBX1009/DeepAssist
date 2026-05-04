from typing import List, Dict, Any
from backend.common.event_bus import event_bus
from backend.domain.entities.tooling import ToolCall
from backend.domain.interfaces.memory_db import BaseMemoryStore
from backend.domain.entities.message import Message, AIMessage
from backend.common.logger import get_logger
from backend.services.session.context_window_manager import PriorityContextWindowManager

logger = get_logger(__name__)

class SessionManager:
    """会话上下文管家"""
    def __init__(
        self,
        memory_store: BaseMemoryStore,
        context_window_manager: PriorityContextWindowManager | None = None,
    ):
        self.store = memory_store
        self.context_window_manager = context_window_manager or PriorityContextWindowManager()

    def get_chat_context(self, session_id: str, max_rounds: int = 5) -> List[Dict[str, Any]]:
        history_entities = self.store.get_history(
            session_id,
            limit=max(max_rounds * 3, max_rounds),
        )
        raw_messages =[msg.model_dump(exclude_none=True) for msg in history_entities]
        raw_messages = [self._normalize_message_for_llm(msg) for msg in raw_messages]

        # ==========================================
        # 🛡️ 终极防御：基于状态机的严格历史清洗算法
        # 确保 OpenAI 接收到的上下文 100% 遵守 tool_calls 协议
        # ==========================================
        valid_messages =[]
        expected_tool_call_ids = set()

        for msg in raw_messages:
            role = msg.get("role")

            if role in ["user", "system"]:
                # 出现用户消息，说明前一轮结束。若此时还有未闭合的 tool_call，说明是残缺的崩溃轮次，直接回滚丢弃！
                if expected_tool_call_ids:
                    while valid_messages and (valid_messages[-1].get("role") == "tool" or "tool_calls" in valid_messages[-1]):
                        valid_messages.pop()
                    expected_tool_call_ids.clear()
                valid_messages.append(msg)

            elif role == "assistant":
                if expected_tool_call_ids:
                    # 发现连续的 assistant 且上一次的 tool 没返回，回滚上一个异常链
                    while valid_messages and (valid_messages[-1].get("role") == "tool" or "tool_calls" in valid_messages[-1]):
                        valid_messages.pop()
                    expected_tool_call_ids.clear()

                valid_messages.append(msg)
                # 记录这句 assistant 发起了哪些工具调用
                if "tool_calls" in msg and msg["tool_calls"]:
                    expected_tool_call_ids = {tc["id"] for tc in msg["tool_calls"]}

            elif role == "tool":
                # 如果当前根本没在等工具返回（孤儿工具结果），直接丢弃
                if not expected_tool_call_ids:
                    continue

                tool_id = msg.get("tool_call_id")
                # 如果这个工具是期望中的一个，保留它并从等待清单划掉
                if tool_id in expected_tool_call_ids:
                    valid_messages.append(msg)
                    expected_tool_call_ids.remove(tool_id)
                else:
                    continue # 乱入的无效工具结果，丢弃

        # 尾部兜底：如果整个历史记录的最后一条是残缺的工具调用，回滚删除它，防止污染接下来的提问
        if expected_tool_call_ids:
             while valid_messages and (valid_messages[-1].get("role") == "tool" or "tool_calls" in valid_messages[-1]):
                 valid_messages.pop()

        context_budget = max(1, max_rounds)
        trimmed_messages = self.context_window_manager.trim(
            valid_messages,
            budget=context_budget,
        )
        logger.debug(
            "Context window selected %s/%s messages for session %s with budget=%s",
            len(trimmed_messages),
            len(valid_messages),
            session_id,
            context_budget,
        )
        return trimmed_messages

    def save_interaction(self, session_id: str, user_query: str, ai_response: str):
        messages: List[Dict[str, Any]] = []
        if user_query:
            self.store.add_message(session_id, Message(role="user", content=user_query))
            messages.append({"role": "user", "content": user_query})
        if ai_response:
            self.store.add_message(session_id, Message(role="assistant", content=ai_response))
            messages.append({"role": "assistant", "content": ai_response})
        if messages:
            event_bus.publish(
                "conversation.completed",
                {
                    "session_id": session_id,
                    "messages": messages,
                    "user_query": user_query,
                    "assistant_response": ai_response,
                },
            )

    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        """将 engine 返回的 Dict 转换为强类型 Message 并入库"""
        for msg in messages:
            # 动态判断是普通消息还是携带 tool_calls 的复杂消息
            if "tool_calls" in msg and msg["tool_calls"]:
                entity = AIMessage(**msg)
            else:
                entity = Message(**msg)
            self.store.add_message(session_id, entity)

        if messages:
            event_bus.publish(
                "conversation.completed",
                {
                    "session_id": session_id,
                    "messages": messages,
                },
            )
            
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

    def _normalize_message_for_llm(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            msg = dict(msg)
            msg["tool_calls"] = [
                ToolCall.model_validate(tool_call).to_openai_tool_call()
                for tool_call in msg["tool_calls"]
            ]
        return msg
