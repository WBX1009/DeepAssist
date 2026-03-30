import os
import sqlite3
import json
from typing import List, Dict, Any

from backend.domain.interfaces.memory_db import BaseMemoryStore
from backend.domain.entities.message import Message, AIMessage
from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class SQLiteMemoryStore(BaseMemoryStore):
    def __init__(self):
        base_dir = os.path.dirname(settings.VECTOR_DB_PATH)
        if not base_dir:
            base_dir = "."
            
        os.makedirs(base_dir, exist_ok=True)
        self.db_path = os.path.join(base_dir, "chat_history.db")
        
        self._init_db()
        logger.info(f"✅ SQLite 记忆库初始化完成: {self.db_path}")

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 🚀 扩展表结构：支持工具调用的各种特殊字段
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT,
                        name TEXT,
                        tool_call_id TEXT,
                        tool_calls TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)")
                conn.commit()
        except Exception as e:
            logger.error(f"初始化数据库表失败: {e}")

    def get_history(self, session_id: str, limit: int = 10) -> List[Message]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                safe_limit = limit * 4 
                cursor.execute("""
                    SELECT role, content, name, tool_call_id, tool_calls 
                    FROM messages 
                    WHERE session_id = ? 
                    ORDER BY created_at DESC LIMIT ?
                """, (session_id, safe_limit))
                
                rows = cursor.fetchall()
                history =[]
                
                for row in reversed(rows):
                    role, content, name, tool_call_id, tool_calls_json = row
                    
                    # 🟢 架构优化：直接反序列化为 Pydantic Domain 实体
                    if tool_calls_json:
                        msg = AIMessage(
                            role=role,
                            content=content or "",
                            tool_calls=json.loads(tool_calls_json)
                        )
                    else:
                        msg = Message(
                            role=role,
                            content=content or "",
                            name=name,
                            tool_call_id=tool_call_id
                        )
                    history.append(msg)
                    
                return history
        except Exception as e:
            logger.error(f"读取历史记录失败[{session_id}]: {e}")
            return[]

    def add_message(self, session_id: str, message: Message) -> bool:
        try:
            # 🟢 架构优化：从 Pydantic 实体中提取数据，类型极其安全
            role = message.role
            content = message.content
            name = message.name
            tool_call_id = message.tool_call_id
            
            tool_calls_json = None
            if isinstance(message, AIMessage) and message.tool_calls:
                # model_dump() 会将 Pydantic 对象安全地转为 dict 列表
                tool_calls_json = json.dumps([tc.model_dump() for tc in message.tool_calls], ensure_ascii=False)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO messages (session_id, role, content, name, tool_call_id, tool_calls)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, role, content, name, tool_call_id, tool_calls_json))
                conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"写入历史记录失败 [{session_id}]: {e}")
            return False

    def clear_history(self, session_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"清空历史记录失败: {e}")
            return False

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """获取所有历史会话的列表，按最后活跃时间倒序排列"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 按 session_id 分组，获取最早的一条作为标题，最晚的时间作为排序依据
                cursor.execute("""
                    SELECT session_id, MIN(created_at) as start_time, MAX(created_at) as last_time
                    FROM messages
                    GROUP BY session_id
                    ORDER BY last_time DESC
                """)
                rows = cursor.fetchall()
                
                sessions =[]
                for row in rows:
                    s_id = row[0]
                    last_time = row[2]
                    
                    # 获取该 session 的第一条用户发言作为标题
                    cursor.execute("""
                        SELECT content FROM messages 
                        WHERE session_id = ? AND role = 'user' 
                        ORDER BY created_at ASC LIMIT 1
                    """, (s_id,))
                    title_row = cursor.fetchone()
                    
                    title = "新对话"
                    if title_row and title_row[0]:
                        # 截取前 15 个字符作为展示标题
                        title = title_row[0][:15].replace('\n', ' ')
                        if len(title_row[0]) > 15:
                            title += "..."
                            
                    sessions.append({
                        "session_id": s_id,
                        "title": title,
                        "updated_at": last_time
                    })
                return sessions
        except Exception as e:
            logger.error(f"获取会话列表失败: {e}")
            return[]          