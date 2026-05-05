import os
import sqlite3
import json
from typing import List, Dict, Any, Optional

from backend.domain.interfaces.memory_db import BaseMemoryStore
from backend.domain.entities.context_window import ContextSummary
from backend.domain.entities.message import Message, AIMessage
from backend.domain.entities.task_snapshot import TaskSnapshot
from backend.common.config import settings
from backend.common.logger import get_logger

logger = get_logger(__name__)


class SQLiteMemoryStore(BaseMemoryStore):
    def __init__(self):
        self.db_path = settings.CONVERSATION_DB_PATH
        base_dir = os.path.dirname(self.db_path)
        if not base_dir:
            base_dir = "."
            
        os.makedirs(base_dir, exist_ok=True)
        
        self._init_db()
        logger.info(f"✅ SQLite 记忆库初始化完成: {self.db_path}")

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 1. 聊天消息表 (已有的)
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
                # 2. 用户画像表 (新增：存储 Key-Value 型的偏好)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS task_snapshots (
                        session_id TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        route_worker TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_summaries (
                        session_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        source_turn_ids TEXT,
                        dropped_turn_count INTEGER NOT NULL,
                        dropped_message_count INTEGER NOT NULL,
                        reason_counts TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                
                # 🚨 致命 Bug 修复：将 ORDER BY created_at 改为 ORDER BY id
                # 因为 id 是 AUTOINCREMENT，绝对保证了插入的物理先后顺序
                cursor.execute("""
                    SELECT role, content, name, tool_call_id, tool_calls 
                    FROM messages 
                    WHERE session_id = ? 
                    ORDER BY id DESC LIMIT ?
                """, (session_id, safe_limit))
                
                rows = cursor.fetchall()
                history =[]
                
                for row in reversed(rows):
                    role, content, name, tool_call_id, tool_calls_json = row
                    
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
                cursor.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM task_snapshots WHERE session_id = ?", (session_id,))
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

    def get_profile(self, key: str) -> Optional[str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM user_profiles WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to read user profile [{key}]: {e}")
            return None

    def set_profile(self, key: str, value: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO user_profiles (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (key, value),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to write user profile [{key}]: {e}")
            return False

    def get_all_profiles(self) -> Dict[str, str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key, value FROM user_profiles ORDER BY key ASC")
                return {key: value for key, value in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to read user profiles: {e}")
            return {}

    def get_task_snapshot(self, session_id: str) -> Optional[TaskSnapshot]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, query, route_worker, status, payload, updated_at
                    FROM task_snapshots
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                payload = json.loads(row[4]) if row[4] else {}
                return TaskSnapshot(
                    session_id=row[0],
                    query=row[1],
                    route_worker=row[2],
                    status=row[3],
                    payload=payload if isinstance(payload, dict) else {},
                    updated_at=row[5],
                )
        except Exception as e:
            logger.error(f"Failed to read task snapshot [{session_id}]: {e}")
            return None

    def save_task_snapshot(self, snapshot: TaskSnapshot) -> bool:
        try:
            payload_json = json.dumps(snapshot.payload, ensure_ascii=False)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO task_snapshots (session_id, query, route_worker, status, payload, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        query = excluded.query,
                        route_worker = excluded.route_worker,
                        status = excluded.status,
                        payload = excluded.payload,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        snapshot.session_id,
                        snapshot.query,
                        snapshot.route_worker,
                        snapshot.status,
                        payload_json,
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save task snapshot [{snapshot.session_id}]: {e}")
            return False

    def clear_task_snapshot(self, session_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM task_snapshots WHERE session_id = ?", (session_id,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to clear task snapshot [{session_id}]: {e}")
            return False

    def get_session_summary(self, session_id: str) -> Optional[ContextSummary]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT content, source, source_turn_ids, dropped_turn_count,
                           dropped_message_count, reason_counts
                    FROM session_summaries
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return ContextSummary(
                    content=row[0],
                    source=row[1] or "persisted",
                    source_turn_ids=json.loads(row[2]) if row[2] else [],
                    dropped_turn_count=int(row[3] or 0),
                    dropped_message_count=int(row[4] or 0),
                    reason_counts=json.loads(row[5]) if row[5] else {},
                )
        except Exception as e:
            logger.error(f"Failed to read session summary [{session_id}]: {e}")
            return None

    def save_session_summary(self, session_id: str, summary: ContextSummary) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO session_summaries (
                        session_id, content, source, source_turn_ids, dropped_turn_count,
                        dropped_message_count, reason_counts, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        content = excluded.content,
                        source = excluded.source,
                        source_turn_ids = excluded.source_turn_ids,
                        dropped_turn_count = excluded.dropped_turn_count,
                        dropped_message_count = excluded.dropped_message_count,
                        reason_counts = excluded.reason_counts,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        session_id,
                        summary.content,
                        summary.source,
                        json.dumps(summary.source_turn_ids, ensure_ascii=False),
                        summary.dropped_turn_count,
                        summary.dropped_message_count,
                        json.dumps(summary.reason_counts, ensure_ascii=False),
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save session summary [{session_id}]: {e}")
            return False

    def clear_session_summary(self, session_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to clear session summary [{session_id}]: {e}")
            return False
