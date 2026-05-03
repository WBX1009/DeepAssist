import sqlite3
import json
import tiktoken  # 官方 Token 计算库 (需 pip install tiktoken)


# ==========================================
# 1. 基础设施层 (Infrastructure): 真实的 SQLite 读写
# 对应项目: infrastructure/databases/sqlite_memory.py
# ==========================================
class SQLiteMemoryStore:
    def __init__(self, db_path=":memory:"):  # 为了方便测试，使用内存数据库
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()   # 光标，游标，操作代理，读写手柄
        # 1. 会话记录表 (永久保存，用于 UI 恢复和长期追溯)
        # 真实项目中你加了 tool_calls 字段防止 400 报错，这里完美还原！
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT 
            )
        """)
        # 2. 用户画像表 (长期记忆)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self.conn.commit()

        # 预设一条长期记忆（模拟前端设置）
        cursor.execute(
            "INSERT OR IGNORE INTO user_profiles (key, value) VALUES ('identity', '资深后端架构师，喜欢看代码')")
        self.conn.commit()

    def add_message(self, session_id: str, role: str, content: str, tool_calls: list = None):
        tc_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        self.conn.cursor().execute(
            "INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?, ?, ?, ?)",
            (session_id, role, content, tc_json)
        )
        self.conn.commit()

    def get_full_history(self, session_id: str) -> list:
        cursor = self.conn.cursor()
        cursor.execute("SELECT role, content, tool_calls FROM messages WHERE session_id = ? ORDER BY id ASC",
                       (session_id,))
        return cursor.fetchall()

    def get_user_profile(self) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM user_profiles")

        profile_dict = {row[0]: row[1] for row in cursor.fetchall()}
        return json.dumps(profile_dict, indent = 2, ensure_ascii = False)
        # return "\n".join([f"{row[0]}: {row[1]}" for row in cursor.fetchall()])


# ==========================================
# 2. 服务层 (Service): 会话管家 (解决 Token 截断痛点)
# 对应项目: services/session/manager.py
# ==========================================
class SessionManager:
    def __init__(self, store: SQLiteMemoryStore, max_tokens: int = 2000):
        self.store = store
        self.max_tokens = max_tokens
        # OpenAI 官方 tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        if not text: return 0
        return len(self.tokenizer.encode(text))

    def get_token_optimized_context(self, session_id: str) -> list:
        """
        🔥 高级算法：按 Token 限制反向截断历史记录！
        """
        raw_history = self.store.get_full_history(session_id)

        context_messages = []
        current_tokens = 0

        # 从最新的一条消息开始，往前倒推
        for row in reversed(raw_history):
            role, content, tc_json = row
            msg = {"role": role, "content": content}
            if tc_json:
                msg["tool_calls"] = json.loads(tc_json)

            # 计算这条消息的 Token (近似计算)
            msg_tokens = self._count_tokens(content) + (50 if tc_json else 0)

            # 🚀 解决痛点 1：一旦达到 Token 阈值，果断截断，放弃更老的记忆
            if current_tokens + msg_tokens > self.max_tokens:
                break

            # 因为是倒推的，所以要插在列表最前面
            context_messages.insert(0, msg)
            current_tokens += msg_tokens

        return context_messages


# ==========================================
# 3. 应用层 (Application): 业务统筹与长短记忆组装
# 对应项目: application/agent_app.py
# ==========================================
class AgentApplication:
    def __init__(self, store: SQLiteMemoryStore, session_mgr: SessionManager):
        self.store = store
        self.session_mgr = session_mgr

    def run_agent_task(self, session_id: str, query: str):
        # 1. 业务逻辑：无论如何，先把用户的最新提问存进全量日志本 (SQLite)
        self.store.add_message(session_id, "user", query)

        # 2. 核心：获取【经过 Token 截断】的短期工作记忆
        messages = self.session_mgr.get_token_optimized_context(session_id)

        # 3. 🚀 解决痛点 2：将【不可被遗忘】的长期偏好，死死钉在 System Prompt 里！
        user_profile = self.store.get_user_profile()
        system_prompt = f"你是一个高级 AI 助理。\n【用户长期偏好】:\n{user_profile}"

        # 将 system 强行塞入对话的最前方
        messages.insert(0, {"role": "system", "content": system_prompt})

        print("======== 最终发送给大模型的 Payload ========")
        print(json.dumps(messages, indent=2, ensure_ascii=False))
        print("============================================\n")


if __name__ == "__main__":
    # 需要先 pip install tiktoken
    db_store = SQLiteMemoryStore()

    # 假设我们设定 Token 上限非常小（模拟很容易爆显存的情况），比如只允许 50 个 Token
    session_mgr = SessionManager(store=db_store, max_tokens=50)
    app = AgentApplication(store=db_store, session_mgr=session_mgr)

    session_id = "session_001"

    print("📝 往日记本里灌入大量历史废话...")
    db_store.add_message(session_id, "user", "今天天气真好，我想出去玩。")
    db_store.add_message(session_id, "assistant", "是的，非常适合郊游！")
    db_store.add_message(session_id, "user", "但我也想写点 Python 代码，比如学学 Fastapi 和 SQLite，这对我很有帮助。")
    db_store.add_message(session_id, "assistant", "学习编程是一个绝佳的选择！你可以先写一个简单的 CRUD 接口。")

    print("🚀 用户发起了新的请求，观察大模型上下文...")
    # 虽然历史记录很长，但因为设定了 max_tokens=50，最前面的废话会被丢弃
    # 但是，System Prompt 里的“长期偏好”绝不会丢失！
    app.run_agent_task(session_id, "你能帮我写一段连接 SQLite 的代码吗？")