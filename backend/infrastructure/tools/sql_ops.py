import sqlite3
from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

def query_business_database(sql_query: str) -> str:
    """
    执行只读 SQL 查询以获取业务数据。
    当前连接的数据库包含以下表结构：
    1. messages (id, session_id, role, content, created_at) - 记录了所有用户的历史聊天
    2. user_profiles (key, value, updated_at) - 记录了用户画像偏好
    :param sql_query: 要执行的 SQLite SELECT 语句
    """
    logger.info(f"🛠️ [Tool] 正在执行 SQL 查询: {sql_query}")
    
    # 🚨 核心安全机制：应用层拦截非查询语句 (防止 Agent 删库跑路)
    sql_upper = sql_query.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "【安全拦截】: 权限不足！该工具仅支持只读查询 (SELECT)。禁止执行 INSERT/UPDATE/DELETE/DROP 等操作。"

    try:
        # 连接到我们现有的聊天记录数据库
        with sqlite3.connect(settings.CONVERSATION_DB_PATH, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # 获取列名
            columns = [description[0] for description in cursor.description] if cursor.description else[]
            rows = cursor.fetchall()
            
            if not rows:
                return "查询执行成功，但结果集为空 (0 rows)。"
                
            # 限制返回行数，防止大模型 Token 爆表
            MAX_ROWS = 50
            is_truncated = len(rows) > MAX_ROWS
            display_rows = rows[:MAX_ROWS]
            
            # 格式化为 Markdown 表格返回给大模型
            header = "| " + " | ".join(columns) + " |"
            separator = "|-" + "-|-".join(["-" * len(c) for c in columns]) + "-|"
            
            body_lines =[]
            for row in display_rows:
                # 将 None 转为空字符串，并处理换行符防止破坏表格
                safe_row =[str(item).replace('\n', ' ') if item is not None else "NULL" for item in row]
                body_lines.append("| " + " | ".join(safe_row) + " |")
                
            result_str = f"查询成功，返回 {len(rows)} 行数据。\n\n"
            result_str += "\n".join([header, separator] + body_lines)
            
            if is_truncated:
                result_str += f"\n\n*(注意：为保护上下文长度，仅展示前 {MAX_ROWS} 行，其余数据已截断)*"
                
            return result_str
            
    except sqlite3.OperationalError as e:
        return f"SQL 语法错误或表不存在: {str(e)}。请检查你的 SQL 语句并重试。"
    except Exception as e:
        logger.error(f"SQL 执行异常: {e}")
        return f"数据库查询失败: {str(e)}"