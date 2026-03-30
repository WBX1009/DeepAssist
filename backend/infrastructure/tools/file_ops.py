import os
from pathlib import Path
from backend.core.logger import get_logger

logger = get_logger(__name__)

# 定义一个沙箱目录，防止 Agent 乱删系统文件
SANDBOX_DIR = Path("./workspace/agent_sandbox").resolve()
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

def read_local_file(file_name: str) -> str:
    """
    读取工作区内的文本文件内容。
    :param file_name: 文件名（如 "config.txt"）
    """
    try:
        target_path = (SANDBOX_DIR / file_name).resolve()
        if not str(target_path).startswith(str(SANDBOX_DIR)):
            return "安全拦截：禁止访问工作区之外的文件！"
            
        if not target_path.exists():
            return f"文件 {file_name} 不存在。"
            
        return target_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"读取文件失败: {str(e)}"

def write_local_file(file_name: str, content: str) -> str:
    """
    将内容写入工作区的文件中（覆盖写入）。
    :param file_name: 文件名
    :param content: 要写入的文本内容
    """
    try:
        target_path = (SANDBOX_DIR / file_name).resolve()
        if not str(target_path).startswith(str(SANDBOX_DIR)):
            return "安全拦截：禁止访问工作区之外的文件！"
            
        target_path.write_text(content, encoding="utf-8")
        logger.info(f"🛠️ [Tool] Agent 写入文件: {file_name}")
        return f"成功将内容写入 {file_name}"
    except Exception as e:
        return f"写入文件失败: {str(e)}"