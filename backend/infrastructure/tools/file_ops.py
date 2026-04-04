import os
from pathlib import Path
from backend.core.logger import get_logger

logger = get_logger(__name__)

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

def list_sandbox_files(path: str = ".") -> str:
    """
    列出工作区沙箱目录下的所有文件和文件夹。
    :param path: 相对沙箱根目录的子路径，默认为 "." 表示根目录
    """
    try:
        target_path = (SANDBOX_DIR / path).resolve()
        if not str(target_path).startswith(str(SANDBOX_DIR)):
            return "安全拦截：禁止访问工作区之外的目录！"
            
        if not target_path.exists() or not target_path.is_dir():
            return f"目录 {path} 不存在或不是一个文件夹。"
            
        items = []
        for item in target_path.iterdir():
            prefix = "[DIR] " if item.is_dir() else "[FILE]"
            size = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
            items.append(f"{prefix} {item.name}{size}")
            
        if not items:
            return f"目录 {path} 为空。"
            
        return "沙箱目录内容如下：\n" + "\n".join(items)
    except Exception as e:
        return f"列出目录失败: {str(e)}"