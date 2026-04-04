import subprocess
import tempfile
import os
from backend.core.logger import get_logger

logger = get_logger(__name__)

def execute_python_code(code: str) -> str:
    """
    执行 Python 代码片段并返回控制台输出 (stdout/stderr)。
    适用于数学计算、数据处理、算法验证等场景。
    注意：代码在受限沙箱中运行，最长执行时间为 10 秒。
    :param code: 要执行的 Python 代码字符串
    """
    logger.info("🛠️ [Tool] 正在沙箱中执行 Python 代码...")
    
    # 创建一个临时文件来存放代码
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        # 使用 subprocess 启动独立进程执行代码
        # capture_output=True 捕获输出，text=True 返回字符串
        result = subprocess.run(
            ["python", temp_file_path],
            capture_output=True,
            text=True,
            timeout=10.0  # 🚨 核心安全机制：防止死循环 (如 while True)
        )
        
        output = result.stdout
        error = result.stderr
        
        if result.returncode == 0:
            return f"代码执行成功。\n[标准输出]:\n{output if output else '无输出'}"
        else:
            return f"代码执行报错 (Exit Code {result.returncode})。\n[错误信息]:\n{error}"
            
    except subprocess.TimeoutExpired:
        return "【系统拦截】: 代码执行超时 (超过 10 秒)，已被强制终止。请检查是否存在死循环或耗时过长的操作。"
    except Exception as e:
        logger.error(f"Python 沙箱执行异常: {e}")
        return f"沙箱环境异常: {str(e)}"
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)