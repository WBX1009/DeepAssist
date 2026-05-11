import subprocess
import tempfile
import os
import ast
from backend.common.logger import get_logger

logger = get_logger(__name__)

def execute_python_code(code: str) -> str:
    logger.info("🛠️ [Tool] 正在沙箱中执行 Python 代码...")

    def _is_code_safe(code_str: str) -> str | None:
        try:
            tree = ast.parse(code_str, mode="exec")
        except SyntaxError as e:
            return f"代码语法错误: {str(e)}"
        
        # 🚀 修复点：增加对 __builtins__ 和 eval/exec 等高危内置方法的屏蔽
        banned_names = {
            "os", "sys", "subprocess", "socket", "shutil", "requests",
            "open", "eval", "exec", "__import__", "compile", "input",
            "__builtins__", "getattr", "setattr", "delattr"
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "【安全拦截】: 禁止使用 import 语句。"
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in banned_names:
                    return f"【安全拦截】: 禁止调用危险函数 `{func.id}`。"
                if isinstance(func, ast.Attribute) and func.attr in banned_names:
                    return f"【安全拦截】: 禁止调用危险属性 `{func.attr}`。"
            if isinstance(node, ast.Attribute) and getattr(node, 'attr', None) in banned_names:
                return f"【安全拦截】: 禁止访问危险属性 `{node.attr}`。"
            if isinstance(node, ast.Name) and node.id in banned_names:
                return f"【安全拦截】: 禁止使用危险名称 `{node.id}`。"
        return None

    unsafe_reason = _is_code_safe(code)
    if unsafe_reason:
        return unsafe_reason

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            ["python", temp_file_path],
            capture_output=True,
            text=True,
            timeout=10.0
        )
        if result.returncode == 0:
            return f"代码执行成功。\n[标准输出]:\n{result.stdout if result.stdout else '无输出'}"
        else:
            return f"代码执行报错 (Exit Code {result.returncode})。\n[错误信息]:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "【系统拦截】: 代码执行超时 (超过 10 秒)，已被强制终止。"
    except Exception as e:
        logger.error(f"Python 沙箱执行异常: {e}")
        return f"沙箱环境异常: {str(e)}"
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)