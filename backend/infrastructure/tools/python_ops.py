import subprocess
import tempfile
import os
import ast
from backend.common.logger import get_logger

logger = get_logger(__name__)

def execute_python_code(code: str) -> str:
    """
    在沙箱中执行用户提供的 Python 代码。为了最大限度保护宿主机安全，
    这里在真正执行前会对代码做静态分析，拦截常见的危险语句，例如 import、
    调用系统库、打开文件等。只有在检测通过后，才会启动独立子进程运行代码，
    并限制最大执行时间为 10 秒。
    """
    logger.info("🛠️ [Tool] 正在沙箱中执行 Python 代码...")

    def _is_code_safe(code_str: str) -> str | None:
        """增强的静态分析检查，彻底封锁内置方法反射与魔法属性逃逸。"""
        try:
            tree = ast.parse(code_str, mode="exec")
        except SyntaxError as e:
            return f"代码语法错误: {str(e)}"
        
        # 🌟 安全升级：囊括高级内建函数、反射模块、模块抓取关键词
        banned_names = {
            "os", "sys", "subprocess", "socket", "shutil", "requests",
            "importlib", "types", "inspect",
            "globals", "locals", "vars", "dir", "help", "repr",
            "open", "eval", "exec", "__import__", "compile", "input",
            "__builtins__", "getattr", "setattr", "delattr",
            "type", "object",
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
            
            if isinstance(node, ast.Attribute):
                # 🌟 安全升级：彻底封杀形如 __subclasses__、__class__、__globals__ 等用于越狱的魔法属性
                attr_name = getattr(node, 'attr', '')
                if attr_name.startswith('__') and attr_name.endswith('__'):
                    return f"【安全拦截】: 禁止访问 Python 魔法属性 `{attr_name}`。"
                if attr_name in banned_names:
                    return f"【安全拦截】: 禁止访问危险属性 `{attr_name}`。"
                    
            if isinstance(node, ast.Name) and node.id in banned_names:
                return f"【安全拦截】: 禁止使用危险名称 `{node.id}`。"
                
        return None

    unsafe_reason = _is_code_safe(code)
    if unsafe_reason:
        return unsafe_reason

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