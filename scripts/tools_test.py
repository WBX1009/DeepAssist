import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.api.dependencies import get_agent_app

def main():
    print("🌟 DeepAssist 多跳跨模态测试 (金融域) 🌟\n")
    
    app = get_agent_app()
    
    # 设定一个专门用于测试的 Session ID
    TEST_SESSION_ID = "test_financial_hop_001"
    
    # 【高阶测试 Prompt】：RAG -> Python -> File
    complex_query = """
请作为高级量化分析师，帮我完成以下多步任务：

1. 【概念检索】：使用搜索工具在我们的内部知识库中，查阅并提取“市盈率(PE)”和“市净率(PB)”的准确定义与计算公式。
2. 【沙箱计算】：假设某上市公司当前股价为 85.5 元，最新财报显示其每股收益(EPS)为 3.2 元，每股净资产(BPS)为 14.8 元。请你编写并执行一段 Python 代码，精确计算该公司的 PE 和 PB 值（保留两位小数）。
3. 【报告生成】：将步骤1查到的定义，以及步骤2计算出来的确切数值，整理成一份简明的 Markdown 研报。必须使用工具将报告写入沙箱文件中，文件名为 'financial_analysis_report.md'。

请严格按步骤执行。
    """
    
    print(f"👤[User Query]:\n{complex_query}\n")
    print("=" * 60)
    
    try:
        generator = app.stream_agent_task(
            session_id=TEST_SESSION_ID, 
            query=complex_query, 
            use_user_memory=False
        )
        
        for chunk in generator:
            if chunk.startswith("data: "):
                data_str = chunk[6:].strip()
                if data_str == "[DONE]":
                    print("\n\n✅ 任务流式执行完毕。")
                    break
                try:
                    data = json.loads(data_str)
                    if "content" in data:
                        print(data["content"], end="", flush=True)
                    elif "error" in data:
                        print(f"\n❌ [引擎报错]: {data['error']}")
                except json.JSONDecodeError:
                    pass
    finally:
        # 🧹 测试生命周期收尾：打扫战场，绝不留脏数据！
        print("\n\n" + "=" * 60)
        print("🧹 [生命周期管理] 正在清理测试产生的孤儿数据...")
        app.session_mgr.delete_session(TEST_SESSION_ID)
        print("✅ 测试环境清理完毕。")

if __name__ == "__main__":
    main()