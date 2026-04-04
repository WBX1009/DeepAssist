import sys
import json
from pathlib import Path

# 将项目根目录加入 sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.api.dependencies import get_agent_app

def main():
    print("🌟 DeepAssist 极限压力测试启动 🌟\n")
    
    app = get_agent_app()
    
    # 【深渊级测试 Prompt】：强制 Agent 进行多步跨域协同
    complex_query = """
请帮我完成一份深度架构与运营状态调研报告，要求必须按照以下步骤自动完成：

1. 【全网搜索】：查询“DeepSeek-V3 模型的总参数量和激活参数量分别大约是多少？”。
2. 【数据库查询】：统计我们的业务数据库中 `messages` 表一共有多少条数据记录。
3. 【知识库检索】：查阅内部知识库，了解“如何优化 RAG 的检索效率”。（即使没查到也没关系，只需附上检索结果的结论）。
4. 【Python 计算】：编写一段 Python 代码，假设一条 message 占用 2KB，计算步骤2中查到的总记录数，会占用多少 MB 的存储空间？
5. 【文件写入】：将以上 1 到 4 步收集到的所有真实数据和结论，整理成一份结构清晰的 Markdown 格式报告，保存到沙箱中，文件命名为 'deepseek_arch_report.md'。

请开始你的多步推理！
    """
    
    print(f"👤 [User Query]:\n{complex_query}\n")
    print("=" * 60)
    
    # 调用底层的流式任务
    generator = app.stream_agent_task(
        session_id="stress_test_999", 
        query=complex_query, 
        use_user_memory=False
    )
    
    # 消费并解析 SSE 数据流，在控制台漂亮地打印出来
    for chunk in generator:
        if chunk.startswith("data: "):
            data_str = chunk[6:].strip()
            if data_str == "[DONE]":
                print("\n\n✅ 任务流式执行完毕。")
                break
            try:
                data = json.loads(data_str)
                if "content" in data:
                    # 实时打印 Agent 的心智活动与最终输出
                    print(data["content"], end="", flush=True)
                elif "error" in data:
                    print(f"\n❌ [引擎报错]: {data['error']}")
            except json.JSONDecodeError:
                pass

if __name__ == "__main__":
    main()