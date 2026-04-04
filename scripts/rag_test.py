import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中，以便正确导入 backend
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# 覆盖环境变量，强制离线加载 BGE-M3 模型
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from backend.core.config import settings
from backend.core.logger import get_logger

# 导入底层基础设施
from backend.infrastructure.embeddings.bge_m3_local import BGEM3Local
from backend.infrastructure.databases.chroma_store import ChromaStore
from backend.infrastructure.databases.whoosh_store import WhooshStore
from backend.infrastructure.llms.deepseek_client import DeepSeekClient

# 导入检索服务
from backend.services.rag.fusion import HybridRetriever

logger = get_logger("Local_RAG_Test")

def main():
    print("="*60)
    print("🚀 DeepAssist V2 - 本地四大知识库 RAG 质量测试启动")
    print("="*60)

    # 1. 初始化所有底层组件
    logger.info("⏳ 正在加载离线 Embedding 模型与本地双库...")
    embedding_model = BGEM3Local()
    vector_db = ChromaStore()
    keyword_db = WhooshStore()
    
    # 2. 初始化混合检索器
    retriever = HybridRetriever(
        vector_db=vector_db, 
        keyword_db=keyword_db, 
        embedding_model=embedding_model
    )

    # 3. 初始化 DeepSeek 客户端 (强制指定使用 Reasoner 模型)
    reasoner_llm = DeepSeekClient(model_name=settings.LLM_REASONER_MODEL)
    
    # 我们希望多召回一些数据给 Reasoner 看
    TEST_TOP_K = 10 

    # 4. 构造针对四大垂直知识库的测试用例
    # 你可以根据你实际放入的数据集内容，修改这里的提问
    test_cases =[
        {"kb": "medical_kb", "query": "患者长期头痛并伴有视力模糊和恶心，可能是什么疾病？应该做哪些检查？"},
        {"kb": "legal_kb", "query": "如果在租赁合同中没有明确约定违约金比例，单方提前退租在法律上应如何赔偿？"},
        {"kb": "financial_kb", "query": "公司资产负债率突然从 40% 飙升到 75%，主要可能面临哪些财务风险？"},
        {"kb": "enron_kb", "query": "What were the main concerns about Enron's off-balance-sheet partnerships (like LJM) in the emails?"}
    ]

    for idx, case in enumerate(test_cases):
        kb_name = case["kb"]
        query = case["query"]
        
        print(f"\n\n" + "★"*60)
        print(f"🧪 测试用例 {idx+1}: 知识库 [{kb_name}]")
        print(f"👤 用户提问: {query}")
        print("★"*60)

        # ==========================================
        # 步骤 A：执行混合检索 (多召回)
        # ==========================================
        logger.info(f"🔍 正在从 {kb_name} 中检索 Top-{TEST_TOP_K} 相关的知识块...")
        retrieved_docs = retriever.retrieve(kb_name, query, top_k=TEST_TOP_K)
        
        if not retrieved_docs:
            logger.warning(f"⚠️ 知识库 {kb_name} 中未检索到任何内容，请检查数据是否成功入库！")
            continue
            
        print("\n📚 [检索质量验证] - 召回的前 3 个最高分 Chunk 预览:")
        for i, doc in enumerate(retrieved_docs[:3]):
            score = getattr(doc, 'score', 'N/A')
            # 截断展示，防止刷屏
            preview = doc.content[:150].replace('\n', ' ') + "..." 
            print(f"[{i+1}] 得分: {score:.4f} | 来源: {doc.metadata.get('source', '未知')} | 内容: {preview}")

        # ==========================================
        # 步骤 B：组装 Prompt 喂给 Reasoner
        # ==========================================
        context_str = "\n\n".join([f"【参考资料 {i+1}】\n{d.content}" for i, d in enumerate(retrieved_docs)])
        
        system_prompt = """你是一个严谨的垂直领域专家。
请必须基于我提供的【参考资料】来回答用户的提问。
请先在心里仔细推理这些资料的关联性，然后给出极其专业、结构清晰的最终回答。
如果参考资料不足以回答问题，请直接说明。"""

        final_prompt = f"=== 参考资料 ===\n{context_str}\n\n=== 用户提问 ===\n{query}"
        
        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ]

        # ==========================================
        # 步骤 C：流式输出 Reasoner 的思考与回答
        # ==========================================
        print("\n🧠 [DeepSeek-Reasoner 正在深度思考]...")
        
        is_thinking = False
        try:
            # 调用我们之前封装好的 chat_stream
            for chunk in reasoner_llm.chat_stream(messages):
                # 遇到我们自己封装的思维链标记时，改变打印颜色或格式
                if chunk == "\n```thought\n":
                    is_thinking = True
                    # 打印灰色的思考过程 (ANSI转义码)
                    print("\033[90m", end="") 
                    continue
                elif chunk == "\n```\n\n" and is_thinking:
                    is_thinking = False
                    # 恢复默认颜色，并打印回答分隔线
                    print("\033[0m") 
                    print("\n💡 [最终回答]:")
                    continue
                
                # 实时打印 Token
                print(chunk, end="", flush=True)
                
        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            
        print("\n" + "-"*60)

if __name__ == "__main__":
    main()