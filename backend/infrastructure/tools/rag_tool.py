from backend.services.rag.fusion import HybridRetriever
from backend.core.logger import get_logger

logger = get_logger(__name__)

class KnowledgeBaseTool:
    """知识库检索工具包装器"""
    def __init__(self, retriever: HybridRetriever, collection_name: str = "tech_docs_kb"):
        self.retriever = retriever
        self.collection_name = collection_name

    def search(self, query: str) -> str:
        """
        当你需要回答专业技术、系统配置、企业内部文档或复杂业务逻辑时，必须调用此工具。
        :param query: 提炼后的高价值检索关键词（尽量精简，如 "DeepSeek 部署配置"）
        """
        logger.info(f"🛠️ [Tool] Agent 触发知识库检索: {query}")
        try:
            docs = self.retriever.retrieve(self.collection_name, query)
            if not docs:
                return "知识库中未检索到相关内容，请基于通用知识回答或告知用户查无此文。"
                
            # 格式化返回给大模型的观察结果 (Observation)
            context = "\n\n".join([f"[片段 {i+1}]: {d.content}" for i, d in enumerate(docs)])
            return f"检索成功，以下是知识库中的参考内容：\n{context}"
        except Exception as e:
            logger.error(f"RAG 工具执行异常: {e}")
            return f"检索失败，数据库异常: {str(e)}"