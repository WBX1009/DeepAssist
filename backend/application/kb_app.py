from backend.services.rag.chunking import DocumentChunker
from backend.domain.interfaces.embedding import BaseEmbedding
from backend.domain.interfaces.vector_db import BaseVectorDB
from backend.domain.interfaces.keyword_db import BaseKeywordDB
from backend.core.logger import get_logger

logger = get_logger(__name__)

class KnowledgeBaseApp:
    """
    知识库管理统筹器 (面向 API 与前端交互)
    处理前端上传文档的解析、切分与双写入库。
    """
    def __init__(self, 
                 chunker: DocumentChunker, 
                 embedding_model: BaseEmbedding,
                 vector_db: BaseVectorDB,
                 keyword_db: BaseKeywordDB):
        self.chunker = chunker
        self.embedding = embedding_model
        self.vector_db = vector_db
        self.keyword_db = keyword_db

    def process_document(self, file_name: str, content: str, collection_name: str = "tech_docs_kb") -> dict:
        logger.info(f"📥 接收到文档处理请求: {file_name}")
        
        # 1. 切分文档
        chunks = self.chunker.split_markdown(content, source_name=file_name)
        if not chunks:
            return {"status": "error", "message": "文档内容过短或切分失败"}
            
        # 2. 批量计算向量
        texts =[c.content for c in chunks]
        embeddings = self.embedding.embed_documents(texts)
        
        # 3. 双写一致性入库
        v_success = self.vector_db.add_chunks(collection_name, chunks, embeddings)
        k_success = self.keyword_db.build_index(collection_name, chunks)
        
        if v_success and k_success:
            logger.info(f"✅ 文档 {file_name} 成功入库，共 {len(chunks)} 个 Chunk。")
            return {"status": "success", "message": f"成功处理 {len(chunks)} 个文本块"}
        else:
            logger.error(f"❌ 文档 {file_name} 入库失败。")
            return {"status": "error", "message": "数据库写入失败"}