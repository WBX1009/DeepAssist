from typing import List, Dict
from backend.domain.entities.document import DocumentChunk
from backend.domain.interfaces.vector_db import BaseVectorDB
from backend.domain.interfaces.keyword_db import BaseKeywordDB
from backend.domain.interfaces.embedding import BaseEmbedding
from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

class HybridRetriever:
    """
    工业级混合检索服务 (Hybrid Search via RRF)
    依赖注入：只依赖抽象接口，不依赖具体实现（Chroma/Whoosh）
    """
    def __init__(self, 
                 vector_db: BaseVectorDB, 
                 keyword_db: BaseKeywordDB, 
                 embedding_model: BaseEmbedding):
        self.vector_db = vector_db
        self.keyword_db = keyword_db
        self.embedding_model = embedding_model

    def retrieve(self, collection_name: str, query: str, top_k: int = settings.RETRIEVAL_TOP_K) -> List[DocumentChunk]:
        """
        执行双路召回与 RRF 融合打分
        """
        logger.info(f"🔍 开始混合检索: [{query}]")
        
        # 1. 获取向量特征 (Dense)
        query_vector = self.embedding_model.embed_text(query)
        
        # 2. 双路召回 (为了保证融合效果，单路召回的数量通常是最终 top_k 的 2-3 倍)
        candidate_k = top_k * 2
        
        vector_results = []
        keyword_results =[]
        
        if query_vector:
            vector_results = self.vector_db.search(collection_name, query_vector, candidate_k)
            
        if query.strip():
            keyword_results = self.keyword_db.search(collection_name, query, candidate_k)

        # 3. RRF (Reciprocal Rank Fusion) 融合算法
        # 公式: score = 1 / (K + rank)
        rrf_k = settings.RRF_K
        chunk_map: Dict[str, DocumentChunk] = {}
        score_map: Dict[str, float] = {}
        
        # 处理向量路排名
        for rank, chunk in enumerate(vector_results):
            chunk_map[chunk.id] = chunk
            score_map[chunk.id] = score_map.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        # 处理关键词路排名
        for rank, chunk in enumerate(keyword_results):
            chunk_map[chunk.id] = chunk
            score_map[chunk.id] = score_map.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        # 4. 根据 RRF 得分倒序排列
        sorted_ids = sorted(score_map.keys(), key=lambda cid: score_map[cid], reverse=True)
        
        # 5. 组装并返回最终 Top-K 结果
        final_results =[]
        for cid in sorted_ids[:top_k]:
            chunk = chunk_map[cid]
            chunk.score = score_map[cid] # 替换为融合得分
            final_results.append(chunk)
            
        logger.info(f"✅ 检索完成，向量召回 {len(vector_results)} 条，BM25召回 {len(keyword_results)} 条，融合返回 {len(final_results)} 条。")
        return final_results