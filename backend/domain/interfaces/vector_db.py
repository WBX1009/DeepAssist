from abc import ABC, abstractmethod
from typing import List
from backend.domain.entities.document import DocumentChunk

class BaseVectorDB(ABC):
    """向量数据库抽象基类（如 Chroma, Milvus, pgvector）"""
    
    @abstractmethod
    def add_chunks(self, collection_name: str, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> bool:
        """将带向量的文档块写入库中"""
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], top_k: int) -> List[DocumentChunk]:
        """根据向量进行相似度检索"""
        pass