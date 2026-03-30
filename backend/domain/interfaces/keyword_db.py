from abc import ABC, abstractmethod
from typing import List
from backend.domain.entities.document import DocumentChunk

class BaseKeywordDB(ABC):
    """关键词搜索引擎抽象基类（如 Whoosh, ElasticSearch, BM25Okapi）"""
    
    @abstractmethod
    def build_index(self, collection_name: str, chunks: List[DocumentChunk]) -> bool:
        """构建/更新关键词索引"""
        pass

    @abstractmethod
    def search(self, collection_name: str, query_text: str, top_k: int) -> List[DocumentChunk]:
        """基于关键词执行全文检索"""
        pass