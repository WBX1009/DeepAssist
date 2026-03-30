from abc import ABC, abstractmethod
from typing import List

class BaseEmbedding(ABC):
    """
    文本向量化抽象基类。
    剥离具体模型依赖（不论是 bge-m3 还是 OpenAI Embedding）。
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """将单条文本转为向量 (常用于查询阶段)"""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将批量文本转为向量 (常用于知识库构建阶段)"""
        pass