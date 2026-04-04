import os
import json
from typing import List, Dict, Any
import chromadb

from backend.domain.interfaces.vector_db import BaseVectorDB
from backend.domain.entities.document import DocumentChunk
from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

class ChromaStore(BaseVectorDB):
    """
    ChromaDB 向量库底层实现。
    完全遵守 domain/interfaces/vector_db.py 的契约。
    """
    def __init__(self):
        db_path = settings.VECTOR_DB_PATH
        os.makedirs(db_path, exist_ok=True)
        
        # 强制使用本地持久化客户端
        self.client = chromadb.PersistentClient(path=db_path)
        logger.info(f"✅ ChromaDB 本地持久化客户端初始化完成: {db_path}")

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        🛡️ 核心修复：清洗元数据，防止 ChromaDB 报错。
        Chroma 的 metadata 值仅支持: str, int, float, bool
        禁止包含 dict, list 或 None。
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                continue  # 丢弃 None 值
                
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                # 将列表和字典序列化为 JSON 字符串
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            else:
                # 其他未知类型，强转为字符串兜底
                sanitized[key] = str(value)
                
        return sanitized

    def add_chunks(self, collection_name: str, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> bool:
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            
            ids = []
            documents =[]
            metadatas =[]
            
            for chunk in chunks:
                ids.append(chunk.id)
                documents.append(chunk.content)
                # 🚀 写入前必须经过严格清洗
                clean_meta = self._sanitize_metadata(chunk.metadata)
                metadatas.append(clean_meta)
            
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"成功向 Chroma 集合 '{collection_name}' 写入 {len(chunks)} 条向量数据")
            return True
        except Exception as e:
            logger.error(f"ChromaDB 写入失败: {e}")
            return False

    def search(self, collection_name: str, query_vector: List[float], top_k: int) -> List[DocumentChunk]:
        try:
            collection = self.client.get_collection(name=collection_name)
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            chunks = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    
                    # 取出原始 metadata
                    raw_meta = results['metadatas'][0][i] or {}
                    
                    # （可选）在这里可以尝试将 JSON 字符串还原回 dict/list，
                    # 但为了保证检索效率，通常原样返回给 LLM 即可。
                    
                    chunks.append(DocumentChunk(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        metadata=raw_meta,
                        score=1.0 - results['distances'][0][i]  # 转换为相似度得分
                    ))
            return chunks
        except Exception as e:
            logger.error(f"ChromaDB 检索失败: {e}")
            return []

    def delete_by_source(self, collection_name: str, source_file: str) -> bool:
        """根据文件路径精准删除所属的所有 Chunk"""
        try:
            # 如果集合不存在，说明是第一次建库，直接跳过
            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception:
                return True
                
            collection.delete(where={"source_file": source_file})
            logger.info(f"🗑️ 已从 Chroma 集合 {collection_name} 中清理文件: {source_file}")
            return True
        except Exception as e:
            logger.error(f"ChromaDB 删除历史文件失败: {e}")
            return False        