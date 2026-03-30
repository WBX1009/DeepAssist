import os
import json
from typing import List, Dict
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser
from jieba.analyse import ChineseAnalyzer

from backend.domain.interfaces.keyword_db import BaseKeywordDB
from backend.domain.entities.document import DocumentChunk
from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

class WhooshStore(BaseKeywordDB):
    def __init__(self):
        self.base_path = settings.KEYWORD_DB_PATH
        os.makedirs(self.base_path, exist_ok=True)
        self.analyzer = ChineseAnalyzer()
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            content=TEXT(stored=True, analyzer=self.analyzer),
            metadata=STORED()
        )
        
        # 🟢 性能优化：在内存中缓存已打开的 Index 对象，避免高频 I/O 摩擦
        self._index_cache: Dict[str, Any] = {}
        logger.info(f"✅ Whoosh 关键词索引客户端初始化完成: {self.base_path}")

    def _get_collection_dir(self, collection_name: str) -> str:
        col_dir = os.path.join(self.base_path, collection_name)
        os.makedirs(col_dir, exist_ok=True)
        return col_dir

    def _get_or_open_index(self, collection_name: str):
        """获取或缓存打开的索引对象"""
        if collection_name in self._index_cache:
            return self._index_cache[collection_name]
            
        col_dir = self._get_collection_dir(collection_name)
        if exists_in(col_dir):
            ix = open_dir(col_dir)
            self._index_cache[collection_name] = ix
            return ix
        return None

    def build_index(self, collection_name: str, chunks: List[DocumentChunk]) -> bool:
        col_dir = self._get_collection_dir(collection_name)
        try:
            if not exists_in(col_dir):
                ix = create_in(col_dir, self.schema)
            else:
                ix = open_dir(col_dir)
                
            writer = ix.writer()
            for chunk in chunks:
                writer.update_document(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=json.dumps(chunk.metadata, ensure_ascii=False)
                )
            writer.commit()
            
            # 刷新缓存
            self._index_cache[collection_name] = ix
            logger.info(f"成功向 Whoosh 索引 '{collection_name}' 写入 {len(chunks)} 条文本数据")
            return True
        except Exception as e:
            logger.error(f"Whoosh 写入失败: {e}")
            return False

    def search(self, collection_name: str, query_text: str, top_k: int) -> List[DocumentChunk]:
        ix = self._get_or_open_index(collection_name)
        if not ix:
            logger.warning(f"Whoosh 索引库 {collection_name} 不存在或尚未构建")
            return []
            
        try:
            chunks =[]
            # 使用上下文管理器打开 searcher，高效安全
            with ix.searcher() as searcher:
                query = QueryParser("content", ix.schema).parse(query_text)
                results = searcher.search(query, limit=top_k)
                
                for hit in results:
                    chunks.append(DocumentChunk(
                        id=hit["id"],
                        content=hit["content"],
                        metadata=json.loads(hit["metadata"]),
                        score=hit.score 
                    ))
            return chunks
        except Exception as e:
            logger.error(f"Whoosh 检索失败: {e}")
            return[]