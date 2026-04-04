import os
import json
import re
from typing import List, Dict, Any
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser, OrGroup
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
        
        # 🌟 核心：必须保留 source_file，这是支持 CRUD 精准删除的基石
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            source_file=ID(stored=True),
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
                
            # 分配缓冲，加速大批量写入
            writer = ix.writer(limitmb=1024, multisegment=True)
            
            try:
                for chunk in chunks:
                    writer.update_document(
                        id=chunk.id,
                        source_file=chunk.metadata.get("source_file", "unknown"),
                        content=chunk.content,
                        metadata=json.dumps(chunk.metadata, ensure_ascii=False)
                    )
                writer.commit()
            except Exception as inner_e:
                # 🚨 极其关键：一旦按了 Ctrl+C 导致中断，强制取消写入，删除 .lock 锁文件！
                writer.cancel()
                raise inner_e
            
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
            # 过滤掉特殊字符，防止 Whoosh 解析器报错
            clean_query = re.sub(r'[^\w\u4e00-\u9fa5]+', ' ', query_text)
            
            with ix.searcher() as searcher:
                # 使用 OrGroup 提高召回率（只要匹配部分关键词就召回）
                parser = QueryParser("content", ix.schema, group=OrGroup.factory(0.9))
                query = parser.parse(clean_query)
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

    def delete_by_source(self, collection_name: str, source_file: str) -> bool:
        """根据文件路径精准删除所属的所有 Chunk"""
        col_dir = self._get_collection_dir(collection_name)
        if not exists_in(col_dir): 
            return True
            
        try:
            ix = self._get_or_open_index(collection_name)
            if not ix: 
                return True
                
            writer = ix.writer()
            # 🛠️ 修复了你第一版代码中不小心删掉的这一行
            writer.delete_by_term("source_file", source_file)
            writer.commit()
            
            self._index_cache[collection_name] = ix
            logger.info(f"🗑️ 已从 Whoosh 集合 {collection_name} 中清理文件: {source_file}")
            return True
        except Exception as e:
            logger.error(f"Whoosh 删除历史文件失败: {e}")
            return False