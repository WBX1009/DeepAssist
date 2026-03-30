import uuid
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter

from backend.domain.entities.document import DocumentChunk
from backend.core.logger import get_logger

logger = get_logger(__name__)

class DocumentChunker:
    """
    文档结构化切分服务。
    保留文档的层级上下文（如 # 标题，## 子标题），这对基于 Tech/PDF 文档的 RAG 极其重要。
    """
    def __init__(self):
        # 定义需要识别的 Markdown 层级
        headers_to_split_on =[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        # strip_headers=False：保留标题文本在 chunk 内部，有助于语义完整性
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False 
        )

    def split_markdown(self, text: str, source_name: str = "unknown") -> List[DocumentChunk]:
        """
        将原始 Markdown 文本切分为统一的 DocumentChunk 列表
        """
        try:
            # 使用 LangChain 的切分器进行初步物理切分
            lc_documents = self.splitter.split_text(text)
            
            chunks =[]
            for doc in lc_documents:
                # 过滤掉无效的空行/极短文本
                if len(doc.page_content.strip()) < 10:
                    continue
                    
                # 提取 LangChain 提取出的层级元数据
                metadata = doc.metadata.copy()
                metadata["source"] = source_name
                
                # 转换为我们在 Domain 定义的统一实体类
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),  # 生成唯一 ID
                    content=doc.page_content,
                    metadata=metadata
                )
                chunks.append(chunk)
                
            logger.info(f"成功将文档 [{source_name}] 切分为 {len(chunks)} 个结构化 Chunk")
            return chunks
            
        except Exception as e:
            logger.error(f"文档切分失败 [{source_name}]: {e}")
            return[]