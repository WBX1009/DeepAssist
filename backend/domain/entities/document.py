from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class DocumentChunk(BaseModel):
    """统一的文档块实体，无论是从 Chroma 还是 Whoosh 查出来，最终都必须转成这个样子"""
    id: str = Field(..., description="Chunk 的唯一标识")
    content: str = Field(..., description="文档正文内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据，如来源、页码、标题层级等")
    score: Optional[float] = Field(default=None, description="检索打分（可选）")