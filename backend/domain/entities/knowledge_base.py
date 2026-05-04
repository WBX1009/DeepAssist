from typing import Any, Dict

from pydantic import BaseModel, Field


class KnowledgeBaseFile(BaseModel):
    """File-level summary for a knowledge-base collection."""

    source_file: str = Field(..., description="Original uploaded file name")
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks in this store")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
