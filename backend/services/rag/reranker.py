import re
from dataclasses import dataclass
from typing import List, Set

from backend.domain.entities.document import DocumentChunk
from backend.domain.interfaces.reranker import BaseReranker


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "lexical_overlap_fallback"
    fused_score_weight: float = 0.3
    lexical_score_weight: float = 0.7


class NoOpReranker(BaseReranker):
    """Reranker placeholder that preserves first-stage order."""

    model_name = "noop"

    def rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int,
    ) -> List[DocumentChunk]:
        return list(documents[:top_k])


class LexicalOverlapReranker(BaseReranker):
    """Deterministic fallback reranker until a cross-encoder adapter is installed."""

    _term_pattern = re.compile(r"[A-Za-z0-9_./#:+-]+|[\u4e00-\u9fff]{2,}")

    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()
        self.model_name = self.config.model_name

    def rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int,
    ) -> List[DocumentChunk]:
        if not documents:
            return []

        query_terms = self._terms(query)
        max_fused_score = max((doc.score or 0.0) for doc in documents) or 1.0

        scored_docs = []
        for doc in documents:
            fused_score = (doc.score or 0.0) / max_fused_score
            lexical_score = self._lexical_overlap(query_terms, doc)
            rerank_score = (
                self.config.fused_score_weight * fused_score
                + self.config.lexical_score_weight * lexical_score
            )
            reranked_doc = doc.model_copy(deep=True)
            reranked_doc.score = rerank_score
            reranked_doc.metadata = {
                **reranked_doc.metadata,
                "rerank": {
                    "model": self.model_name,
                    "fused_score": doc.score,
                    "lexical_score": lexical_score,
                    "rerank_score": rerank_score,
                },
            }
            scored_docs.append(reranked_doc)

        scored_docs.sort(key=lambda doc: doc.score or 0.0, reverse=True)

        final_docs: List[DocumentChunk] = []
        for rank, doc in enumerate(scored_docs[:top_k], start=1):
            doc.metadata = {
                **doc.metadata,
                "rerank": {
                    **doc.metadata.get("rerank", {}),
                    "rerank_rank": rank,
                },
            }
            final_docs.append(doc)
        return final_docs

    def _terms(self, text: str) -> Set[str]:
        return {
            term.lower()
            for term in self._term_pattern.findall(text or "")
            if len(term.strip()) > 1
        }

    def _lexical_overlap(self, query_terms: Set[str], doc: DocumentChunk) -> float:
        if not query_terms:
            return 0.0

        doc_terms = self._terms(doc.content)
        metadata = doc.metadata or {}
        for key in ("source_file", "title", "section", "title_path", "heading_path"):
            value = metadata.get(key)
            if isinstance(value, list):
                doc_terms.update(str(item).lower() for item in value)
            elif value:
                doc_terms.update(self._terms(str(value)))

        if not doc_terms:
            return 0.0
        return len(query_terms & doc_terms) / len(query_terms)
