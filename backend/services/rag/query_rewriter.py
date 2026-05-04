from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class QueryRewriteConfig:
    enable_rewrite: bool = True
    enable_multi_query: bool = True
    max_sub_queries: int = 3
    short_query_term_threshold: int = 4


@dataclass(frozen=True)
class QueryRewriteResult:
    rewritten_query: str
    semantic_queries: List[str]
    keyword_queries: List[str]
    sub_queries: List[str]
    strategy: str


class QueryRewriteService:
    """Deterministic query rewrite and multi-query expansion.

    This is the replaceable foundation for a future LLM-based rewriter. The
    current implementation avoids external model calls while exposing the same
    pipeline shape interviewers expect: rewrite, multi-query recall, fusion.
    """

    def __init__(self, config: QueryRewriteConfig | None = None):
        self.config = config or QueryRewriteConfig()

    def expand(
        self,
        normalized_query: str,
        keyword_query: str,
        key_terms: List[str],
        quoted_phrases: List[str],
    ) -> QueryRewriteResult:
        rewritten_query = self._rewrite(normalized_query, key_terms, quoted_phrases)

        semantic_queries = [normalized_query]
        keyword_queries = [keyword_query or normalized_query]
        sub_queries: List[str] = []

        if self.config.enable_rewrite and rewritten_query and rewritten_query != normalized_query:
            semantic_queries.append(rewritten_query)
            sub_queries.append(rewritten_query)

        if self.config.enable_multi_query:
            sub_queries.extend(self._build_sub_queries(normalized_query, key_terms, quoted_phrases))
            semantic_queries.extend(sub_queries)
            keyword_queries.extend(sub_queries)

        semantic_queries = self._dedupe(query for query in semantic_queries if query)
        keyword_queries = self._dedupe(query for query in keyword_queries if query)
        sub_queries = self._dedupe(query for query in sub_queries if query and query != normalized_query)

        strategy = "rewrite_multi_query" if len(semantic_queries) > 1 else "single_query_hybrid"
        return QueryRewriteResult(
            rewritten_query=rewritten_query,
            semantic_queries=semantic_queries,
            keyword_queries=keyword_queries,
            sub_queries=sub_queries[: self.config.max_sub_queries],
            strategy=strategy,
        )

    def _rewrite(
        self,
        normalized_query: str,
        key_terms: List[str],
        quoted_phrases: List[str],
    ) -> str:
        if not self.config.enable_rewrite or not normalized_query:
            return normalized_query

        if len(key_terms) > self.config.short_query_term_threshold:
            return normalized_query

        anchors = self._dedupe([*quoted_phrases, *key_terms])
        if not anchors:
            return normalized_query

        anchor_text = ", ".join(anchors)
        return (
            f"{normalized_query}. Technical context about {anchor_text}: "
            "definition, architecture, implementation, configuration, and common issues."
        )

    def _build_sub_queries(
        self,
        normalized_query: str,
        key_terms: List[str],
        quoted_phrases: List[str],
    ) -> List[str]:
        sub_queries: List[str] = []
        for phrase in quoted_phrases:
            sub_queries.append(phrase)

        if key_terms:
            sub_queries.append(" ".join(key_terms))
        if len(key_terms) >= 2:
            sub_queries.append(" ".join(key_terms[:2]))
        if len(key_terms) >= 4:
            sub_queries.append(" ".join(key_terms[2:4]))

        return [
            query
            for query in self._dedupe(sub_queries)
            if query and query != normalized_query
        ][: self.config.max_sub_queries]

    def _dedupe(self, values: Iterable[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for value in values:
            normalized = " ".join(value.split()).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped
