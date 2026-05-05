from dataclasses import dataclass
import re
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
    rewrite_notes: List[str]
    stripped_fillers: List[str]
    domain_hints: List[str]


class QueryRewriteService:
    """Deterministic query rewrite and multi-query expansion.

    This is the replaceable foundation for a future LLM-based rewriter. The
    current implementation avoids external model calls while exposing the same
    pipeline shape interviewers expect: rewrite, multi-query recall, fusion.
    """

    _filler_patterns = (
        r"^(请问|请帮我|帮我|麻烦你|你能不能|你能否|可以帮我|想请教一下)\s*",
        r"^(could you|can you|please|help me)\s+",
        r"\s*(吗|么|呢)\s*$",
    )
    _domain_hint_map = {
        "api": ["api", "接口", "鉴权", "认证", "配置"],
        "deployment": ["部署", "安装", "docker", "运行", "服务"],
        "troubleshooting": ["报错", "错误", "异常", "失败", "排查"],
        "rag": ["rag", "检索", "召回", "向量", "索引", "embedding"],
        "agent": ["agent", "工具", "调用", "规划", "多步"],
        "database": ["sql", "数据库", "sqlite", "schema", "表"],
    }

    def __init__(self, config: QueryRewriteConfig | None = None):
        self.config = config or QueryRewriteConfig()

    def expand(
        self,
        normalized_query: str,
        keyword_query: str,
        key_terms: List[str],
        quoted_phrases: List[str],
    ) -> QueryRewriteResult:
        stripped_query, stripped_fillers = self._strip_fillers(normalized_query)
        domain_hints = self._infer_domain_hints(stripped_query, key_terms, quoted_phrases)
        rewritten_query, rewrite_notes = self._rewrite(
            stripped_query,
            key_terms,
            quoted_phrases,
            domain_hints,
        )

        semantic_queries = [stripped_query or normalized_query]
        keyword_queries = [keyword_query or stripped_query or normalized_query]
        sub_queries: List[str] = []

        if self.config.enable_rewrite and rewritten_query and rewritten_query != normalized_query:
            semantic_queries.append(rewritten_query)
            sub_queries.append(rewritten_query)

        if self.config.enable_multi_query:
            sub_queries.extend(
                self._build_sub_queries(
                    stripped_query or normalized_query,
                    key_terms,
                    quoted_phrases,
                    domain_hints,
                )
            )
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
            rewrite_notes=rewrite_notes,
            stripped_fillers=stripped_fillers,
            domain_hints=domain_hints,
        )

    def _rewrite(
        self,
        normalized_query: str,
        key_terms: List[str],
        quoted_phrases: List[str],
        domain_hints: List[str],
    ) -> tuple[str, List[str]]:
        if not self.config.enable_rewrite or not normalized_query:
            return normalized_query, []

        rewrite_notes: List[str] = []

        if len(key_terms) > self.config.short_query_term_threshold:
            if domain_hints:
                rewrite_notes.append("domain_hint_expansion")
                suffix = " ".join(domain_hints[:4])
                return f"{normalized_query} {suffix}", rewrite_notes
            return normalized_query, rewrite_notes

        anchors = self._dedupe([*quoted_phrases, *key_terms])
        if domain_hints:
            anchors.extend(hint for hint in domain_hints if hint not in anchors)
            rewrite_notes.append("domain_hint_expansion")
        if not anchors:
            return normalized_query, rewrite_notes

        anchor_text = " ".join(anchors[:6])
        rewrite_notes.append("short_query_context_expansion")
        return f"{normalized_query} {anchor_text} 定义 架构 配置 实现 常见问题", rewrite_notes

    def _build_sub_queries(
        self,
        normalized_query: str,
        key_terms: List[str],
        quoted_phrases: List[str],
        domain_hints: List[str],
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
        if domain_hints:
            sub_queries.append(" ".join(domain_hints[:3]))

        return [
            query
            for query in self._dedupe(sub_queries)
            if query and query != normalized_query
        ][: self.config.max_sub_queries]

    def _strip_fillers(self, normalized_query: str) -> tuple[str, List[str]]:
        stripped = normalized_query
        matched_patterns: List[str] = []
        for pattern in self._filler_patterns:
            updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
            if updated != stripped:
                matched_patterns.append(pattern)
                stripped = updated
        stripped = " ".join(stripped.split()).strip()
        return stripped or normalized_query, matched_patterns

    def _infer_domain_hints(
        self,
        normalized_query: str,
        key_terms: List[str],
        quoted_phrases: List[str],
    ) -> List[str]:
        haystack = " ".join([normalized_query, *key_terms, *quoted_phrases]).lower()
        hints: List[str] = []
        for values in self._domain_hint_map.values():
            if any(value.lower() in haystack for value in values):
                hints.extend(values)
        return self._dedupe(hints)

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
