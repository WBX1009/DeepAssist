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
    _instruction_tail_patterns = (
        r"[，,。；; ]*根据(?:外挂)?(?:医疗|法律|金融)?(?:知识库|数据库).*$",
        r"[，,。；; ]*(?:没有|没|未)查(?:找|到).*$",
        r"[，,。；; ]*找不到.*$",
        r"[，,。；; ]*如果没.*$",
        r"[，,。；; ]*查不到.*$",
    )
    _domain_hint_map = {
        "api": ["api", "接口", "鉴权", "认证", "配置"],
        "deployment": ["部署", "安装", "docker", "运行", "服务"],
        "troubleshooting": ["报错", "错误", "异常", "失败", "排查"],
        "rag": ["rag", "检索", "召回", "向量", "索引", "embedding"],
        "agent": ["agent", "工具", "调用", "规划", "多步"],
        "database": ["sql", "数据库", "sqlite", "schema", "表"],
    }
    _synonym_map = {
        "头疼": ["头疼", "头痛"],
        "头痛": ["头疼", "头痛"],
        "邮件": ["邮件", "email", "e-mail", "mail"],
        "前列腺炎": ["前列腺炎", "前列腺", "尿频", "尿急"],
        "辞退员工": ["辞退员工", "解除劳动合同", "经济补偿"],
        "上市": ["上市", "ipo", "财务条件", "审核"],
    }
    _term_pattern = re.compile(r"[A-Za-z0-9_./#:+-]+|[\u4e00-\u9fff]{2,}")
    _generic_terms = {"资料", "数据库", "知识库", "回答", "查找", "找到", "没有", "根据", "外挂"}

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
        focused_query, stripped_instruction_tails = self._strip_instruction_tails(stripped_query)
        working_query = focused_query or stripped_query
        focused_key_terms = self._focused_key_terms(working_query, key_terms)
        domain_hints = self._infer_domain_hints(working_query, focused_key_terms, quoted_phrases)
        synonym_hints = self._expand_synonyms(working_query, key_terms, quoted_phrases)
        combined_hints = self._dedupe([*domain_hints, *synonym_hints])
        rewritten_query, rewrite_notes = self._rewrite(
            working_query,
            focused_key_terms,
            quoted_phrases,
            combined_hints,
        )
        if stripped_instruction_tails:
            rewrite_notes.append("instruction_tail_removed")
        if synonym_hints:
            rewrite_notes.append("synonym_expansion")

        semantic_queries = [working_query or normalized_query]
        keyword_queries = [
            self._build_keyword_query(
                keyword_query=keyword_query,
                working_query=working_query or normalized_query,
                synonym_hints=synonym_hints,
            )
        ]
        sub_queries: List[str] = []

        if self.config.enable_rewrite and rewritten_query and rewritten_query != working_query:
            semantic_queries.append(rewritten_query)
            sub_queries.append(rewritten_query)

        if self.config.enable_multi_query:
            sub_queries.extend(
                self._build_sub_queries(
                    working_query or normalized_query,
                    focused_key_terms,
                    quoted_phrases,
                    combined_hints,
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
            stripped_fillers=[*stripped_fillers, *stripped_instruction_tails],
            domain_hints=combined_hints,
        )

    def _build_keyword_query(
        self,
        keyword_query: str,
        working_query: str,
        synonym_hints: List[str],
    ) -> str:
        return " ".join(
            self._dedupe([working_query or keyword_query, *synonym_hints])
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

    def _strip_instruction_tails(self, normalized_query: str) -> tuple[str, List[str]]:
        stripped = normalized_query
        matched_patterns: List[str] = []
        for pattern in self._instruction_tail_patterns:
            updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
            if updated != stripped:
                matched_patterns.append(pattern)
                stripped = updated
        stripped = stripped.strip(" ，,。；;")
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

    def _expand_synonyms(
        self,
        normalized_query: str,
        key_terms: List[str],
        quoted_phrases: List[str],
    ) -> List[str]:
        haystack = " ".join([normalized_query, *key_terms, *quoted_phrases]).lower()
        hints: List[str] = []
        for anchor, values in self._synonym_map.items():
            if anchor.lower() in haystack:
                hints.extend(values)
        return self._dedupe(hints)

    def _focused_key_terms(
        self,
        working_query: str,
        original_key_terms: List[str],
    ) -> List[str]:
        extracted: List[str] = []
        for term in self._term_pattern.findall(working_query or ""):
            if not term:
                continue
            extracted.extend(self._segment_term(term))
        extracted = [term for term in extracted if term and term.lower() not in self._generic_terms]
        if extracted:
            return self._dedupe(extracted)
        return self._dedupe(
            term for term in original_key_terms if term.lower() not in self._generic_terms
        )

    def _segment_term(self, term: str) -> List[str]:
        normalized = term.strip()
        if not normalized:
            return []
        if not re.search(r"[\u4e00-\u9fff]", normalized):
            return [normalized]
        try:
            import jieba

            return [item.strip() for item in jieba.lcut(normalized) if item.strip()]
        except Exception:
            return [normalized]

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
