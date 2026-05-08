# from dataclasses import dataclass
# import re
# from typing import Iterable, List


# @dataclass(frozen=True)
# class QueryRewriteConfig:
#     enable_rewrite: bool = True
#     enable_multi_query: bool = True
#     max_sub_queries: int = 3
#     short_query_term_threshold: int = 4


# @dataclass(frozen=True)
# class QueryRewriteResult:
#     rewritten_query: str
#     semantic_queries: List[str]
#     keyword_queries: List[str]
#     sub_queries: List[str]
#     strategy: str
#     rewrite_notes: List[str]
#     stripped_fillers: List[str]
#     domain_hints: List[str]


# class QueryRewriteService:
#     """Deterministic query rewrite and multi-query expansion.

#     This is the replaceable foundation for a future LLM-based rewriter. The
#     current implementation avoids external model calls while exposing the same
#     pipeline shape interviewers expect: rewrite, multi-query recall, fusion.
#     """

#     _filler_patterns = (
#         r"^(请问|请帮我|帮我|麻烦你|你能不能|你能否|可以帮我|想请教一下)\s*",
#         r"^(could you|can you|please|help me)\s+",
#         r"\s*(吗|么|呢)\s*$",
#     )
#     _instruction_tail_patterns = (
#         r"[，,。；; ]*根据(?:外挂)?(?:医疗|法律|金融)?(?:知识库|数据库).*$",
#         r"[，,。；; ]*(?:没有|没|未)查(?:找|到).*$",
#         r"[，,。；; ]*找不到.*$",
#         r"[，,。；; ]*如果没.*$",
#         r"[，,。；; ]*查不到.*$",
#     )
#     _domain_hint_map = {
#         "api": ["api", "接口", "鉴权", "认证", "配置"],
#         "deployment": ["部署", "安装", "docker", "运行", "服务"],
#         "troubleshooting": ["报错", "错误", "异常", "失败", "排查"],
#         "rag": ["rag", "检索", "召回", "向量", "索引", "embedding"],
#         "agent": ["agent", "工具", "调用", "规划", "多步"],
#         "database": ["sql", "数据库", "sqlite", "schema", "表"],
#     }
#     _synonym_map = {
#         "头疼": ["头疼", "头痛"],
#         "头痛": ["头疼", "头痛"],
#         "邮件": ["邮件", "email", "e-mail", "mail"],
#         "前列腺炎": ["前列腺炎", "前列腺", "尿频", "尿急"],
#         "辞退员工": ["辞退员工", "解除劳动合同", "经济补偿"],
#         "上市": ["上市", "ipo", "财务条件", "审核"],
#     }
#     _term_pattern = re.compile(r"[A-Za-z0-9_./#:+-]+|[\u4e00-\u9fff]{2,}")
#     _generic_terms = {"资料", "数据库", "知识库", "回答", "查找", "找到", "没有", "根据", "外挂"}

#     def __init__(self, config: QueryRewriteConfig | None = None):
#         self.config = config or QueryRewriteConfig()

#     def expand(
#         self,
#         normalized_query: str,
#         keyword_query: str,
#         key_terms: List[str],
#         quoted_phrases: List[str],
#     ) -> QueryRewriteResult:
#         stripped_query, stripped_fillers = self._strip_fillers(normalized_query)
#         focused_query, stripped_instruction_tails = self._strip_instruction_tails(stripped_query)
#         working_query = focused_query or stripped_query
#         focused_key_terms = self._focused_key_terms(working_query, key_terms)
#         domain_hints = self._infer_domain_hints(working_query, focused_key_terms, quoted_phrases)
#         synonym_hints = self._expand_synonyms(working_query, key_terms, quoted_phrases)
#         combined_hints = self._dedupe([*domain_hints, *synonym_hints])
#         rewritten_query, rewrite_notes = self._rewrite(
#             working_query,
#             focused_key_terms,
#             quoted_phrases,
#             combined_hints,
#         )
#         if stripped_instruction_tails:
#             rewrite_notes.append("instruction_tail_removed")
#         if synonym_hints:
#             rewrite_notes.append("synonym_expansion")

#         semantic_queries = [working_query or normalized_query]
#         keyword_queries = [
#             self._build_keyword_query(
#                 keyword_query=keyword_query,
#                 working_query=working_query or normalized_query,
#                 synonym_hints=synonym_hints,
#             )
#         ]
#         sub_queries: List[str] = []

#         if self.config.enable_rewrite and rewritten_query and rewritten_query != working_query:
#             semantic_queries.append(rewritten_query)
#             sub_queries.append(rewritten_query)

#         if self.config.enable_multi_query:
#             sub_queries.extend(
#                 self._build_sub_queries(
#                     working_query or normalized_query,
#                     focused_key_terms,
#                     quoted_phrases,
#                     combined_hints,
#                 )
#             )
#             semantic_queries.extend(sub_queries)
#             keyword_queries.extend(sub_queries)

#         semantic_queries = self._dedupe(query for query in semantic_queries if query)
#         keyword_queries = self._dedupe(query for query in keyword_queries if query)
#         sub_queries = self._dedupe(query for query in sub_queries if query and query != normalized_query)

#         strategy = "rewrite_multi_query" if len(semantic_queries) > 1 else "single_query_hybrid"
#         return QueryRewriteResult(
#             rewritten_query=rewritten_query,
#             semantic_queries=semantic_queries,
#             keyword_queries=keyword_queries,
#             sub_queries=sub_queries[: self.config.max_sub_queries],
#             strategy=strategy,
#             rewrite_notes=rewrite_notes,
#             stripped_fillers=[*stripped_fillers, *stripped_instruction_tails],
#             domain_hints=combined_hints,
#         )

#     def _build_keyword_query(
#         self,
#         keyword_query: str,
#         working_query: str,
#         synonym_hints: List[str],
#     ) -> str:
#         return " ".join(
#             self._dedupe([working_query or keyword_query, *synonym_hints])
#         )

#     def _rewrite(
#         self,
#         normalized_query: str,
#         key_terms: List[str],
#         quoted_phrases: List[str],
#         domain_hints: List[str],
#     ) -> tuple[str, List[str]]:
#         if not self.config.enable_rewrite or not normalized_query:
#             return normalized_query, []

#         rewrite_notes: List[str] = []

#         if len(key_terms) > self.config.short_query_term_threshold:
#             if domain_hints:
#                 rewrite_notes.append("domain_hint_expansion")
#                 suffix = " ".join(domain_hints[:4])
#                 return f"{normalized_query} {suffix}", rewrite_notes
#             return normalized_query, rewrite_notes

#         anchors = self._dedupe([*quoted_phrases, *key_terms])
#         if domain_hints:
#             anchors.extend(hint for hint in domain_hints if hint not in anchors)
#             rewrite_notes.append("domain_hint_expansion")
#         if not anchors:
#             return normalized_query, rewrite_notes

#         anchor_text = " ".join(anchors[:6])
#         rewrite_notes.append("short_query_context_expansion")
#         return f"{normalized_query} {anchor_text} 定义 架构 配置 实现 常见问题", rewrite_notes

#     def _build_sub_queries(
#         self,
#         normalized_query: str,
#         key_terms: List[str],
#         quoted_phrases: List[str],
#         domain_hints: List[str],
#     ) -> List[str]:
#         sub_queries: List[str] = []
#         for phrase in quoted_phrases:
#             sub_queries.append(phrase)

#         if key_terms:
#             sub_queries.append(" ".join(key_terms))
#         if len(key_terms) >= 2:
#             sub_queries.append(" ".join(key_terms[:2]))
#         if len(key_terms) >= 4:
#             sub_queries.append(" ".join(key_terms[2:4]))
#         if domain_hints:
#             sub_queries.append(" ".join(domain_hints[:3]))

#         return [
#             query
#             for query in self._dedupe(sub_queries)
#             if query and query != normalized_query
#         ][: self.config.max_sub_queries]

#     def _strip_fillers(self, normalized_query: str) -> tuple[str, List[str]]:
#         stripped = normalized_query
#         matched_patterns: List[str] = []
#         for pattern in self._filler_patterns:
#             updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
#             if updated != stripped:
#                 matched_patterns.append(pattern)
#                 stripped = updated
#         stripped = " ".join(stripped.split()).strip()
#         return stripped or normalized_query, matched_patterns

#     def _strip_instruction_tails(self, normalized_query: str) -> tuple[str, List[str]]:
#         stripped = normalized_query
#         matched_patterns: List[str] = []
#         for pattern in self._instruction_tail_patterns:
#             updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
#             if updated != stripped:
#                 matched_patterns.append(pattern)
#                 stripped = updated
#         stripped = stripped.strip(" ，,。；;")
#         stripped = " ".join(stripped.split()).strip()
#         return stripped or normalized_query, matched_patterns

#     def _infer_domain_hints(
#         self,
#         normalized_query: str,
#         key_terms: List[str],
#         quoted_phrases: List[str],
#     ) -> List[str]:
#         haystack = " ".join([normalized_query, *key_terms, *quoted_phrases]).lower()
#         hints: List[str] = []
#         for values in self._domain_hint_map.values():
#             if any(value.lower() in haystack for value in values):
#                 hints.extend(values)
#         return self._dedupe(hints)

#     def _expand_synonyms(
#         self,
#         normalized_query: str,
#         key_terms: List[str],
#         quoted_phrases: List[str],
#     ) -> List[str]:
#         haystack = " ".join([normalized_query, *key_terms, *quoted_phrases]).lower()
#         hints: List[str] = []
#         for anchor, values in self._synonym_map.items():
#             if anchor.lower() in haystack:
#                 hints.extend(values)
#         return self._dedupe(hints)

#     def _focused_key_terms(
#         self,
#         working_query: str,
#         original_key_terms: List[str],
#     ) -> List[str]:
#         extracted: List[str] = []
#         for term in self._term_pattern.findall(working_query or ""):
#             if not term:
#                 continue
#             extracted.extend(self._segment_term(term))
#         extracted = [term for term in extracted if term and term.lower() not in self._generic_terms]
#         if extracted:
#             return self._dedupe(extracted)
#         return self._dedupe(
#             term for term in original_key_terms if term.lower() not in self._generic_terms
#         )

#     def _segment_term(self, term: str) -> List[str]:
#         normalized = term.strip()
#         if not normalized:
#             return []
#         if not re.search(r"[\u4e00-\u9fff]", normalized):
#             return [normalized]
#         try:
#             import jieba

#             return [item.strip() for item in jieba.lcut(normalized) if item.strip()]
#         except Exception:
#             return [normalized]

#     def _dedupe(self, values: Iterable[str]) -> List[str]:
#         seen = set()
#         deduped: List[str] = []
#         for value in values:
#             normalized = " ".join(value.split()).strip()
#             key = normalized.lower()
#             if not normalized or key in seen:
#                 continue
#             seen.add(key)
#             deduped.append(normalized)
#         return deduped
# __________________________________________________________________________

# from dataclasses import dataclass, replace
# from typing import Iterable, List, Optional, Set
# import re

# from backend.domain.interfaces.llm import BaseLLM

# @dataclass(frozen=True)
# class QueryRewriteConfig:
#     enable_rewrite: bool = True
#     enable_multi_query: bool = True
#     max_sub_queries: int = 3
#     short_query_term_threshold: int = 4
#     # 新增开关：是否启用 LLM 重写
#     enable_llm_rewrite: bool = True

# @dataclass(frozen=True)
# class QueryRewriteResult:
#     rewritten_query: str
#     semantic_queries: List[str]
#     keyword_queries: List[str]
#     sub_queries: List[str]
#     strategy: str
#     rewrite_notes: List[str]
#     stripped_fillers: List[str]
#     domain_hints: List[str]

# class QueryRewriteService:
#     """查询重写器：优先使用 LLM 提取关键词，失败时回退到正则+同义词处理。"""

#     # 以下常量仍作为兜底逻辑存在，也可以迁移到配置文件中
#     _filler_patterns = (
#         r"^(请问|请帮我|帮我|麻烦你|你能不能|你能否|可以帮我|想请教一下)\s*",
#         r"^(could you|can you|please|help me)\s+",
#         r"\s*(吗|么|呢)\s*$",
#     )
#     _instruction_tail_patterns = (
#         r"[，,。；; ]*根据(?:外挂)?(?:医疗|法律|金融)?(?:知识库|数据库).*$",
#         r"[，,。；; ]*(?:没有|没|未)查(?:找|到).*$",
#         r"[，,。；; ]*找不到.*$",
#         r"[，,。；; ]*如果没.*$",
#         r"[，,。；; ]*查不到.*$",
#     )
#     _synonym_map = {
#         "头疼": ["头疼", "头痛"],
#         "头痛": ["头疼", "头痛"],
#         "邮件": ["邮件", "email", "e-mail", "mail"],
#         "前列腺炎": ["前列腺炎", "前列腺", "尿频", "尿急"],
#         "辞退员工": ["辞退员工", "解除劳动合同", "经济补偿"],
#         "上市": ["上市", "ipo", "财务条件", "审核"],
#     }
#     _term_pattern = re.compile(r"[A-Za-z0-9_./#:+-]+|[\u4e00-\u9fff]{2,}")
#     _generic_terms = {"资料", "数据库", "知识库", "回答", "查找", "找到", "没有", "根据", "外挂"}

#     def __init__(
#         self,
#         config: Optional[QueryRewriteConfig] = None,
#         llm: Optional[BaseLLM] = None,
#     ):
#         self.config = config or QueryRewriteConfig()
#         self.llm = llm

#     # ===== LLM 重写逻辑 =====
#     def _rewrite_via_llm(self, query: str) -> str:
#         """调用 LLM 提取核心关键词，不成功返回空字符串。"""
#         if not self.llm or not self.config.enable_llm_rewrite:
#             return ""

#         system_message = (
#             "你是一个专业的搜索查询重写助手。"
#             "请去掉用户问题中的客套话、指令或情绪表达，只保留最关键的 2-4 个搜索关键词。"
#             "直接输出关键词，用空格分隔，不要加任何说明。"
#         )
#         messages = [
#             {"role": "system", "content": system_message},
#             {"role": "user", "content": query},
#         ]
#         try:
#             response_chunks: List[str] = []
#             for chunk in self.llm.chat_stream(messages=messages, model_name=None, temperature=0.1, top_p=1.0):
#                 if chunk:
#                     response_chunks.append(chunk)
#             full_response = "".join(response_chunks).strip()
#             return full_response.strip("'\"")
#         except Exception as exc:
#             # LLM 调用失败时记录并回退
#             return ""

#     # ===== 传统正则重写 =====
#     def _strip_fillers(self, normalized_query: str) -> tuple[str, List[str]]:
#         stripped = normalized_query
#         matched_patterns: List[str] = []
#         for pattern in self._filler_patterns:
#             updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
#             if updated != stripped:
#                 matched_patterns.append(pattern)
#                 stripped = updated
#         stripped = " ".join(stripped.split()).strip()
#         return stripped or normalized_query, matched_patterns

#     def _strip_instruction_tails(self, normalized_query: str) -> tuple[str, List[str]]:
#         stripped = normalized_query
#         matched_patterns: List[str] = []
#         for pattern in self._instruction_tail_patterns:
#             updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
#             if updated != stripped:
#                 matched_patterns.append(pattern)
#                 stripped = updated
#         stripped = stripped.strip(" ，,。；;")
#         stripped = " ".join(stripped.split()).strip()
#         return stripped or normalized_query, matched_patterns

#     def _expand_synonyms(
#         self,
#         normalized_query: str,
#     ) -> List[str]:
#         """简化版同义词扩展：仅根据 synonym_map 进行字面替换，可根据需要迁移到配置。"""
#         haystack = normalized_query.lower()
#         hints: List[str] = []
#         for anchor, values in self._synonym_map.items():
#             if anchor.lower() in haystack:
#                 hints.extend(values)
#         return self._dedupe(hints)

#     # ===== 主入口 =====
#     def expand(
#         self,
#         normalized_query: str,
#         keyword_query: str,
#         key_terms: List[str],
#         quoted_phrases: List[str],
#     ) -> QueryRewriteResult:
#         """
#         优先使用 LLM 重写查询；如果失败，则使用正则和同义词兜底。
#         最终返回 QueryRewriteResult，兼容后续管道。
#         """
#         # 1. 尝试 LLM 重写
#         llm_rewritten = self._rewrite_via_llm(normalized_query)

#         # 2. 回退到正则清洗、同义词扩展，生成基础结果
#         stripped_query, stripped_fillers = self._strip_fillers(normalized_query)
#         focused_query, stripped_instruction_tails = self._strip_instruction_tails(stripped_query)
#         working_query = focused_query or stripped_query
#         rewrite_notes: List[str] = []
#         if stripped_instruction_tails:
#             rewrite_notes.append("instruction_tail_removed")

#         # 只做简易的同义词扩展，没有原先复杂的 domain_hints 等
#         synonym_hints = self._expand_synonyms(working_query)
#         if synonym_hints:
#             rewrite_notes.append("synonym_expansion")

#         # 默认情况下 rewriten_query = working_query
#         rewritten_query = working_query
#         if self.config.enable_llm_rewrite and llm_rewritten:
#             # LLM 重写成功，替换 rewriten_query，并记录说明
#             rewritten_query = llm_rewritten
#             rewrite_notes.append("llm_rewrite_applied")

#         # 构建 semantic_queries / keyword_queries
#         semantic_queries = [rewritten_query]
#         keyword_queries = [" ".join(self._dedupe([keyword_query, *synonym_hints]))]
#         sub_queries: List[str] = []

#         # 可选多查询扩展（此处根据需要自行实现或关闭）
#         if self.config.enable_multi_query and synonym_hints:
#             sub_queries = synonym_hints[: self.config.max_sub_queries]
#             semantic_queries.extend(sub_queries)
#             keyword_queries.extend(sub_queries)

#         semantic_queries = self._dedupe(semantic_queries)
#         keyword_queries = self._dedupe(keyword_queries)

#         strategy = "llm_rewrite" if llm_rewritten else (
#             "rewrite_multi_query" if len(semantic_queries) > 1 else "single_query"
#         )

#         return QueryRewriteResult(
#             rewritten_query=rewritten_query,
#             semantic_queries=semantic_queries,
#             keyword_queries=keyword_queries,
#             sub_queries=sub_queries,
#             strategy=strategy,
#             rewrite_notes=rewrite_notes,
#             stripped_fillers=[*stripped_fillers, *stripped_instruction_tails],
#             domain_hints=[],
#         )

#     # ===== 工具函数 =====
#     def _dedupe(self, values: Iterable[str]) -> List[str]:
#         seen = set()
#         deduped: List[str] = []
#         for value in values:
#             normalized = " ".join(value.split()).strip()
#             key = normalized.lower()
#             if not normalized or key in seen:
#                 continue
#             seen.add(key)
#             deduped.append(normalized)
#         return deduped



# backend/services/rag/query_rewriter.py
from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Set
import re
from backend.domain.interfaces.llm import BaseLLM

@dataclass(frozen=True)
class QueryRewriteConfig:
    enable_rewrite: bool = True
    enable_multi_query: bool = True
    max_sub_queries: int = 3
    # 开关：是否启用 LLM 重写
    enable_llm_rewrite: bool = True

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
    """优先通过 LLM 提炼关键词的查询重写器，失败时回退到正则逻辑。"""
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
    _synonym_map = {
        # 简化版同义词，可以迁移到配置文件
        "前列腺炎": ["前列腺炎", "前列腺", "尿频", "尿急"],
        "辞退员工": ["辞退员工", "解除劳动合同", "经济补偿"],
        "上市": ["上市", "ipo", "财务条件", "审核"],
    }
    _term_pattern = re.compile(r"[A-Za-z0-9_./#:+-]+|[\u4e00-\u9fff]{2,}")
    _generic_terms = {"资料", "数据库", "知识库", "回答", "查找", "找到", "没有", "根据"}

    def __init__(
        self,
        config: Optional[QueryRewriteConfig] = None,
        llm: Optional[BaseLLM] = None,
    ):
        self.config = config or QueryRewriteConfig()
        self.llm = llm

    def _rewrite_via_llm(self, query: str) -> str:
        """调用 LLM 提炼关键词；改用流式拼接防止网络挂起。"""
        if not self.llm or not self.config.enable_llm_rewrite:
            return ""
            
        system_message = (
            "你是一个专业的搜索查询重写助手。"
            "请去掉问题中的客套话、指令或情绪表达，只保留最关键的 2~4 个搜索关键词。"
            "直接输出关键词，用空格分隔，不要添加说明。"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
        
        try:
            # 🚀 核心：使用 chat_stream 替代 chat
            response_chunks = []
            for chunk in self.llm.chat_stream(messages=messages, temperature=0.1):
                if chunk:
                    response_chunks.append(chunk)
            
            full_response = "".join(response_chunks).strip()
            # 清洗可能存在的引号
            return full_response.strip("'\"")
        except Exception as exc:
            # 如果流式也失败，记录日志并返回空，触发正则兜底
            import logging
            logging.getLogger(__name__).error(f"LLM重写流式调用失败: {exc}")
            return ""

    # 以下是简化版正则清洗和同义词扩展（兜底）
    def _strip_fillers(self, query: str) -> tuple[str, List[str]]:
        stripped = query
        matched = []
        for pattern in self._filler_patterns:
            updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
            if updated != stripped:
                matched.append(pattern)
                stripped = updated
        return stripped.strip() or query, matched

    def _strip_instruction_tails(self, query: str) -> tuple[str, List[str]]:
        stripped = query
        matched = []
        for pattern in self._instruction_tail_patterns:
            updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
            if updated != stripped:
                matched.append(pattern)
                stripped = updated
        return stripped.strip(" ，,。；;").strip() or query, matched

    def _expand_synonyms(self, query: str) -> List[str]:
        lower = query.lower()
        hints = []
        for anchor, values in self._synonym_map.items():
            if anchor.lower() in lower:
                hints.extend(values)
        return self._dedupe(hints)

    def expand(
        self,
        normalized_query: str,
        keyword_query: str,
        key_terms: List[str],
        quoted_phrases: List[str],
    ) -> QueryRewriteResult:
        # 1. LLM 提炼关键词
        llm_query = self._rewrite_via_llm(normalized_query)
        # 2. 回退到正则清洗
        stripped, fillers = self._strip_fillers(normalized_query)
        focused, tails = self._strip_instruction_tails(stripped)
        working = focused or stripped
        rewrite_notes: List[str] = []
        if tails:
            rewrite_notes.append("instruction_tail_removed")
        # 3. 同义词扩展
        synonyms = self._expand_synonyms(working)
        if synonyms:
            rewrite_notes.append("synonym_expansion")
        # 4. 确定最终的 rewritten_query
        rewritten_query = llm_query if llm_query else working
        if llm_query:
            rewrite_notes.append("llm_rewrite_applied")
        # 5. 构建查询列表
        semantic_queries = [rewritten_query]
        keyword_queries = [" ".join(self._dedupe([keyword_query, *synonyms]))]
        sub_queries: List[str] = []
        if self.config.enable_multi_query and synonyms:
            sub_queries = synonyms[: self.config.max_sub_queries]
            semantic_queries.extend(sub_queries)
            keyword_queries.extend(sub_queries)
        semantic_queries = self._dedupe(semantic_queries)
        keyword_queries = self._dedupe(keyword_queries)
        strategy = "llm_rewrite" if llm_query else ("rewrite_multi_query" if len(semantic_queries) > 1 else "single_query")
        return QueryRewriteResult(
            rewritten_query=rewritten_query,
            semantic_queries=semantic_queries,
            keyword_queries=keyword_queries,
            sub_queries=sub_queries,
            strategy=strategy,
            rewrite_notes=rewrite_notes,
            stripped_fillers=[*fillers, *tails],
            domain_hints=[],
        )

    def _dedupe(self, values: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        deduped: List[str] = []
        for v in values:
            n = " ".join(v.split()).strip()
            key = n.lower()
            if n and key not in seen:
                seen.add(key)
                deduped.append(n)
        return deduped