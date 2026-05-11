from dataclasses import dataclass
from typing import Iterable, List, Optional, Set
import re
from backend.domain.interfaces.llm import BaseLLM
from backend.common.logger import get_logger
from backend.common.config import settings

logger = get_logger(__name__)

@dataclass(frozen=True)
class QueryRewriteConfig:
    enable_rewrite: bool = True
    enable_multi_query: bool = True
    max_sub_queries: int = 3
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
    
    # 🚀 修复点：扩充常用同义词表
    _synonym_map = {
        "前列腺炎":["前列腺炎", "前列腺", "尿频", "尿急"],
        "头疼":["头疼", "头痛", "头晕"],
        "颈椎病":["颈椎病", "颈椎", "颈椎综合征"],
        "辞退员工":["辞退员工", "解除劳动合同", "经济补偿"],
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
        # 🚀 修复点：动态应用 config.py 的配置
        if not self.llm or not getattr(self.config, 'enable_llm_rewrite', True) or not settings.ENABLE_LLM_REWRITE:
            return ""

        system_message = (
            "你是一个专业的搜索查询重写助手。"
            "请去掉问题中的客套话、指令或情绪表达，只保留最关键的 2~4 个搜索关键词。"
            "直接输出关键词，用空格分隔，不要添加说明。"
        )
        messages =[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]

        try:
            response_chunks =[]
            for chunk in self.llm.chat_stream(messages=messages, temperature=0.1):
                if isinstance(chunk, dict) and chunk.get("type") == "content":
                    response_chunks.append(chunk["content"])
                elif isinstance(chunk, str):
                    response_chunks.append(chunk)

            full_response = "".join(response_chunks).strip()
            return full_response.strip("'\"")
        except Exception as exc:
            # 🚀 修复点：使用统一日志组件
            logger.error(f"LLM重写流式调用失败: {exc}")
            return ""

    def _strip_fillers(self, query: str) -> tuple[str, List[str]]:
        stripped = query
        matched =[]
        for pattern in self._filler_patterns:
            updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
            if updated != stripped:
                matched.append(pattern)
                stripped = updated
        return stripped.strip() or query, matched

    def _strip_instruction_tails(self, query: str) -> tuple[str, List[str]]:
        stripped = query
        matched =[]
        for pattern in self._instruction_tail_patterns:
            updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
            if updated != stripped:
                matched.append(pattern)
                stripped = updated
        return stripped.strip(" ，,。；;").strip() or query, matched

    def _expand_synonyms(self, query: str) -> List[str]:
        lower = query.lower()
        hints =[]
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
        llm_query = self._rewrite_via_llm(normalized_query)
        stripped, fillers = self._strip_fillers(normalized_query)
        focused, tails = self._strip_instruction_tails(stripped)
        working = focused or stripped
        rewrite_notes: List[str] =[]
        if tails:
            rewrite_notes.append("instruction_tail_removed")
            
        synonyms = self._expand_synonyms(working)
        if synonyms:
            rewrite_notes.append("synonym_expansion")
            
        rewritten_query = llm_query if llm_query else working
        if llm_query:
            rewrite_notes.append("llm_rewrite_applied")
            
        semantic_queries =[rewritten_query]
        keyword_queries = [" ".join(self._dedupe([keyword_query, *synonyms]))]
        sub_queries: List[str] =[]
        
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
        deduped: List[str] =[]
        for v in values:
            n = " ".join(v.split()).strip()
            key = n.lower()
            if n and key not in seen:
                seen.add(key)
                deduped.append(n)
        return deduped