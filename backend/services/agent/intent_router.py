import re
from typing import Iterable, Optional, Set

from backend.domain.entities.intent import IntentDecision, IntentType


class IntentRouter:
    """Lightweight deterministic intent router for workflow selection."""

    RAG_PATTERNS = (
        "\u77e5\u8bc6\u5e93",  # knowledge base
        "\u6587\u6863",  # document
        "\u8d44\u6599",  # material
        "\u53c2\u8003",  # reference
        "\u68c0\u7d22",  # retrieval/search
        "\u6839\u636e.*(\u5185\u5bb9|\u8d44\u6599|\u6587\u6863)",  # according to content/material/doc
        r"rag",
        r"chunk",
        "\u5411\u91cf",  # vector
        "\u53ec\u56de",  # recall
        "\u7d22\u5f15",  # index
    )
    AGENT_PATTERNS = (
        "\u6267\u884c",  # execute
        "\u8fd0\u884c",  # run
        "\u8c03\u7528",  # call
        "\u5de5\u5177",  # tool
        "\u5199\u5165",  # write
        "\u521b\u5efa.*\u6587\u4ef6",  # create file
        "\u8bfb\u53d6.*\u6587\u4ef6",  # read file
        "\u67e5\u5929\u6c14",  # check weather
        "\u5929\u6c14",  # weather
        r"sql",
        "\u6570\u636e\u5e93",  # database
        "\u8ba1\u7b97",  # calculate
        "\u751f\u6210.*\u62a5\u544a",  # generate report
        "\u5b8c\u6210.*\u4efb\u52a1",  # complete task
        "\u5206\u6b65\u9aa4",  # step by step
    )

    def route(
        self,
        query: str,
        allowed_intents: Optional[Iterable[IntentType]] = None,
    ) -> IntentDecision:
        text = query.strip().lower()
        allowed: Set[IntentType] = set(allowed_intents or IntentType)

        if not text:
            return self._coerce(
                IntentDecision(
                    intent=IntentType.CHAT,
                    confidence=0.4,
                    reason="empty query defaults to chat",
                    signals=[],
                ),
                allowed,
            )

        rag_hits = self._match_patterns(text, self.RAG_PATTERNS)
        agent_hits = self._match_patterns(text, self.AGENT_PATTERNS)

        if agent_hits and len(agent_hits) >= len(rag_hits):
            return self._coerce(
                IntentDecision(
                    intent=IntentType.AGENT,
                    confidence=min(0.9, 0.55 + 0.1 * len(agent_hits)),
                    reason="tool/action signals detected",
                    signals=agent_hits,
                ),
                allowed,
            )

        if rag_hits:
            return self._coerce(
                IntentDecision(
                    intent=IntentType.RAG,
                    confidence=min(0.85, 0.55 + 0.08 * len(rag_hits)),
                    reason="knowledge retrieval signals detected",
                    signals=rag_hits,
                ),
                allowed,
            )

        return self._coerce(
            IntentDecision(
                intent=IntentType.CHAT,
                confidence=0.6,
                reason="no retrieval or tool signals detected",
                signals=[],
            ),
            allowed,
        )

    def _match_patterns(self, text: str, patterns: Iterable[str]) -> list[str]:
        return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]

    def _coerce(self, decision: IntentDecision, allowed: Set[IntentType]) -> IntentDecision:
        if decision.intent in allowed:
            return decision

        fallback = IntentType.CHAT if IntentType.CHAT in allowed else next(iter(allowed))
        return IntentDecision(
            intent=fallback,
            confidence=max(0.3, decision.confidence - 0.3),
            reason=f"{decision.reason}; coerced to {fallback.value}",
            signals=decision.signals,
        )
