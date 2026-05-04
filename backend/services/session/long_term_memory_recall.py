import json
import re
from typing import List

from backend.domain.entities.long_term_memory import LongTermMemoryItem
from backend.domain.interfaces.memory_db import BaseMemoryStore


class LongTermMemoryRecallService:
    """Recall profile-derived memory items relevant to the current query."""

    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,8}")

    def __init__(
        self,
        memory_store: BaseMemoryStore,
        max_items: int = 2,
    ):
        self.store = memory_store
        self.max_items = max_items

    def recall(self, query: str, max_items: int | None = None) -> List[LongTermMemoryItem]:
        profiles = self.store.get_all_profiles()
        if not profiles:
            return []

        candidates: List[LongTermMemoryItem] = []
        query_tokens = self._tokenize(query)

        facts = self._parse_json_list(profiles.get("user_facts", "[]"))
        for index, fact in enumerate(facts):
            score = self._score_text(query_tokens, fact, base_score=1.4)
            candidates.append(
                LongTermMemoryItem(
                    key=f"user_fact_{index}",
                    content=fact,
                    category="fact",
                    score=score,
                )
            )

        topics = self._parse_json_list(profiles.get("topics", "[]"))
        for topic in topics:
            score = self._score_text(query_tokens, topic, base_score=1.1)
            candidates.append(
                LongTermMemoryItem(
                    key=f"topic_{topic}",
                    content=f"User often discusses {topic}.",
                    category="topic",
                    score=score,
                    metadata={"topic": topic},
                )
            )

        last_user_query = str(profiles.get("last_user_query", "") or "").strip()
        if last_user_query:
            score = self._score_text(query_tokens, last_user_query, base_score=0.8)
            candidates.append(
                LongTermMemoryItem(
                    key="last_user_query",
                    content=f"Recent user query: {last_user_query}",
                    category="recency",
                    score=score,
                )
            )

        ranked = sorted(
            candidates,
            key=lambda item: (-item.score, item.key),
        )
        item_limit = max_items if max_items is not None else self.max_items
        selected = [item for item in ranked if item.score > 0][: max(0, item_limit)]
        return selected

    def _score_text(self, query_tokens: set[str], content: str, base_score: float) -> float:
        content_tokens = self._tokenize(content)
        if not content_tokens:
            return 0.0

        overlap = len(query_tokens & content_tokens)
        score = base_score + overlap * 1.25
        if overlap == 0 and self._contains_identity_signal(content):
            score += 0.35
        return score

    def _tokenize(self, text: str) -> set[str]:
        tokens = [token.lower() for token in self._TOKEN_PATTERN.findall(text or "")]
        return {token for token in tokens if token}

    def _contains_identity_signal(self, content: str) -> bool:
        lowered = content.lower()
        return any(
            needle in lowered
            for needle in (
                "i am",
                "my role",
                "engineer",
                "\u6211\u662f",
                "\u804c\u4e1a",
                "\u5de5\u7a0b\u5e08",
            )
        )

    def _parse_json_list(self, value: str) -> List[str]:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]
