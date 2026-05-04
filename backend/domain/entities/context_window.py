from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ContextPriorityBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContextTurn(BaseModel):
    """A coherent conversation turn used for context selection."""

    turn_id: str
    started_at_index: int = Field(ge=0)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_cost: int = Field(default=0, ge=0)
    priority_score: int = Field(default=0)
    priority_band: ContextPriorityBand = Field(default=ContextPriorityBand.LOW)
    pinned: bool = Field(default=False)
    reasons: List[str] = Field(default_factory=list)

    def flattened_messages(self) -> List[Dict[str, Any]]:
        return list(self.messages)


class ContextWindowPlan(BaseModel):
    """Priority-based context window selection result."""

    budget: int = Field(default=0, ge=0)
    selected_turns: List[ContextTurn] = Field(default_factory=list)
    dropped_turns: List[ContextTurn] = Field(default_factory=list)

    @property
    def selected_cost(self) -> int:
        return sum(turn.estimated_cost for turn in self.selected_turns)

    def flattened_messages(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for turn in sorted(self.selected_turns, key=lambda item: item.started_at_index):
            messages.extend(turn.flattened_messages())
        return messages
