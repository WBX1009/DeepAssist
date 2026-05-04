from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class AgentRunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TOOL_ERROR_EXCEEDED = "tool_error_exceeded"


class AgentTerminationReason(str, Enum):
    FINAL_ANSWER = "final_answer"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TOOL_ERROR_EXCEEDED = "tool_error_exceeded"
    REPEATED_TOOL_CALL_EXCEEDED = "repeated_tool_call_exceeded"
    LLM_ERROR = "llm_error"
    INVALID_STATE = "invalid_state"


class AgentRunConfig(BaseModel):
    max_iterations: int = Field(default=10, ge=1)
    max_tool_errors: int = Field(default=3, ge=0)
    repeated_tool_call_limit: int = Field(default=3, ge=1)
    max_self_corrections: int = Field(default=2, ge=0)


class AgentRunState(BaseModel):
    status: AgentRunStatus = AgentRunStatus.RUNNING
    termination_reason: Optional[AgentTerminationReason] = None
    iterations: int = 0
    tool_errors: int = 0
    self_corrections: int = 0
    repeated_tool_calls: Dict[str, int] = Field(default_factory=dict)
    final_answer: str = ""
    error: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status != AgentRunStatus.RUNNING

    def terminal_payload(self) -> Dict[str, object]:
        return {
            "status": self.status.value,
            "termination_reason": self.termination_reason.value
            if self.termination_reason
            else None,
            "iterations": self.iterations,
            "tool_errors": self.tool_errors,
            "self_corrections": self.self_corrections,
            "error": self.error,
        }
