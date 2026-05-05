from typing import Any, Dict, List

from pydantic import BaseModel, Field

from backend.domain.entities.agent_worker import AgentWorkerType


class CollaboratorTask(BaseModel):
    """One decomposed subtask assigned to a focused collaborator worker."""

    task_id: str
    title: str
    worker: AgentWorkerType
    query: str
    rationale: str

    def to_trace_data(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class MultiAgentPlan(BaseModel):
    """Structured orchestration plan for a complex agent task."""

    mode: str = "sequential_collaboration"
    complexity: str = "medium"
    signals: List[str] = Field(default_factory=list)
    tasks: List[CollaboratorTask] = Field(default_factory=list)

    def to_trace_data(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "complexity": self.complexity,
            "signal_count": len(self.signals),
            "signals": list(self.signals),
            "task_count": len(self.tasks),
            "tasks": [task.to_trace_data() for task in self.tasks],
        }
