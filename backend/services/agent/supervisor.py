from typing import Any, Dict, Iterator, List, Optional

from backend.domain.entities.agent_worker import (
    AgentWorkerType,
    SupervisorDecision,
)
from backend.domain.entities.intent import IntentDecision, IntentType
from backend.services.agent.intent_router import IntentRouter
from backend.services.agent.workers import BaseAgentWorker


class AgentSupervisor:
    """Lightweight supervisor that routes a task to a focused worker."""

    def __init__(
        self,
        intent_router: IntentRouter,
        chat_worker: BaseAgentWorker,
        rag_worker: Optional[BaseAgentWorker],
        tool_worker: BaseAgentWorker,
    ):
        self.intent_router = intent_router
        self.chat_worker = chat_worker
        self.rag_worker = rag_worker
        self.tool_worker = tool_worker

    def decide(self, query: str) -> SupervisorDecision:
        if self.intent_router.is_tool_inventory_query(query):
            intent = IntentDecision(
                intent=IntentType.AGENT,
                confidence=0.95,
                reason="tool inventory query detected",
                signals=["tool_inventory"],
            )
            return SupervisorDecision(
                worker=AgentWorkerType.TOOL,
                intent=intent,
                reason="tool inventory query detected; routed to tool_agent_worker",
                signals=intent.signals,
            )

        if self.intent_router.is_kb_catalog_query(query):
            intent = IntentDecision(
                intent=IntentType.AGENT,
                confidence=0.93,
                reason="knowledge-base catalog or capability query detected",
                signals=["kb_catalog"],
            )
            return SupervisorDecision(
                worker=AgentWorkerType.TOOL,
                intent=intent,
                reason="knowledge-base catalog query detected; routed to tool_agent_worker",
                signals=intent.signals,
            )

        allowed = {IntentType.CHAT, IntentType.AGENT}
        if self.rag_worker is not None:
            allowed.add(IntentType.RAG)

        intent = self.intent_router.route(query, allowed_intents=allowed)
        if intent.intent == IntentType.RAG and self.rag_worker is not None:
            worker = AgentWorkerType.RAG
        elif intent.intent == IntentType.AGENT:
            worker = AgentWorkerType.TOOL
        else:
            worker = AgentWorkerType.CHAT

        return SupervisorDecision(
            worker=worker,
            intent=intent,
            reason=f"{intent.reason}; routed to {worker.value}",
            signals=intent.signals,
        )

    def stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        decision = self.decide(query)
        yield {
            "type": "supervisor_route",
            "worker": decision.worker.value,
            "worker_kind": self._worker_kind(decision.worker),
            "intent": decision.intent.intent.value,
            "confidence": decision.intent.confidence,
            "reason": decision.reason,
            "signals": decision.signals,
        }

        worker = self._worker_for(decision.worker)
        yield from worker.stream(
            query=query,
            history=history,
            user_profile=user_profile,
            model_options=model_options,
        )

    def _worker_for(self, worker_type: AgentWorkerType) -> BaseAgentWorker:
        if worker_type == AgentWorkerType.RAG and self.rag_worker is not None:
            return self.rag_worker
        if worker_type == AgentWorkerType.TOOL:
            return self.tool_worker
        return self.chat_worker

    def _worker_kind(self, worker_type: AgentWorkerType) -> str:
        if worker_type == AgentWorkerType.RAG:
            return "rag"
        if worker_type == AgentWorkerType.TOOL:
            return "tool"
        return "chat"
