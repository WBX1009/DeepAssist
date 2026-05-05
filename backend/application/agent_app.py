import asyncio
import re
from typing import Any, Dict, Iterator, Optional

from backend.common.logger import get_logger
from backend.domain.entities.stream_event import StreamEvent
from backend.domain.entities.task_snapshot import TaskSnapshot
from backend.services.agent.engine import AgentEngine
from backend.services.agent.supervisor import AgentSupervisor
from backend.services.context_engine import ContextEngine
from backend.services.profile_extractor import ProfileExtractor
from backend.services.session.manager import SessionManager
from backend.services.streaming.sse_manager import SSEManager

logger = get_logger(__name__)

RECOVERY_QUERY_PATTERN = re.compile(
    r"^(继续|接着|恢复|继续刚才|接着刚才|resume|continue)",
    flags=re.IGNORECASE,
)


class AgentApplication:
    """Agent workflow orchestrator."""

    def __init__(
        self,
        agent_engine: AgentEngine,
        session_manager: SessionManager,
        context_engine: ContextEngine,
        profile_extractor: ProfileExtractor,
        supervisor: AgentSupervisor | None = None,
    ):
        self.engine = agent_engine
        self.session_mgr = session_manager
        self.context_engine = context_engine
        self.profile_extractor = profile_extractor
        self.supervisor = supervisor

    def stream_agent_task(
        self,
        session_id: str,
        query: str,
        use_user_memory: bool = False,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        history_rounds: int = 10,
    ) -> Iterator[str]:
        final_answer_accumulated = ""
        persist_messages: list[dict] | None = None
        should_persist_partial = False
        history_budget = max(1, history_rounds)
        latest_snapshot: TaskSnapshot | None = None
        persisted_query = query
        try:
            logger.info("Entering agent mode for session %s", session_id)

            context_plan = self.session_mgr.plan_chat_context(
                session_id,
                max_rounds=history_budget,
                query=query,
                use_long_term_memory=use_user_memory,
            )
            history = context_plan.flattened_messages()
            yield SSEManager.format_event(
                StreamEvent.context_window_trace(context_plan.to_trace_data())
            )
            if context_plan.recalled_memories:
                yield SSEManager.format_event(
                    StreamEvent.status(
                        f"Recalled {len(context_plan.recalled_memories)} long-term memory item(s)."
                    )
                )
            if context_plan.summary is not None:
                yield SSEManager.format_event(
                    StreamEvent.status(
                        f"Compressed {context_plan.summary.dropped_turn_count} earlier turns into a continuity summary."
                    )
                )
            user_profile = (
                self.profile_extractor.render_profile() if use_user_memory else None
            )
            model_options: Dict[str, Any] = {
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
            }
            recovery_snapshot = self.session_mgr.get_task_snapshot(session_id)
            should_resume = self._should_resume_task(query, recovery_snapshot)
            if should_resume and recovery_snapshot is not None:
                latest_snapshot = recovery_snapshot
                persisted_query = recovery_snapshot.query
                yield SSEManager.format_event(
                    StreamEvent.task_recovery(recovery_snapshot.to_trace_data())
                )
                event_stream = self.supervisor.stream_recovery(
                    worker_kind=recovery_snapshot.route_worker,
                    query=recovery_snapshot.query,
                    history=history,
                    user_profile=user_profile,
                    model_options=model_options,
                    recovery_state=recovery_snapshot.payload,
                )
            else:
                event_stream = (
                    self.supervisor.stream(
                        query,
                        history,
                        user_profile=user_profile,
                        model_options=model_options,
                    )
                    if self.supervisor
                    else self._stream_legacy_agent(
                        query,
                        history,
                        use_user_memory,
                        user_profile,
                        model_options,
                    )
                )

            for event in event_stream:
                event_type = event.get("type")

                if event_type == "supervisor_route":
                    yield SSEManager.format_event(
                        StreamEvent(
                            event="supervisor_route",
                            data={
                                "worker": event.get("worker"),
                                "worker_kind": event.get("worker_kind"),
                                "intent": event.get("intent"),
                                "confidence": event.get("confidence"),
                                "reason": event.get("reason"),
                                "signals": event.get("signals", []),
                            },
                        )
                    )

                elif event_type == "status":
                    yield SSEManager.format_event(StreamEvent.status(event.get("content", "")))

                elif event_type == "reasoning":
                    yield SSEManager.format_event(StreamEvent.reasoning(event.get("content", "")))

                elif event_type == "task_snapshot":
                    latest_snapshot = self._build_task_snapshot(
                        session_id=session_id,
                        query=recovery_snapshot.query if should_resume and recovery_snapshot else query,
                        event_data=event.get("data", {}) or {},
                    )
                    self.session_mgr.save_task_snapshot(latest_snapshot)

                elif event_type == "message_delta":
                    delta = event.get("content", "")
                    final_answer_accumulated += delta
                    should_persist_partial = True
                    yield SSEManager.format_event(StreamEvent.message_delta(delta))

                elif event_type == "retrieval_trace":
                    yield SSEManager.format_event(
                        StreamEvent.retrieval_trace(event.get("data", {}))
                    )

                elif event_type == "multi_agent_plan":
                    yield SSEManager.format_event(
                        StreamEvent.multi_agent_plan(event.get("data", {}))
                    )

                elif event_type == "collaborator_trace":
                    yield SSEManager.format_event(
                        StreamEvent.collaborator_trace(event.get("data", {}))
                    )

                elif event_type == "citation_trace":
                    yield SSEManager.format_event(
                        StreamEvent.citation_trace(event.get("data", {}))
                    )

                elif event_type == "answer_guard":
                    yield SSEManager.format_event(
                        StreamEvent.answer_guard(event.get("data", {}))
                    )

                elif event_type == "self_correction":
                    data = event.get("data", {}) or {}
                    yield SSEManager.format_event(
                        StreamEvent.self_correction(
                            message=event.get("content", ""),
                            data={
                                "tool_call_id": event.get("tool_call_id"),
                                "name": event.get("name"),
                                "error": event.get("error"),
                                **data,
                                "state": event.get("state", {}),
                            },
                        )
                    )

                elif event_type == "failure_recovery":
                    yield SSEManager.format_event(
                        StreamEvent.failure_recovery(
                            message=event.get("content", ""),
                            data={
                                "tool_call_id": event.get("tool_call_id"),
                                "name": event.get("name"),
                                "error": event.get("error"),
                                **(event.get("data", {}) or {}),
                                "state": event.get("state", {}),
                            },
                        )
                    )

                elif event_type == "plan_assessment":
                    yield SSEManager.format_event(
                        StreamEvent.plan_assessment(event.get("data", {}))
                    )

                elif event_type == "task_recovery":
                    yield SSEManager.format_event(
                        StreamEvent.task_recovery(event.get("data", {}))
                    )

                elif event_type == "tool_call":
                    yield SSEManager.format_event(
                        StreamEvent.tool_call(
                            name=event.get("name", ""),
                            args=event.get("args", {}),
                            tool_call_id=event.get("tool_call_id"),
                        )
                    )

                elif event_type == "tool_result":
                    yield SSEManager.format_event(
                        StreamEvent.tool_result(
                            name=event.get("name", ""),
                            content=str(event.get("content", "")),
                            tool_call_id=event.get("tool_call_id"),
                            success=event.get("success"),
                            error=event.get("error"),
                            metadata=event.get("metadata", {}),
                        )
                    )

                elif event_type == "final_answer":
                    final_answer_accumulated = event.get("content", "")
                    should_persist_partial = True
                    yield SSEManager.format_event(
                        StreamEvent(
                            event="final_answer",
                            content=final_answer_accumulated,
                            data=event.get("state", {}),
                        )
                    )

                elif event_type == "error":
                    yield SSEManager.format_event(
                        StreamEvent(
                            event="error",
                            message=event.get("content", ""),
                            data=event.get("state", {}),
                        )
                    )

                elif event_type == "finish":
                    new_messages = event.get("new_messages", [])
                    persist_messages = [{"role": "user", "content": persisted_query}] + new_messages
                    self.session_mgr.clear_task_snapshot(session_id)
                    yield SSEManager.format_end()
                    return

        except (GeneratorExit, asyncio.CancelledError):
            logger.warning("Agent stream closed early for session %s", session_id)
            if latest_snapshot is not None:
                self.session_mgr.save_task_snapshot(
                    latest_snapshot.model_copy(update={"status": "interrupted"})
                )
            raise
        except Exception as exc:
            logger.error("Agent workflow failed: %s", exc)
            if latest_snapshot is not None:
                self.session_mgr.save_task_snapshot(
                    latest_snapshot.model_copy(update={"status": "failed"})
                )
            yield SSEManager.format_error(str(exc))
            yield SSEManager.format_end()
        finally:
            if persist_messages:
                self.session_mgr.add_messages(session_id, persist_messages)
                logger.info("Persisted %s agent trace messages", len(persist_messages))
            elif should_persist_partial and (query.strip() or final_answer_accumulated.strip()):
                self.session_mgr.save_interaction(
                    session_id,
                    persisted_query,
                    final_answer_accumulated,
                )

    def _stream_legacy_agent(
        self,
        query: str,
        history,
        use_user_memory: bool,
        user_profile: str | None,
        model_options: Optional[Dict[str, Any]] = None,
    ):
        context = self.context_engine.build_agent_context(
            query=query,
            history=history,
            use_user_memory=use_user_memory,
            user_profile=user_profile,
        )
        yield from self.engine.stream_run(context.messages, model_options=model_options)

    def _should_resume_task(
        self,
        query: str,
        snapshot: TaskSnapshot | None,
    ) -> bool:
        if snapshot is None or snapshot.status not in {"running", "interrupted", "failed"}:
            return False
        normalized_query = " ".join((query or "").split()).strip()
        if not normalized_query:
            return False
        if normalized_query == snapshot.query:
            return True
        return bool(RECOVERY_QUERY_PATTERN.search(normalized_query))

    def _build_task_snapshot(
        self,
        session_id: str,
        query: str,
        event_data: Dict[str, Any],
    ) -> TaskSnapshot:
        route_worker = str(event_data.get("route_worker") or "tool_agent_worker")
        status = str(event_data.get("status") or "running")
        payload = {key: value for key, value in event_data.items() if key not in {"route_worker", "status"}}
        return TaskSnapshot(
            session_id=session_id,
            query=query,
            route_worker=route_worker,
            status=status,
            payload=payload,
        )
