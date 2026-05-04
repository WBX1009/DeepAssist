import asyncio
import re
from typing import Iterator, Optional, Tuple

from backend.common.logger import get_logger
from backend.domain.entities.intent import IntentDecision, IntentType
from backend.domain.entities.stream_event import StreamEvent
from backend.domain.interfaces.llm import BaseLLM
from backend.services.agent.intent_router import IntentRouter
from backend.services.context_engine import ContextEngine
from backend.services.profile_extractor import ProfileExtractor
from backend.services.rag.fusion import HybridRetriever
from backend.services.rag.pipeline import RAGPipeline
from backend.services.session.manager import SessionManager
from backend.services.streaming.sse_manager import SSEManager

logger = get_logger(__name__)


class ChatApplication:
    """Chat workflow orchestrator for quick chat and RAG chat modes."""

    _SMALL_TALK_PATTERN = re.compile(
        "^(hi|hello|hey|yo|"
        "\u4f60\u597d|\u60a8\u597d|\u55e8|\u54c8\u55bd|"
        "\u5728\u5417|\u5728\u4e0d\u5728|"
        "\u65e9\u4e0a\u597d|\u4e0a\u5348\u597d|\u4e2d\u5348\u597d|\u4e0b\u5348\u597d|\u665a\u4e0a\u597d|"
        "\u8c22\u8c22|\u591a\u8c22|\u518d\u89c1|\u62dc\u62dc|bye)$",
        flags=re.IGNORECASE,
    )
    _RAG_RELEVANCE_THRESHOLD = 0.35

    def __init__(
        self,
        llm: BaseLLM,
        session_manager: SessionManager,
        context_engine: ContextEngine,
        intent_router: IntentRouter,
        profile_extractor: Optional[ProfileExtractor] = None,
        retriever: Optional[HybridRetriever] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
    ):
        self.llm = llm
        self.session_mgr = session_manager
        self.context_engine = context_engine
        self.intent_router = intent_router
        self.profile_extractor = profile_extractor
        self.retriever = retriever
        self.rag_pipeline = rag_pipeline or (RAGPipeline(retriever) if retriever else None)

    def list_sessions(self) -> list[dict]:
        return self.session_mgr.list_sessions()

    def get_history(self, session_id: str, max_rounds: int = 50) -> list[dict]:
        return self.session_mgr.get_chat_context(session_id, max_rounds=max_rounds)

    def delete_session(self, session_id: str) -> bool:
        return self.session_mgr.delete_session(session_id)

    def stream_chat(
        self,
        session_id: str,
        query: str,
        mode: str,
        collection_name: str = "tech_docs_kb",
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        history_rounds: int = 10,
        use_user_memory: bool = False,
    ) -> Iterator[str]:
        full_response = ""
        response_chunks: list[str] = []
        should_persist = False
        history_budget = max(1, history_rounds)
        try:
            mode, intent_decision = self._resolve_mode(mode, query)
            if intent_decision:
                yield SSEManager.format_event(
                    StreamEvent.status(
                        f"Auto-routed to {mode} ({intent_decision.reason}, confidence={intent_decision.confidence:.2f})"
                    )
                )

            if mode == "quick":
                logger.info("Entering quick chat mode")
                history = self.session_mgr.get_chat_context(
                    session_id,
                    max_rounds=history_budget,
                )
                user_profile = self._render_user_profile(use_user_memory)
                context = self.context_engine.build_quick_context(
                    query=query,
                    history=history,
                    user_profile=user_profile,
                )
                should_persist = True
                yield from self._stream_llm_response(
                    context.messages,
                    response_chunks,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p,
                )
                full_response = "".join(response_chunks)
                yield SSEManager.format_end()
                return

            if mode == "rag":
                logger.info("Entering RAG chat mode for session %s", session_id)
                if self._is_small_talk_query(query):
                    logger.info("Bypassing RAG retrieval for small-talk query in session %s", session_id)
                    yield SSEManager.format_event(
                        StreamEvent.status(
                            "Detected a casual chat query; switching to general chat."
                        )
                    )
                    history = self.session_mgr.get_chat_context(
                        session_id,
                        max_rounds=history_budget,
                    )
                    user_profile = self._render_user_profile(use_user_memory)
                    fallback_context = self.context_engine.build_quick_context(
                        query=query,
                        history=history,
                        user_profile=user_profile,
                    )
                    should_persist = True
                    yield from self._stream_llm_response(
                        fallback_context.messages,
                        response_chunks,
                        model_name=model_name,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    full_response = "".join(response_chunks)
                    yield SSEManager.format_end()
                    return

                if self.rag_pipeline is None:
                    yield SSEManager.format_error(
                        "RAG components are unavailable. Please check the embedding model and databases."
                    )
                    yield SSEManager.format_end()
                    return

                history = self.session_mgr.get_chat_context(
                    session_id,
                    max_rounds=history_budget,
                )
                user_profile = self._render_user_profile(use_user_memory)
                pipeline_result = self.rag_pipeline.build_context(
                    query=query,
                    collection_name=collection_name,
                )
                context = self.context_engine.build_rag_context_from_pipeline(
                    history=history,
                    pipeline_result=pipeline_result,
                    user_profile=user_profile,
                )
                if context.retrieval_result:
                    yield SSEManager.format_event(
                        StreamEvent.retrieval_trace(
                            context.retrieval_result.to_stream_data()
                        )
                    )
                if context.rag_context_pack:
                    yield SSEManager.format_event(
                        StreamEvent.citation_trace(
                            context.rag_context_pack.to_stream_data()
                        )
                    )

                if self._should_fallback_from_rag(pipeline_result):
                    logger.info(
                        "RAG relevance is insufficient; falling back to quick chat for session %s",
                        session_id,
                    )
                    yield SSEManager.format_event(
                        StreamEvent.status(
                            "No sufficiently relevant KB context was found; falling back to general chat."
                        )
                    )
                    fallback_context = self.context_engine.build_quick_context(
                        query=query,
                        history=history,
                        user_profile=user_profile,
                    )
                    should_persist = True
                    yield from self._stream_llm_response(
                        fallback_context.messages,
                        response_chunks,
                        model_name=model_name,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    full_response = "".join(response_chunks)
                    yield SSEManager.format_end()
                    return

                should_persist = True
                yield from self._stream_llm_response(
                    context.messages,
                    response_chunks,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p,
                )
                full_response = "".join(response_chunks)

                if context.rag_context_pack:
                    guard_report = self.rag_pipeline.check_answer(
                        full_response,
                        pipeline_result,
                    )
                    yield SSEManager.format_event(
                        StreamEvent.answer_guard(guard_report.to_stream_data())
                    )

                yield SSEManager.format_end()
                return

            yield SSEManager.format_error(f"Unknown chat mode: {mode}")
            yield SSEManager.format_end()

        except (GeneratorExit, asyncio.CancelledError):
            logger.warning("Chat stream closed early for session %s", session_id)
            raise
        except Exception as exc:
            logger.error("Streaming chat failed: %s", exc)
            yield SSEManager.format_error(str(exc))
            yield SSEManager.format_end()
        finally:
            if response_chunks and not full_response:
                full_response = "".join(response_chunks)
            if should_persist and (query.strip() or full_response.strip()):
                self.session_mgr.save_interaction(session_id, query, full_response)

    def _resolve_mode(self, mode: str, query: str) -> Tuple[str, Optional[IntentDecision]]:
        normalized = (mode or "quick").lower()
        if normalized != "auto":
            return normalized, None

        decision = self.intent_router.route(
            query,
            allowed_intents={IntentType.CHAT, IntentType.RAG},
        )
        logger.info(
            "Auto chat intent resolved to %s with confidence %.2f: %s",
            decision.intent.value,
            decision.confidence,
            decision.reason,
        )
        return ("rag" if decision.intent == IntentType.RAG else "quick"), decision

    def _render_user_profile(self, enabled: bool) -> Optional[str]:
        if not enabled or self.profile_extractor is None:
            return None
        profile = self.profile_extractor.render_profile()
        return profile or None

    def _stream_llm_response(
        self,
        messages,
        response_chunks: list[str],
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        for chunk in self.llm.chat_stream(
            messages,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
        ):
            response_chunks.append(chunk)
            yield SSEManager.format_event(StreamEvent.message_delta(chunk))

    def _is_small_talk_query(self, query: str) -> bool:
        normalized = re.sub(r"\s+", "", query or "").strip().lower()
        return bool(normalized and self._SMALL_TALK_PATTERN.fullmatch(normalized))

    def _should_fallback_from_rag(self, pipeline_result) -> bool:
        documents = pipeline_result.retrieval_result.documents
        if not documents:
            return True

        best_score = documents[0].score or 0.0
        return best_score < self._RAG_RELEVANCE_THRESHOLD
