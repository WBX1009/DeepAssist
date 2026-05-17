import asyncio
import json
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

    # NOTE: Legacy constant retained for backward-compatibility. 动态诊断已取代该阈值。
    _RAG_RELEVANCE_THRESHOLD = 0.1

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
        return self.session_mgr.get_session_history(session_id, limit=max_rounds)

    def delete_session(self, session_id: str) -> bool:
        return self.session_mgr.delete_session(session_id)

    def _analyze_query_intent(self, query: str) -> dict:
        default_intent = {"is_small_talk": False, "is_explicit_kb": False, "is_negative_kb": False}
        if not query.strip():
            return default_intent

        system_prompt = (
            "你是一个自然语言理解引擎。你的任务是分析用户的输入，并输出纯JSON对象，包含以下三个布尔值字段：\n"
            "1. is_small_talk: 用户是否仅仅在进行日常问候、告别或简单闲聊（如“你好”、“在吗”、“拜拜”、“谢谢”）。如果包含实质性提问，设为 false。\n"
            "2. is_explicit_kb: 用户是否明确要求根据“知识库”、“文档”、“资料”或“参考”来回答。\n"
            "3. is_negative_kb: 用户是否明确要求【不要】查知识库、不查资料或不查数据库。\n"
            "请注意：只返回JSON，不要包含任何Markdown标记（如```json）或其他说明文字。"
        )
        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.llm.chat(messages, temperature=0.0)
            content = response.content.strip()
            
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            result = json.loads(content)
            return {
                "is_small_talk": bool(result.get("is_small_talk", False)),
                "is_explicit_kb": bool(result.get("is_explicit_kb", False)),
                "is_negative_kb": bool(result.get("is_negative_kb", False)),
            }
        except Exception as exc:
            logger.error(f"LLM query intent analysis failed: {exc}")
            return default_intent

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
        response_chunks: list[str] =[]
        should_persist = False
        history_budget = max(1, history_rounds)
        try:
            mode, intent_decision = self._resolve_mode(mode, query)
            
            query_intent = self._analyze_query_intent(query)

            if intent_decision:
                yield SSEManager.format_event(
                    StreamEvent.status(
                        f"Auto-routed to {mode} ({intent_decision.reason}, confidence={intent_decision.confidence:.2f})"
                    )
                )

            if mode == "rag" and query_intent.get("is_negative_kb"):
                logger.info("User explicitly requested to bypass KB in session %s", session_id)
                yield SSEManager.format_event(
                    StreamEvent.status("已识别到不查阅知识库的指令；自动切换至快速对话模式。")
                )
                mode = "quick"

            if mode == "quick":
                logger.info("Entering quick chat mode")
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
                
                if query_intent.get("is_small_talk"):
                    logger.info("Bypassing RAG retrieval for small-talk query in session %s", session_id)
                    yield SSEManager.format_event(
                        StreamEvent.status(
                            "Detected a casual chat query; switching to general chat."
                        )
                    )
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

                rag_decision = self._decide_rag_fallback(pipeline_result, query_intent.get("is_explicit_kb", False))
                
                if rag_decision["action"] == "direct_kb_miss":
                    logger.info(
                        "RAG retrieval is insufficient for explicit KB query in session %s",
                        session_id,
                    )
                    yield SSEManager.format_event(
                        StreamEvent.status(rag_decision["status"])
                    )
                    should_persist = True
                    full_response = rag_decision["answer"]
                    yield from self._emit_buffered_text(full_response, response_chunks)
                    yield SSEManager.format_end()
                    return

                if rag_decision["action"] == "fallback_to_chat":
                    logger.info(
                        "RAG retrieval is insufficient; falling back to quick chat for session %s",
                        session_id,
                    )
                    yield SSEManager.format_event(StreamEvent.status(rag_decision["status"]))
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

                if rag_decision["action"] == "proceed_with_warning":
                    yield SSEManager.format_event(StreamEvent.status(rag_decision["status"]))

                should_persist = True
                
                rag_answer_chunks =[]
                for chunk in self.llm.chat_stream(
                    context.messages,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p,
                ):
                    if isinstance(chunk, dict):
                        if chunk.get("type") == "reasoning":
                            yield SSEManager.format_event(StreamEvent.reasoning(chunk["content"]))
                        else:
                            rag_answer_chunks.append(chunk["content"])
                    else:
                        rag_answer_chunks.append(chunk)

                rag_answer = "".join(rag_answer_chunks)

                if context.rag_context_pack:
                    guard_report = self.rag_pipeline.check_answer(
                        rag_answer,
                        pipeline_result,
                    )
                    yield SSEManager.format_event(
                        StreamEvent.answer_guard(guard_report.to_stream_data())
                    )
                    rag_answer = self._apply_rag_guard_action(
                        answer=rag_answer,
                        guard_report=guard_report,
                        pipeline_result=pipeline_result,
                        is_explicit_kb=query_intent.get("is_explicit_kb", False)
                    )
                full_response = rag_answer
                yield from self._emit_buffered_text(full_response, response_chunks)

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
            if isinstance(chunk, dict):
                if chunk.get("type") == "reasoning":
                    yield SSEManager.format_event(StreamEvent.reasoning(chunk["content"]))
                else:
                    response_chunks.append(chunk["content"])
                    yield SSEManager.format_event(StreamEvent.message_delta(chunk["content"]))
            else:
                response_chunks.append(chunk)
                yield SSEManager.format_event(StreamEvent.message_delta(chunk))

    def _emit_buffered_text(
        self,
        text: str,
        response_chunks: list[str],
        chunk_size: int = 160,
    ) -> Iterator[str]:
        if not text:
            return
        for start in range(0, len(text), chunk_size):
            chunk = text[start : start + chunk_size]
            response_chunks.append(chunk)
            yield SSEManager.format_event(StreamEvent.message_delta(chunk))

    def _should_fallback_from_rag(self, pipeline_result) -> bool:
        """
        🌟 Legacy fallback predicate based on a static relevance threshold.
        It remains for backward-compatibility but is now practically unused.
        Fallback decisions rely on dynamic diagnostics such as 'reason_code' in retrieval metadata.
        """
        diagnostics = (
            pipeline_result.retrieval_result.metadata.get("diagnostics", {})
            if pipeline_result.retrieval_result.metadata
            else {}
        )
        reason_code = diagnostics.get("reason_code")
        return reason_code in {"no_hits", "low_relevance", "all_channels_failed"}

    def _decide_rag_fallback(self, pipeline_result, is_explicit_kb: bool) -> dict:
        diagnostics = (
            pipeline_result.retrieval_result.metadata.get("diagnostics", {})
            if pipeline_result.retrieval_result.metadata
            else {}
        )
        reason_code = diagnostics.get("reason_code", "ok")
        reason_message = diagnostics.get("reason_message", "retrieval_ready")

        if reason_code in {"all_channels_failed"}:
            return {
                "action": "fallback_to_chat",
                "status": "Knowledge-base retrieval is currently degraded; falling back to general chat.",
            }

        if reason_code in {"no_hits", "low_relevance"}:
            if is_explicit_kb:
                return {
                    "action": "direct_kb_miss",
                    "status": "The knowledge-base retrieval result is insufficient for a source-grounded answer.",
                    "answer": self._render_kb_miss_answer(reason_message),
                }
            return {
                "action": "fallback_to_chat",
                "status": "No sufficiently relevant KB context was found; falling back to general chat.",
            }

        if reason_code in {"partial_channel_failure", "single_channel_recall"}:
            return {
                "action": "proceed_with_warning",
                "status": "Retrieval completed with partial signal quality; grounding checks will be applied carefully.",
            }

        return {"action": "proceed", "status": ""}

    def _render_kb_miss_answer(self, reason_message: str) -> str:
        return (
            "根据当前知识库检索，暂时没有找到足够相关、可直接支撑答案的资料。"
            f"原因：{reason_message}。"
            "你可以换一个更具体的关键词，或者切换到智能对话模式获取通用回答。"
        )

    def _apply_rag_guard_action(
        self,
        answer: str,
        guard_report,
        pipeline_result,
        is_explicit_kb: bool,
    ) -> str:
        if guard_report.grounded:
            return answer

        if guard_report.recommended_action == "fallback_without_kb_claims":
            if is_explicit_kb:
                return self._render_kb_miss_answer(guard_report.reason)
            return (
                f"{answer}\n\n"
                "注：当前回答没有可靠的知识库检索证据支撑，请将其视为通用回答，而不是知识库结论。"
            ).strip()

        if guard_report.recommended_action in {
            "regenerate_with_citations",
            "regenerate_with_known_citations",
        }:
            refs = ", ".join(
                citation.ref_id for citation in pipeline_result.context_pack.citations[:3]
            ) or "无"
            return (
                "根据当前检索到的知识库片段，系统暂时无法生成带可靠引用的答案，"
                "因此已触发防幻觉降级。"
                f"可参考的检索片段编号：{refs}。"
                "请换一个更具体的问题，或明确要求“仅基于知识库并附引用回答”。"
            )

        return answer