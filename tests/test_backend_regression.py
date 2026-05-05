import json
import unittest
from types import SimpleNamespace
from typing import Any

from backend.application.agent_app import AgentApplication
from backend.application.chat_app import ChatApplication
from backend.application.kb_app import KnowledgeBaseApp
from backend.application.runtime_app import RuntimeApplication
from backend.domain.entities.context_window import ContextSummary
from backend.domain.entities.agent_run import AgentRunConfig
from backend.domain.entities.agent_plan import MultiAgentPlan
from backend.domain.entities.agent_worker import AgentWorkerType
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import KnowledgeBaseHealthReport
from backend.domain.entities.rag_pipeline import RAGPipelineResult
from backend.domain.entities.retrieval import Citation, RAGContextPack, RerankTrace, RetrievalResult
from backend.domain.entities.task_snapshot import TaskSnapshot
from backend.domain.entities.tooling import ToolCall
from backend.domain.interfaces.memory_db import BaseMemoryStore
from backend.infrastructure.tools.kb_catalog_tool import KnowledgeBaseCatalogTool
from backend.infrastructure.tools.rag_tool import KnowledgeBaseTool
from backend.services.agent.engine import AgentEngine
from backend.services.agent.intent_router import IntentRouter
from backend.services.agent.supervisor import AgentSupervisor
from backend.services.agent.task_decomposer import TaskDecomposer
from backend.services.agent.tooling import ToolRegistry
from backend.services.agent.workers import OrchestratorWorker, ToolAgentWorker
from backend.services.context_engine import ContextEngine
from backend.services.rag.answer_guard import SourceAwareResponseGuard
from backend.services.rag.fusion import HybridRetriever
from backend.services.rag.query_planner import QueryPlanner
from backend.services.session.manager import SessionManager


class FakeMemoryStore(BaseMemoryStore):
    def __init__(self):
        self.messages: dict[str, list[Any]] = {}
        self.profiles: dict[str, str] = {}
        self.task_snapshots: dict[str, TaskSnapshot] = {}
        self.session_summaries: dict[str, ContextSummary] = {}

    def get_history(self, session_id: str, limit: int = 10):
        return list(self.messages.get(session_id, []))[-limit:]

    def add_message(self, session_id: str, message):
        self.messages.setdefault(session_id, []).append(message)
        return True

    def clear_history(self, session_id: str) -> bool:
        self.messages.pop(session_id, None)
        return True

    def get_all_sessions(self):
        return []

    def get_profile(self, key: str):
        return self.profiles.get(key)

    def set_profile(self, key: str, value: str) -> bool:
        self.profiles[key] = value
        return True

    def get_all_profiles(self):
        return dict(self.profiles)

    def get_task_snapshot(self, session_id: str):
        return self.task_snapshots.get(session_id)

    def save_task_snapshot(self, snapshot: TaskSnapshot) -> bool:
        self.task_snapshots[snapshot.session_id] = snapshot
        return True

    def clear_task_snapshot(self, session_id: str) -> bool:
        self.task_snapshots.pop(session_id, None)
        return True

    def get_session_summary(self, session_id: str):
        return self.session_summaries.get(session_id)

    def save_session_summary(self, session_id: str, summary: ContextSummary) -> bool:
        self.session_summaries[session_id] = summary
        return True

    def clear_session_summary(self, session_id: str) -> bool:
        self.session_summaries.pop(session_id, None)
        return True


class FakeLLM:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.calls: list[list[dict[str, Any]]] = []

    def chat_stream(self, messages, model_name=None, temperature=None, top_p=None):
        self.calls.append(messages)
        for chunk in self.chunks:
            yield chunk


class FakeAgentLLM:
    def __init__(self, responses: list[Any]):
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None, model_name=None, temperature=None, top_p=None):
        self.calls.append(messages)
        return self.responses[len(self.calls) - 1]


class FakeProfileExtractor:
    def render_profile(self) -> str:
        return "- role: platform engineer"


class NoCallPipeline:
    def build_context(self, query: str, collection_name: str):
        raise AssertionError("RAG pipeline should not run for small-talk downgrade")


class FakeRAGPipeline:
    def __init__(self, score: float = 0.8):
        self.score = score
        self.calls: list[tuple[str, str]] = []

    def build_context(self, query: str, collection_name: str) -> RAGPipelineResult:
        self.calls.append((query, collection_name))
        retrieval = RetrievalResult(
            query=query,
            collection_name=collection_name,
            documents=[
                DocumentChunk(
                    id="chunk-1",
                    content="DeepSeek API Key guide",
                    metadata={"source_file": "kb.md"},
                    score=self.score,
                )
            ],
            top_k=5,
            candidate_k=10,
            metadata={
                "diagnostics": {
                    "reason_code": "ok" if self.score >= 0.35 else "low_relevance",
                    "reason_message": "retrieval_ready"
                    if self.score >= 0.35
                    else "top_document_score_below_threshold",
                    "suggested_action": "proceed_with_rag"
                    if self.score >= 0.35
                    else "fallback_to_chat",
                    "best_score": self.score,
                    "low_relevance_threshold": 0.35,
                }
            },
            rerank=RerankTrace(
                enabled=True,
                success=True,
                model="fake",
                input_count=1,
                returned=1,
            ),
        )
        context_pack = RAGContextPack(
            query=query,
            citations=[
                Citation(
                    ref_id="C1",
                    chunk_id="chunk-1",
                    content="DeepSeek API Key guide",
                    source_file="kb.md",
                    score=self.score,
                )
            ],
            rendered_context="[C1] DeepSeek API Key guide",
            budget_chars=1000,
            used_chars=28,
        )
        return RAGPipelineResult(
            query=query,
            collection_name=collection_name,
            retrieval_result=retrieval,
            context_pack=context_pack,
        )

    def check_answer(self, answer: str, pipeline_result: RAGPipelineResult):
        class Report:
            grounded = True
            recommended_action = "accept"
            reason = "answer_grounded"

            def to_stream_data(self):
                return {
                    "grounded": True,
                    "recommended_action": "accept",
                    "reason": "answer_grounded",
                    "warnings": [],
                }

        return Report()


class FakeSupervisor:
    def stream(self, query: str, history, user_profile=None, model_options=None):
        yield {"type": "message_delta", "content": "partial-agent"}


class FakeHealthInspector:
    def __init__(self):
        self.load_calls = 0
        self.inspect_calls: list[dict[str, Any]] = []
        self.cached_report = KnowledgeBaseHealthReport(
            vector_db_path="vector",
            keyword_db_path="keyword",
            checked_at="2026-05-04T16:00:00",
        )
        self.live_report = KnowledgeBaseHealthReport(
            vector_db_path="vector",
            keyword_db_path="keyword",
            checked_at="2026-05-04T16:05:00",
        )

    def load_report(self):
        self.load_calls += 1
        return self.cached_report

    def inspect(self, collections=None, repair=False, batch_size=256, persist=True):
        self.inspect_calls.append(
            {
                "collections": collections,
                "repair": repair,
                "batch_size": batch_size,
                "persist": persist,
            }
        )
        return self.live_report


class FakeKBApp:
    def list_collections(self):
        return {
            "status": "success",
            "data": [
                {
                    "collection_name": "medical_kb",
                    "file_count": 2,
                    "chunk_count": 18,
                    "stores": ["keyword", "vector"],
                },
                {
                    "collection_name": "tech_docs_kb",
                    "file_count": 4,
                    "chunk_count": 36,
                    "stores": ["keyword", "vector"],
                },
            ],
        }


class FakeEmbeddingModel:
    def embed_text(self, query: str):
        return [0.1, 0.2, 0.3]


class FakeVectorDB:
    def __init__(self, docs):
        self.docs = docs

    def search(self, collection_name, query_vector, top_k):
        return list(self.docs)[:top_k]


class FakeKeywordDB:
    def __init__(self, docs):
        self.docs = docs

    def search(self, collection_name, query, top_k):
        return list(self.docs)[:top_k]

    def list_files(self, collection_name: str = "tech_docs_kb"):
        return {
            "status": "success",
            "data": [
                {
                    "source_file": "guide.md",
                    "chunk_count": 12,
                    "consistent": True,
                }
            ],
        }


class FakeCollaboratorWorker:
    def __init__(self, content: str, extra_events: list[dict[str, Any]] | None = None):
        self.content = content
        self.extra_events = extra_events or []
        self.calls: list[dict[str, Any]] = []

    def stream(self, query: str, history, user_profile=None, model_options=None, recovery_state=None):
        self.calls.append(
            {
                "query": query,
                "history": history,
                "user_profile": user_profile,
                "model_options": model_options,
                "recovery_state": recovery_state,
            }
        )
        for event in self.extra_events:
            yield event
        yield {
            "type": "finish",
            "new_messages": [{"role": "assistant", "content": self.content}],
            "worker": "fake_worker",
        }


class FakeSynthesisLLM:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.calls: list[list[dict[str, Any]]] = []

    def chat_stream(self, messages, model_name=None, temperature=None, top_p=None):
        self.calls.append(messages)
        for chunk in self.chunks:
            yield chunk


class FakeRecoverySupervisor:
    def __init__(self):
        self.resume_calls: list[dict[str, Any]] = []

    def stream(self, query: str, history, user_profile=None, model_options=None):
        yield {"type": "message_delta", "content": "fresh-run"}
        yield {
            "type": "finish",
            "new_messages": [{"role": "assistant", "content": "fresh-run"}],
            "worker": "tool_agent_worker",
        }

    def stream_recovery(
        self,
        worker_kind: str,
        query: str,
        history,
        user_profile=None,
        model_options=None,
        recovery_state=None,
    ):
        self.resume_calls.append(
            {
                "worker_kind": worker_kind,
                "query": query,
                "history": history,
                "recovery_state": recovery_state,
            }
        )
        yield {"type": "message_delta", "content": "resumed-run"}
        yield {
            "type": "finish",
            "new_messages": [{"role": "assistant", "content": "resumed-run"}],
            "worker": worker_kind,
        }


class BackendRegressionTests(unittest.TestCase):
    def test_chat_disconnect_persists_partial_response(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        app = ChatApplication(
            llm=FakeLLM(["partial-chat", " rest"]),
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
        )

        stream = app.stream_chat("session-chat", "remember me", "quick")
        for chunk in stream:
            if '"event": "message_delta"' in chunk:
                break
        stream.close()

        saved = [message.model_dump() for message in store.messages["session-chat"]]
        self.assertEqual(saved[0]["role"], "user")
        self.assertEqual(saved[0]["content"], "remember me")
        self.assertEqual(saved[1]["role"], "assistant")
        self.assertEqual(saved[1]["content"], "partial-chat")

    def test_agent_disconnect_persists_partial_response(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        app = AgentApplication(
            agent_engine=None,
            session_manager=session_manager,
            context_engine=ContextEngine(),
            profile_extractor=FakeProfileExtractor(),
            supervisor=FakeSupervisor(),
        )

        stream = app.stream_agent_task("session-agent", "do task")
        for chunk in stream:
            if '"event": "message_delta"' in chunk:
                break
        stream.close()

        saved = [message.model_dump() for message in store.messages["session-agent"]]
        self.assertEqual(saved[0]["role"], "user")
        self.assertEqual(saved[0]["content"], "do task")
        self.assertEqual(saved[1]["role"], "assistant")
        self.assertEqual(saved[1]["content"], "partial-agent")

    def test_rag_small_talk_downgrades_without_pipeline_call(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        llm = FakeLLM(["hello"])
        app = ChatApplication(
            llm=llm,
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
            profile_extractor=FakeProfileExtractor(),
            rag_pipeline=NoCallPipeline(),
        )

        list(app.stream_chat("session-rag-smalltalk", "hello", "rag", use_user_memory=True))

        saved = [message.model_dump() for message in store.messages["session-rag-smalltalk"]]
        self.assertEqual(saved[0]["content"], "hello")
        self.assertEqual(saved[1]["content"], "hello")
        self.assertEqual(len(llm.calls), 1)

    def test_rag_prompt_includes_user_profile(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        llm = FakeLLM(["grounded"])
        app = ChatApplication(
            llm=llm,
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
            profile_extractor=FakeProfileExtractor(),
            rag_pipeline=FakeRAGPipeline(score=0.8),
        )

        list(
            app.stream_chat(
                "session-rag-profile",
                "API Key acquisition flow",
                "rag",
                collection_name="__all__",
                use_user_memory=True,
            )
        )

        self.assertTrue(llm.calls)
        self.assertIn("[User Profile]", llm.calls[0][0]["content"])

    def test_agent_knowledge_base_tool_uses_cross_collection_pipeline(self):
        pipeline = FakeRAGPipeline(score=0.8)
        tool = KnowledgeBaseTool(
            retriever=None,
            rag_pipeline=pipeline,
            collection_name="__all__",
        )

        result = tool.search("DeepSeek deployment guide")

        self.assertIn("Knowledge-base retrieval succeeded", result)
        self.assertEqual(pipeline.calls, [("DeepSeek deployment guide", "__all__")])

    def test_kb_health_prefers_cached_report_until_refresh(self):
        inspector = FakeHealthInspector()
        app = KnowledgeBaseApp(
            chunker=object(),
            embedding_model=None,
            vector_db=None,
            keyword_db=None,
            health_inspector=inspector,
        )

        cached = app.get_health_report(refresh=False)
        refreshed = app.get_health_report(refresh=True, collections=["medical_kb"])

        self.assertEqual(cached["status"], "success")
        self.assertEqual(cached["source"], "cached")
        self.assertEqual(cached["data"]["checked_at"], "2026-05-04T16:00:00")
        self.assertEqual(refreshed["status"], "success")
        self.assertEqual(refreshed["source"], "live")
        self.assertEqual(refreshed["data"]["checked_at"], "2026-05-04T16:05:00")
        self.assertEqual(
            inspector.inspect_calls,
            [
                {
                    "collections": ["medical_kb"],
                    "repair": False,
                    "batch_size": 256,
                    "persist": True,
                }
            ],
        )

    def test_priority_context_keeps_profile_turn_under_budget(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        session_id = "session-priority"

        session_manager.save_interaction(session_id, "hello", "hi")
        session_manager.save_interaction(session_id, "tell me a joke", "maybe later")
        session_manager.save_interaction(
            session_id,
            "I am a backend engineer, please answer concisely.",
            "Noted.",
        )
        session_manager.save_interaction(session_id, "random follow-up one", "ack one")
        session_manager.save_interaction(session_id, "random follow-up two", "ack two")
        session_manager.save_interaction(session_id, "latest production issue?", "latest answer")

        history = session_manager.get_chat_context(session_id, max_rounds=6)
        user_messages = [message["content"] for message in history if message["role"] == "user"]

        self.assertEqual(history[0]["role"], "system")
        self.assertIn("[Conversation Summary]", history[0]["content"])
        self.assertIn("I am a backend engineer, please answer concisely.", user_messages)
        self.assertIn("random follow-up two", user_messages)
        self.assertIn("latest production issue?", user_messages)
        self.assertNotIn("hello", user_messages)

    def test_session_history_remains_full_without_summary_injection(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        session_id = "session-history"

        session_manager.save_interaction(session_id, "first", "one")
        session_manager.save_interaction(session_id, "second", "two")
        session_manager.save_interaction(session_id, "third", "three")

        history = session_manager.get_session_history(session_id, limit=10)

        self.assertEqual([message["role"] for message in history], ["user", "assistant"] * 3)
        self.assertNotIn("[Conversation Summary]", "\n".join(message["content"] for message in history))

    def test_long_term_memory_is_recalled_into_context_window(self):
        store = FakeMemoryStore()
        store.set_profile("user_facts", '["I am a backend engineer.", "I prefer concise answers."]')
        store.set_profile("topics", '["architecture", "rag"]')
        session_manager = SessionManager(store)
        session_id = "session-memory"

        session_manager.save_interaction(session_id, "old question", "old answer")
        session_manager.save_interaction(session_id, "new question", "new answer")

        plan = session_manager.plan_chat_context(
            session_id,
            max_rounds=6,
            query="Please answer like a backend engineer and keep it concise.",
            use_long_term_memory=True,
        )
        flattened = plan.flattened_messages()
        memory_messages = [
            message["content"]
            for message in flattened
            if message["role"] == "system" and "[Long-Term Memory]" in message["content"]
        ]

        self.assertTrue(memory_messages)
        self.assertTrue(any("backend engineer" in message for message in memory_messages))
        self.assertTrue(any("concise" in message.lower() for message in memory_messages))

    def test_long_term_memory_consumes_budget_but_keeps_recent_turns(self):
        store = FakeMemoryStore()
        store.set_profile("user_facts", '["I am a backend engineer."]')
        session_manager = SessionManager(store)
        session_id = "session-memory-budget"

        session_manager.save_interaction(session_id, "turn one", "one")
        session_manager.save_interaction(session_id, "turn two", "two")
        session_manager.save_interaction(session_id, "turn three", "three")
        session_manager.save_interaction(session_id, "turn four", "four")

        plan = session_manager.plan_chat_context(
            session_id,
            max_rounds=4,
            query="As a backend engineer, help me with turn four.",
            use_long_term_memory=True,
        )
        user_messages = [
            message["content"]
            for message in plan.flattened_messages()
            if message["role"] == "user"
        ]

        self.assertTrue(plan.recalled_memories)
        self.assertIn("turn four", user_messages)

    def test_context_window_trace_data_contains_summary_and_memory(self):
        store = FakeMemoryStore()
        store.set_profile("user_facts", '["I am a backend engineer."]')
        session_manager = SessionManager(store)
        session_id = "session-trace"

        session_manager.save_interaction(session_id, "turn one", "one")
        session_manager.save_interaction(session_id, "turn two", "two")
        session_manager.save_interaction(session_id, "turn three", "three")
        session_manager.save_interaction(session_id, "turn four", "four")

        plan = session_manager.plan_chat_context(
            session_id,
            max_rounds=4,
            query="backend engineer turn four",
            use_long_term_memory=True,
        )
        trace_data = plan.to_trace_data()

        self.assertIn("budget", trace_data)
        self.assertIn("selected_turns", trace_data)
        self.assertIn("recalled_memories", trace_data)
        self.assertEqual(trace_data["recalled_memory_count"], len(plan.recalled_memories))
        self.assertTrue(trace_data["summary_injected"] or trace_data["dropped_turn_count"] == 0)

    def test_persisted_session_summary_is_injected_when_runtime_summary_is_absent(self):
        store = FakeMemoryStore()
        store.save_session_summary(
            "session-persisted-summary",
            ContextSummary(
                content="[Conversation Summary]\nPersisted earlier context.",
                source="persisted",
                source_turn_ids=["turn-0"],
                dropped_turn_count=1,
                dropped_message_count=2,
                reason_counts={"recent_turn": 1},
            ),
        )
        session_manager = SessionManager(store)
        session_manager.save_interaction("session-persisted-summary", "latest question", "latest answer")

        plan = session_manager.plan_chat_context(
            "session-persisted-summary",
            max_rounds=6,
            query="follow up",
            use_long_term_memory=False,
        )

        self.assertIsNotNone(plan.summary)
        self.assertEqual(plan.summary.source, "persisted")
        self.assertIn("Persisted earlier context", plan.summary.content)

    def test_long_term_memory_recall_matches_style_preference_synonyms(self):
        store = FakeMemoryStore()
        store.set_profile("user_facts", '["I prefer concise answers."]')
        store.set_profile("topics", '["architecture"]')
        session_manager = SessionManager(store)

        plan = session_manager.plan_chat_context(
            "session-style-memory",
            max_rounds=5,
            query="Please keep it short when you explain the architecture.",
            use_long_term_memory=True,
        )

        recalled = [memory.content for memory in plan.recalled_memories]
        self.assertTrue(recalled)
        self.assertTrue(any("concise answers" in item for item in recalled))

    def test_chat_stream_emits_context_window_trace_event(self):
        store = FakeMemoryStore()
        store.set_profile("user_facts", '["I am a backend engineer."]')
        session_manager = SessionManager(store)
        session_id = "session-chat-trace"
        session_manager.save_interaction(session_id, "older one", "one")
        session_manager.save_interaction(session_id, "older two", "two")
        session_manager.save_interaction(session_id, "older three", "three")

        app = ChatApplication(
            llm=FakeLLM(["ok"]),
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
        )

        events = list(
            app.stream_chat(
                session_id,
                "I am a backend engineer, help now",
                "quick",
                history_rounds=4,
                use_user_memory=True,
            )
        )
        payloads = []
        for chunk in events:
            if not chunk.startswith("data: "):
                continue
            raw = chunk[6:].strip()
            if not raw:
                continue
            payloads.append(json.loads(raw))

        trace_payloads = [payload for payload in payloads if payload.get("event") == "context_window_trace"]
        self.assertTrue(trace_payloads)
        self.assertIn("budget", trace_payloads[0]["data"])

    def test_tool_registry_returns_structured_argument_repair_metadata(self):
        def sample_tool(city: str, days: int) -> str:
            return f"{city}:{days}"

        registry = ToolRegistry.from_callables([sample_tool])
        call = ToolCall(id="tool-1", name="sample_tool", args={"city": "beijing"})

        result = registry.execute(call)

        self.assertFalse(result.success)
        self.assertEqual(result.metadata.get("error_type"), "invalid_arguments")
        self.assertEqual(result.metadata.get("repair_strategy"), "fix_arguments")
        self.assertIn("days", result.metadata.get("missing_required_args", []))
        self.assertTrue(result.is_retryable())

    def test_tool_registry_suggests_closest_tool_for_unknown_name(self):
        def weather_lookup(city: str) -> str:
            return city

        registry = ToolRegistry.from_callables([weather_lookup])
        call = ToolCall(id="tool-2", name="weather_lookp", args={"city": "beijing"})

        result = registry.execute(call)

        self.assertFalse(result.success)
        self.assertEqual(result.metadata.get("error_type"), "unknown_tool")
        self.assertEqual(result.metadata.get("suggested_tool"), "weather_lookup")
        self.assertEqual(result.metadata.get("repair_strategy"), "switch_tool")

    def test_agent_engine_injects_self_correction_instruction_after_tool_failure(self):
        def sample_tool(city: str, days: int) -> str:
            return f"{city}:{days}"

        first_response = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "sample_tool",
                        "arguments": json.dumps({"city": "beijing"}),
                    },
                }
            ],
            reasoning_content=None,
        )
        second_response = SimpleNamespace(
            content="Recovered with fallback answer.",
            tool_calls=[],
            reasoning_content=None,
        )
        llm = FakeAgentLLM([first_response, second_response])
        engine = AgentEngine(
            llm=llm,
            tool_registry=ToolRegistry.from_callables([sample_tool]),
            run_config=AgentRunConfig(max_iterations=3, max_self_corrections=2),
        )

        events = list(
            engine.stream_run(
                [
                    {"role": "system", "content": "agent"},
                    {"role": "user", "content": "check weather"},
                ]
            )
        )

        self.assertEqual(len(llm.calls), 2)
        second_call_messages = llm.calls[1]
        system_messages = [
            message["content"]
            for message in second_call_messages
            if message.get("role") == "system"
        ]
        self.assertTrue(
            any("[Self-Correction Instruction]" in content for content in system_messages)
        )
        self.assertTrue(any("Missing required args" in content for content in system_messages))

        self_corrections = [event for event in events if event.get("type") == "self_correction"]
        self.assertTrue(self_corrections)
        self.assertEqual(self_corrections[0]["data"].get("repair_strategy"), "fix_arguments")

    def test_agent_supervisor_routes_kb_catalog_queries_to_tool_worker(self):
        supervisor = AgentSupervisor(
            intent_router=IntentRouter(),
            chat_worker=object(),
            rag_worker=object(),
            tool_worker=object(),
        )

        decision = supervisor.decide("当前的知识库有哪些，你能根据知识库进行回答问题吗？")

        self.assertEqual(decision.worker, AgentWorkerType.TOOL)
        self.assertIn("kb_catalog", decision.signals)

    def test_tool_agent_worker_injects_registered_tool_inventory(self):
        def list_knowledge_base_collections() -> str:
            return "medical_kb, tech_docs_kb"

        llm = FakeAgentLLM(
            [
                SimpleNamespace(
                    content="I can inspect the connected knowledge bases.",
                    tool_calls=[],
                    reasoning_content=None,
                )
            ]
        )
        worker = ToolAgentWorker(
            agent_engine=AgentEngine(
                llm=llm,
                tool_registry=ToolRegistry.from_callables([list_knowledge_base_collections]),
                run_config=AgentRunConfig(max_iterations=2),
            ),
            context_engine=ContextEngine(),
        )

        list(worker.stream("你能调用哪些工具？", history=[]))

        self.assertTrue(llm.calls)
        system_messages = [
            message["content"]
            for message in llm.calls[0]
            if message.get("role") == "system"
        ]
        self.assertTrue(any("Registered tools are listed below" in item for item in system_messages))
        self.assertTrue(any("list_knowledge_base_collections" in item for item in system_messages))

    def test_kb_catalog_tool_lists_connected_collections(self):
        tool = KnowledgeBaseCatalogTool(FakeKBApp())

        result = tool.list_knowledge_base_collections()

        self.assertIn("medical_kb", result)
        self.assertIn("tech_docs_kb", result)
        self.assertIn("across all connected collections", result)

    def test_runtime_capabilities_expose_real_models_tools_and_kb_scopes(self):
        def read_local_file(file: str) -> str:
            return file

        app = RuntimeApplication(
            kb_app=FakeKBApp(),
            tool_registry=ToolRegistry.from_callables([read_local_file]),
        )

        payload = app.get_capabilities()
        data = payload.get("data", {})
        tools = data.get("tools", [])
        scopes = data.get("knowledge_base", {}).get("rag_scope_options", [])

        self.assertEqual(payload.get("status"), "success")
        self.assertTrue(any(item.get("id") == "deepseek-chat" for item in data.get("models", [])))
        self.assertTrue(any(item.get("name") == "read_local_file" for item in tools))
        self.assertTrue(any(item.get("id") == "__all__" for item in scopes))
        self.assertTrue(any(item.get("id") == "medical_kb" for item in scopes))

    def test_query_planner_emits_rewrite_metadata_for_short_technical_query(self):
        planner = QueryPlanner()

        plan = planner.plan("请问 DeepSeek API 报错怎么排查？")

        self.assertTrue(plan.metadata.get("rewrite_applied"))
        self.assertIn("domain_hint_expansion", plan.metadata.get("rewrite_notes", []))
        self.assertTrue(plan.metadata.get("domain_hints"))
        self.assertGreaterEqual(len(plan.semantic_queries), 1)

    def test_hybrid_retriever_attaches_failure_attribution_metadata(self):
        docs = [
            DocumentChunk(
                id="chunk-1",
                content="Unrelated finance report and quarterly summary",
                metadata={"source_file": "guide.md"},
                score=0.12,
            )
        ]
        retriever = HybridRetriever(
            vector_db=FakeVectorDB(docs),
            keyword_db=FakeKeywordDB([]),
            embedding_model=FakeEmbeddingModel(),
        )

        result = retriever.retrieve_with_trace("tech_docs_kb", "DeepSeek 部署报错怎么排查")

        diagnostics = result.metadata.get("diagnostics", {})
        self.assertEqual(diagnostics.get("reason_code"), "low_relevance")
        self.assertEqual(diagnostics.get("suggested_action"), "fallback_to_chat")
        self.assertTrue(diagnostics.get("rewrite_applied"))

    def test_answer_guard_recommends_regeneration_when_citations_are_missing(self):
        guard = SourceAwareResponseGuard()
        context_pack = RAGContextPack(
            query="what is DeepSeek API Key flow",
            citations=[
                Citation(
                    ref_id="C1",
                    chunk_id="chunk-1",
                    content="DeepSeek API Key guide",
                    source_file="kb.md",
                )
            ],
            rendered_context="[C1] DeepSeek API Key guide",
            budget_chars=1000,
            used_chars=28,
        )

        report = guard.check("The API key is issued in the console.", context_pack)

        self.assertFalse(report.grounded)
        self.assertEqual(report.recommended_action, "regenerate_with_citations")
        self.assertIn("answer_missing_citations", report.warnings)

    def test_explicit_kb_query_with_low_relevance_returns_safe_kb_miss_answer(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        llm = FakeLLM(["hallucinated answer"])
        app = ChatApplication(
            llm=llm,
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
            rag_pipeline=FakeRAGPipeline(score=0.1),
        )

        events = list(
            app.stream_chat(
                "session-rag-kb-miss",
                "请根据知识库回答 DeepSeek API Key 的获取流程，并给出引用",
                "rag",
                collection_name="__all__",
            )
        )
        payloads = [
            json.loads(chunk[6:].strip())
            for chunk in events
            if chunk.startswith("data: ") and chunk[6:].strip()
        ]
        answer_text = "".join(
            payload.get("content", "")
            for payload in payloads
            if payload.get("event") == "message_delta"
        )

        self.assertEqual(len(llm.calls), 0)
        self.assertIn("根据当前知识库检索", answer_text)


    def test_agent_engine_assesses_and_dedupes_duplicate_tool_calls_in_one_iteration(self):
        execution_counter = {"count": 0}

        def sample_tool(city: str) -> str:
            execution_counter["count"] += 1
            return f"weather:{city}"

        first_response = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "sample_tool",
                        "arguments": json.dumps({"city": "beijing"}),
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "sample_tool",
                        "arguments": json.dumps({"city": "beijing"}),
                    },
                },
            ],
            reasoning_content=None,
        )
        second_response = SimpleNamespace(
            content="Done.",
            tool_calls=[],
            reasoning_content=None,
        )
        engine = AgentEngine(
            llm=FakeAgentLLM([first_response, second_response]),
            tool_registry=ToolRegistry.from_callables([sample_tool]),
            run_config=AgentRunConfig(max_iterations=3, max_self_corrections=2),
        )

        events = list(
            engine.stream_run(
                [
                    {"role": "system", "content": "agent"},
                    {"role": "user", "content": "check weather twice"},
                ]
            )
        )

        self.assertEqual(execution_counter["count"], 1)
        plan_events = [event for event in events if event.get("type") == "plan_assessment"]
        self.assertTrue(plan_events)
        self.assertEqual(plan_events[0]["data"].get("recommended_mode"), "simplify_plan")
        self.assertIn(
            "duplicate_tool_call_in_iteration",
            plan_events[0]["data"].get("warnings", []),
        )

    def test_agent_engine_escalates_recovery_to_partial_result_finalize_after_repeated_failure(self):
        def fetch_fact() -> str:
            return "partial fact"

        def failing_tool(target: str) -> str:
            raise RuntimeError("temporary timeout while contacting upstream")

        first_response = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "fetch_fact", "arguments": json.dumps({})},
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "failing_tool",
                        "arguments": json.dumps({"target": "alpha"}),
                    },
                },
            ],
            reasoning_content=None,
        )
        second_response = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-3",
                    "type": "function",
                    "function": {
                        "name": "failing_tool",
                        "arguments": json.dumps({"target": "alpha"}),
                    },
                }
            ],
            reasoning_content=None,
        )
        third_response = SimpleNamespace(
            content="Recovered with partial results.",
            tool_calls=[],
            reasoning_content=None,
        )
        engine = AgentEngine(
            llm=FakeAgentLLM([first_response, second_response, third_response]),
            tool_registry=ToolRegistry.from_callables([fetch_fact, failing_tool]),
            run_config=AgentRunConfig(max_iterations=4, max_self_corrections=3),
        )

        events = list(
            engine.stream_run(
                [
                    {"role": "system", "content": "agent"},
                    {"role": "user", "content": "solve with tools"},
                ]
            )
        )

        recovery_events = [event for event in events if event.get("type") == "failure_recovery"]
        self.assertGreaterEqual(len(recovery_events), 2)
        self.assertEqual(
            recovery_events[-1]["data"].get("action"),
            "finalize_with_partial_results",
        )
        self.assertEqual(
            recovery_events[-1]["data"].get("successful_tool_calls"),
            1,
        )

    def test_agent_supervisor_routes_complex_mixed_task_to_orchestrator(self):
        supervisor = AgentSupervisor(
            intent_router=IntentRouter(),
            chat_worker=object(),
            rag_worker=object(),
            tool_worker=object(),
            orchestrator_worker=object(),
            task_decomposer=TaskDecomposer(),
        )

        decision = supervisor.decide(
            "请根据知识库检索 DeepSeek 文档，然后调用 Python 做计算，最后综合总结。"
        )

        self.assertEqual(decision.worker, AgentWorkerType.ORCHESTRATOR)
        self.assertTrue(any(signal in decision.signals for signal in ["然后", "综合", "调用"]))

    def test_orchestrator_worker_decomposes_and_synthesizes_collaborator_results(self):
        rag_worker = FakeCollaboratorWorker(
            "RAG evidence: API Key is created in console.",
            extra_events=[
                {
                    "type": "retrieval_trace",
                    "data": {"hit_count": 2, "candidate_k": 10, "fusion": "weighted_rrf"},
                }
            ],
        )
        tool_worker = FakeCollaboratorWorker("Tool result: computed delta is 4.4.")
        chat_worker = FakeCollaboratorWorker("Chat framing: answer concisely.")
        llm = FakeSynthesisLLM(["Final", " synthesized", " answer"])
        worker = OrchestratorWorker(
            llm=llm,
            chat_worker=chat_worker,
            rag_worker=rag_worker,
            tool_worker=tool_worker,
            task_decomposer=TaskDecomposer(),
        )

        events = list(
            worker.stream(
                query="请根据知识库检索 DeepSeek 文档，然后调用 Python 做计算，最后综合总结。",
                history=[],
            )
        )

        plan_events = [event for event in events if event.get("type") == "multi_agent_plan"]
        collaborator_events = [
            event for event in events if event.get("type") == "collaborator_trace"
        ]
        final_chunks = [
            event.get("content", "")
            for event in events
            if event.get("type") == "message_delta"
        ]

        self.assertTrue(plan_events)
        self.assertEqual(plan_events[0]["data"].get("task_count"), 2)
        self.assertGreaterEqual(len(collaborator_events), 4)
        self.assertEqual("".join(final_chunks), "Final synthesized answer")
        self.assertTrue(llm.calls)
        synthesis_prompt = llm.calls[0][1]["content"]
        self.assertIn("RAG evidence", synthesis_prompt)
        self.assertIn("Tool result", synthesis_prompt)

    def test_agent_application_resumes_interrupted_task_from_snapshot(self):
        store = FakeMemoryStore()
        store.save_task_snapshot(
            TaskSnapshot(
                session_id="session-recover",
                query="compute stats",
                route_worker="tool_agent_worker",
                status="interrupted",
                payload={
                    "messages": [
                        {"role": "system", "content": "agent"},
                        {"role": "user", "content": "compute stats"},
                    ],
                    "lifecycle_state": {"iterations": 1},
                },
            )
        )
        supervisor = FakeRecoverySupervisor()
        app = AgentApplication(
            agent_engine=None,
            session_manager=SessionManager(store),
            context_engine=ContextEngine(),
            profile_extractor=FakeProfileExtractor(),
            supervisor=supervisor,
        )

        events = list(app.stream_agent_task("session-recover", "继续"))
        payloads = [
            json.loads(chunk[6:].strip())
            for chunk in events
            if chunk.startswith("data: ") and chunk[6:].strip()
        ]

        recovery_events = [payload for payload in payloads if payload.get("event") == "task_recovery"]
        self.assertTrue(recovery_events)
        self.assertEqual(supervisor.resume_calls[0]["worker_kind"], "tool_agent_worker")
        self.assertEqual(supervisor.resume_calls[0]["query"], "compute stats")
        self.assertIsNone(store.get_task_snapshot("session-recover"))

    def test_orchestrator_worker_resumes_from_saved_progress(self):
        rag_worker = FakeCollaboratorWorker("RAG evidence: API Key is created in console.")
        tool_worker = FakeCollaboratorWorker("Tool result: computed delta is 4.4.")
        chat_worker = FakeCollaboratorWorker("Chat framing: answer concisely.")
        llm = FakeSynthesisLLM(["Recovered final answer"])
        worker = OrchestratorWorker(
            llm=llm,
            chat_worker=chat_worker,
            rag_worker=rag_worker,
            tool_worker=tool_worker,
            task_decomposer=TaskDecomposer(),
        )
        recovery_state = {
            "plan": MultiAgentPlan(
                tasks=[
                    {
                        "task_id": "task-rag-1",
                        "title": "Collect grounded knowledge-base evidence",
                        "worker": AgentWorkerType.RAG,
                        "query": "query",
                        "rationale": "rag",
                    },
                    {
                        "task_id": "task-tool-1",
                        "title": "Execute tools and gather operational results",
                        "worker": AgentWorkerType.TOOL,
                        "query": "query",
                        "rationale": "tool",
                    },
                ]
            ).model_dump(),
            "collaborator_results": [
                {
                    "task_id": "task-rag-1",
                    "title": "Collect grounded knowledge-base evidence",
                    "worker": AgentWorkerType.RAG.value,
                    "answer": "Recovered RAG evidence",
                }
            ],
            "next_task_index": 1,
        }

        events = list(
            worker.stream(
                query="complex task",
                history=[],
                recovery_state=recovery_state,
            )
        )

        self.assertEqual(len(rag_worker.calls), 0)
        self.assertEqual(len(tool_worker.calls), 1)
        self.assertTrue(any(event.get("type") == "task_snapshot" for event in events))
        final_chunks = [
            event.get("content", "")
            for event in events
            if event.get("type") == "message_delta"
        ]
        self.assertEqual("".join(final_chunks), "Recovered final answer")


if __name__ == "__main__":
    unittest.main()
