import unittest
from typing import Any

from backend.application.agent_app import AgentApplication
from backend.application.chat_app import ChatApplication
from backend.application.kb_app import KnowledgeBaseApp
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import KnowledgeBaseHealthReport
from backend.domain.entities.rag_pipeline import RAGPipelineResult
from backend.domain.entities.retrieval import Citation, RAGContextPack, RerankTrace, RetrievalResult
from backend.domain.interfaces.memory_db import BaseMemoryStore
from backend.infrastructure.tools.rag_tool import KnowledgeBaseTool
from backend.services.agent.intent_router import IntentRouter
from backend.services.context_engine import ContextEngine
from backend.services.session.manager import SessionManager


class FakeMemoryStore(BaseMemoryStore):
    def __init__(self):
        self.messages: dict[str, list[Any]] = {}
        self.profiles: dict[str, str] = {}

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


class FakeLLM:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.calls: list[list[dict[str, Any]]] = []

    def chat_stream(self, messages, model_name=None, temperature=None, top_p=None):
        self.calls.append(messages)
        for chunk in self.chunks:
            yield chunk


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
            def to_stream_data(self):
                return {"grounded": True, "warnings": []}

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
        next(stream)
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
        next(stream)
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


if __name__ == "__main__":
    unittest.main()
