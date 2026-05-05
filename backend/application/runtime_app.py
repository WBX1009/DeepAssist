from typing import Any

from backend.common.config import settings
from backend.domain.entities.agent_worker import AgentWorkerType
from backend.application.kb_app import KnowledgeBaseApp
from backend.services.agent.tooling import ToolRegistry


class RuntimeApplication:
    """Expose the effective runtime contract consumed by the UI layer."""

    def __init__(
        self,
        kb_app: KnowledgeBaseApp,
        tool_registry: ToolRegistry,
    ):
        self.kb_app = kb_app
        self.tool_registry = tool_registry

    def get_capabilities(self) -> dict[str, Any]:
        collections_payload = self.kb_app.list_collections()
        collections = collections_payload.get("data", [])
        if not isinstance(collections, list):
            collections = []

        model_options = [
            {
                "id": settings.LLM_CHAT_MODEL,
                "label": "DeepSeek Chat",
                "supported": True,
                "supports_streaming": True,
                "supports_tool_calls": True,
                "notes": "Default model for quick chat, RAG answering, and tool-enabled agent turns.",
            },
            {
                "id": settings.LLM_REASONER_MODEL,
                "label": "DeepSeek Reasoner",
                "supported": True,
                "supports_streaming": True,
                "supports_tool_calls": False,
                "notes": "Used for reasoning-heavy turns. Tool-enabled flows automatically fall back to the chat model when tool calls are needed.",
            },
        ]

        rag_scope_options = [
            {
                "id": "__all__",
                "label": "All connected collections",
                "kind": "all",
            }
        ]
        for item in collections:
            if not isinstance(item, dict):
                continue
            collection_name = str(item.get("collection_name") or "").strip()
            if not collection_name:
                continue
            rag_scope_options.append(
                {
                    "id": collection_name,
                    "label": collection_name,
                    "kind": "collection",
                }
            )

        return {
            "status": "success",
            "data": {
                "models": model_options,
                "settings_schema": {
                    "temperature": {"min": 0.0, "max": 2.0, "default": 0.7, "step": 0.1},
                    "top_p": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.05},
                    "history_rounds": {"min": 1, "max": 30, "default": 10, "step": 1},
                    "use_user_memory": {"default": False},
                },
                "modes": [
                    {
                        "id": "quick",
                        "title": "Fast Chat",
                        "description": "Pure LLM conversation with session memory and profile-aware context assembly.",
                        "endpoint": "/api/chat/stream",
                        "request_fields": [
                            "session_id",
                            "query",
                            "mode",
                            "model_name",
                            "temperature",
                            "top_p",
                            "history_rounds",
                            "use_user_memory",
                        ],
                    },
                    {
                        "id": "rag",
                        "title": "Knowledge Q&A",
                        "description": "Grounded retrieval over the offline Markdown knowledge bases with citations, diagnostics, and safe fallback.",
                        "endpoint": "/api/chat/stream",
                        "request_fields": [
                            "session_id",
                            "query",
                            "mode",
                            "collection_name",
                            "model_name",
                            "temperature",
                            "top_p",
                            "history_rounds",
                            "use_user_memory",
                        ],
                    },
                    {
                        "id": "agent",
                        "title": "Agent",
                        "description": "Supervisor-routed multi-step execution across chat, RAG, and tool workers with recovery traces.",
                        "endpoint": "/api/agent/stream",
                        "request_fields": [
                            "session_id",
                            "query",
                            "model_name",
                            "temperature",
                            "top_p",
                            "history_rounds",
                            "use_user_memory",
                        ],
                        "default_worker_scope": AgentWorkerType.ORCHESTRATOR.value,
                    },
                ],
                "knowledge_base": {
                    "management_default_collection": "tech_docs_kb",
                    "rag_default_scope": "__all__",
                    "agent_default_scope": "__all__",
                    "rag_scope_options": rag_scope_options,
                    "collections": collections,
                    "notes": [
                        "Knowledge Q&A mode can target all connected collections or one selected collection.",
                        "Agent retrieval always searches across all connected collections by default.",
                    ],
                },
                "tools": self.tool_registry.list_tool_specs(),
            },
        }
