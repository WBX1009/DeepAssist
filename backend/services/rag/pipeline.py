from backend.domain.entities.answer import AnswerGroundingReport
from backend.domain.entities.rag_pipeline import RAGPipelineResult
from backend.domain.entities.retrieval import RetrievalResult
from backend.services.rag.answer_guard import SourceAwareResponseGuard
from backend.services.rag.context_packer import ContextPacker
from backend.services.rag.fusion import HybridRetriever


class RAGPipeline:
    """Coordinates plan, retrieve, rerank, pack, and answer grounding checks."""

    def __init__(
        self,
        retriever: HybridRetriever,
        context_packer: ContextPacker | None = None,
        answer_guard: SourceAwareResponseGuard | None = None,
    ):
        self.retriever = retriever
        self.context_packer = context_packer or ContextPacker()
        self.answer_guard = answer_guard or SourceAwareResponseGuard()

    def build_context(self, query: str, collection_name: str) -> RAGPipelineResult:
        retrieval_result = self._retrieve(query, collection_name)
        context_pack = self.context_packer.pack(query, retrieval_result)
        return RAGPipelineResult(
            query=query,
            collection_name=collection_name,
            retrieval_result=retrieval_result,
            context_pack=context_pack,
        )

    def check_answer(
        self,
        answer: str,
        pipeline_result: RAGPipelineResult,
    ) -> AnswerGroundingReport:
        return self.answer_guard.check(answer, pipeline_result.context_pack)

    def _retrieve(self, query: str, collection_name: str) -> RetrievalResult:
        if collection_name not in {"__all__", "all", "auto"}:
            return self.retriever.retrieve_with_trace(collection_name, query)

        collections = self._available_collections()
        if not collections:
            return self.retriever.retrieve_with_trace("tech_docs_kb", query)

        results = [
            self.retriever.retrieve_with_trace(collection, query)
            for collection in collections
        ]
        docs = []
        for result in results:
            for doc in result.documents:
                annotated = doc.model_copy(deep=True)
                annotated.metadata = {
                    **annotated.metadata,
                    "collection_name": result.collection_name,
                }
                docs.append(annotated)

        docs.sort(key=lambda doc: doc.score or 0.0, reverse=True)
        top_k = self.retriever.config.top_k
        base = results[0].model_copy(deep=True)
        return base.model_copy(
            update={
                "collection_name": "__all__",
                "documents": docs[:top_k],
                "metadata": {
                    **base.metadata,
                    "searched_collections": collections,
                    "per_collection": [
                        {
                            "collection_name": result.collection_name,
                            "hit_count": result.hit_count,
                            "channels": {
                                name: trace.model_dump(exclude_none=True)
                                for name, trace in result.channels.items()
                            },
                        }
                        for result in results
                    ],
                },
            }
        )

    def _available_collections(self) -> list[str]:
        names: set[str] = set()
        for adapter in (self.retriever.vector_db, self.retriever.keyword_db):
            list_collections = getattr(adapter, "list_collections", None)
            if callable(list_collections):
                names.update(list_collections())
        return sorted(name for name in names if name)
