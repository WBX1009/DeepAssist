from backend.common.logger import get_logger
from backend.domain.entities.knowledge_base import KnowledgeBaseFile
from backend.domain.interfaces.embedding import BaseEmbedding
from backend.domain.interfaces.keyword_db import BaseKeywordDB
from backend.domain.interfaces.vector_db import BaseVectorDB
from backend.services.rag.chunking import DocumentChunker

logger = get_logger(__name__)


class KnowledgeBaseApp:
    """Knowledge-base ingestion workflow orchestrator."""

    def __init__(
        self,
        chunker: DocumentChunker,
        embedding_model: BaseEmbedding | None,
        vector_db: BaseVectorDB | None,
        keyword_db: BaseKeywordDB | None,
    ):
        self.chunker = chunker
        self.embedding = embedding_model
        self.vector_db = vector_db
        self.keyword_db = keyword_db

    def _rollback_vector_write(self, collection_name: str, file_name: str) -> bool:
        if self.vector_db is None:
            logger.error("Vector rollback skipped because vector DB is unavailable")
            return False

        logger.warning("Starting vector rollback for %s", file_name)
        try:
            rollback_success = self.vector_db.delete_by_source(collection_name, file_name)
        except Exception as exc:
            logger.error("Vector rollback raised for %s: %s", file_name, exc)
            return False

        if rollback_success:
            logger.info("Vector rollback completed for %s", file_name)
        else:
            logger.error("Vector rollback failed for %s", file_name)
        return rollback_success

    def _is_ready_for_ingestion(self) -> bool:
        return bool(self.embedding and self.vector_db and self.keyword_db)

    def list_files(self, collection_name: str = "tech_docs_kb") -> dict:
        """Return file-level KB inventory merged from vector and keyword stores."""
        records: dict[str, dict] = {}
        errors: dict[str, str] = {}

        if self.vector_db is not None:
            try:
                for item in self.vector_db.list_sources(collection_name):
                    records.setdefault(item.source_file, self._new_file_record(item))
                    records[item.source_file]["vector_chunk_count"] = item.chunk_count
                    records[item.source_file]["stores"].append("vector")
            except Exception as exc:
                logger.error("Failed to list vector KB files: %s", exc)
                errors["vector"] = str(exc)
        else:
            errors["vector"] = "Vector database is unavailable"

        if self.keyword_db is not None:
            try:
                for item in self.keyword_db.list_sources(collection_name):
                    records.setdefault(item.source_file, self._new_file_record(item))
                    records[item.source_file]["keyword_chunk_count"] = item.chunk_count
                    records[item.source_file]["stores"].append("keyword")
            except Exception as exc:
                logger.error("Failed to list keyword KB files: %s", exc)
                errors["keyword"] = str(exc)
        else:
            errors["keyword"] = "Keyword database is unavailable"

        files = []
        for record in records.values():
            vector_count = record["vector_chunk_count"]
            keyword_count = record["keyword_chunk_count"]
            record["chunk_count"] = max(vector_count, keyword_count)
            record["consistent"] = vector_count == keyword_count
            record["stores"] = sorted(set(record["stores"]))
            files.append(record)

        files.sort(key=lambda item: item["source_file"].lower())
        status = "success" if not errors else ("partial_success" if files else "error")
        return {
            "status": status,
            "collection_name": collection_name,
            "data": files,
            "errors": errors,
        }

    def list_collections(self) -> dict:
        """Return collection-level inventory merged from vector and keyword stores."""
        records: dict[str, dict] = {}
        errors: dict[str, str] = {}

        if self.vector_db is not None:
            try:
                for name in self.vector_db.list_collections():
                    records.setdefault(name, {"collection_name": name, "stores": []})
                    records[name]["stores"].append("vector")
            except Exception as exc:
                logger.error("Failed to list vector collections: %s", exc)
                errors["vector"] = str(exc)
        else:
            errors["vector"] = "Vector database is unavailable"

        if self.keyword_db is not None:
            try:
                for name in self.keyword_db.list_collections():
                    records.setdefault(name, {"collection_name": name, "stores": []})
                    records[name]["stores"].append("keyword")
            except Exception as exc:
                logger.error("Failed to list keyword collections: %s", exc)
                errors["keyword"] = str(exc)
        else:
            errors["keyword"] = "Keyword database is unavailable"

        collections = []
        for record in records.values():
            name = record["collection_name"]
            file_payload = self.list_files(name)
            files = file_payload.get("data", [])
            record["stores"] = sorted(set(record["stores"]))
            record["file_count"] = len(files) if isinstance(files, list) else 0
            record["chunk_count"] = sum(
                int(item.get("chunk_count", 0))
                for item in files
                if isinstance(item, dict)
            )
            record["consistent"] = all(
                bool(item.get("consistent"))
                for item in files
                if isinstance(item, dict)
            ) if files else "vector" in record["stores"] and "keyword" in record["stores"]
            collections.append(record)

        collections.sort(key=lambda item: item["collection_name"].lower())
        status = "success" if not errors else ("partial_success" if collections else "error")
        return {
            "status": status,
            "data": collections,
            "errors": errors,
        }

    def delete_file(
        self,
        file_name: str,
        collection_name: str = "tech_docs_kb",
    ) -> dict:
        """Delete a source file from both Chroma and Whoosh indexes."""
        source_file = (file_name or "").strip()
        if not source_file:
            return {"status": "error", "message": "source_file is required"}

        logger.warning("Deleting KB source [%s] from collection [%s]", source_file, collection_name)

        vector_success = False
        keyword_success = False
        errors: dict[str, str] = {}

        if self.vector_db is not None:
            try:
                vector_success = self.vector_db.delete_by_source(collection_name, source_file)
                if not vector_success:
                    errors["vector"] = "Vector database delete returned false"
            except Exception as exc:
                logger.error("Vector delete failed for %s: %s", source_file, exc)
                errors["vector"] = str(exc)
        else:
            errors["vector"] = "Vector database is unavailable"

        if self.keyword_db is not None:
            try:
                keyword_success = self.keyword_db.delete_by_source(collection_name, source_file)
                if not keyword_success:
                    errors["keyword"] = "Keyword database delete returned false"
            except Exception as exc:
                logger.error("Keyword delete failed for %s: %s", source_file, exc)
                errors["keyword"] = str(exc)
        else:
            errors["keyword"] = "Keyword database is unavailable"

        if vector_success and keyword_success:
            logger.info("KB source [%s] deleted from both stores", source_file)
            return {
                "status": "success",
                "message": f"Deleted {source_file}",
                "data": {
                    "source_file": source_file,
                    "vector_deleted": True,
                    "keyword_deleted": True,
                },
            }

        logger.error("KB source delete incomplete for %s: %s", source_file, errors)
        return {
            "status": "error",
            "message": "Knowledge-base delete was incomplete",
            "data": {
                "source_file": source_file,
                "vector_deleted": vector_success,
                "keyword_deleted": keyword_success,
            },
            "errors": errors,
        }

    def _new_file_record(self, item: KnowledgeBaseFile) -> dict:
        return {
            "source_file": item.source_file,
            "chunk_count": item.chunk_count,
            "vector_chunk_count": 0,
            "keyword_chunk_count": 0,
            "consistent": False,
            "stores": [],
            "metadata": item.metadata,
        }

    def process_document(
        self,
        file_name: str,
        content: str,
        collection_name: str = "tech_docs_kb",
    ) -> dict:
        logger.info("Received knowledge-base ingestion request: %s", file_name)

        if not self._is_ready_for_ingestion():
            logger.error("Knowledge-base ingestion rejected because dependencies are unavailable")
            return {
                "status": "error",
                "message": "Knowledge-base ingestion dependencies are unavailable",
            }

        chunks = self.chunker.split_markdown(content, source_name=file_name)
        if not chunks:
            return {"status": "error", "message": "Document is empty or chunking failed"}

        try:
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding.embed_documents(texts)
        except Exception as exc:
            logger.error("Embedding failed for %s: %s", file_name, exc)
            return {"status": "error", "message": f"Embedding failed: {exc}"}

        v_success = False
        try:
            v_success = self.vector_db.add_chunks(collection_name, chunks, embeddings)
            if not v_success:
                logger.error("Vector write failed for %s", file_name)
                return {"status": "error", "message": "Vector database write failed"}

            k_success = self.keyword_db.build_index(collection_name, chunks)
            if k_success:
                logger.info("Document %s ingested successfully with %s chunks", file_name, len(chunks))
                return {"status": "success", "message": f"Successfully processed {len(chunks)} chunks"}

            logger.error("Keyword index write failed for %s; starting vector rollback", file_name)
            rollback_success = self._rollback_vector_write(collection_name, file_name)

            return {
                "status": "error",
                "message": "Keyword index write failed; vector rollback completed"
                if rollback_success
                else "Keyword index write failed; vector rollback failed",
            }

        except Exception as exc:
            logger.error("Knowledge-base ingestion failed for %s: %s", file_name, exc)
            if v_success:
                logger.warning("Exception after vector write for %s; starting vector rollback", file_name)
                self._rollback_vector_write(collection_name, file_name)

            return {"status": "error", "message": f"Database write failed: {exc}"}
