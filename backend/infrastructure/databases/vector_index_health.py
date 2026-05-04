import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import chromadb
from whoosh.index import exists_in, open_dir

from backend.common.logger import get_logger
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import (
    KnowledgeBaseCollectionHealth,
    KnowledgeBaseHealthReport,
)
from backend.domain.interfaces.embedding import BaseEmbedding
from backend.infrastructure.databases.chroma_store import ChromaStore

logger = get_logger(__name__)


class VectorIndexHealthInspector:
    """Inspect and optionally repair persisted Chroma collections."""

    def __init__(
        self,
        vector_db_path: str,
        keyword_db_path: str,
        report_path: str,
        embedding_model: BaseEmbedding | None = None,
    ):
        self.vector_db_path = Path(vector_db_path)
        self.keyword_db_path = Path(keyword_db_path)
        self.report_path = Path(report_path)
        self.embedding_model = embedding_model

    def load_report(self) -> KnowledgeBaseHealthReport | None:
        if not self.report_path.exists():
            return None

        try:
            payload = json.loads(self.report_path.read_text(encoding="utf-8"))
            return KnowledgeBaseHealthReport.model_validate(payload)
        except Exception as exc:
            logger.warning("Failed to load KB health report: %s", exc)
            return None

    def write_report(self, report: KnowledgeBaseHealthReport) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def inspect(
        self,
        collections: list[str] | None = None,
        repair: bool = False,
        batch_size: int = 256,
        persist: bool = True,
    ) -> KnowledgeBaseHealthReport:
        client = chromadb.PersistentClient(path=str(self.vector_db_path))
        manifest = self._load_collection_manifest()
        actual_segment_dirs = set(self._load_actual_segment_dirs())
        selected_collections = sorted(collections or manifest.keys())

        report = KnowledgeBaseHealthReport(
            vector_db_path=str(self.vector_db_path),
            keyword_db_path=str(self.keyword_db_path),
            checked_at=datetime.now().isoformat(timespec="seconds"),
            orphan_segment_dirs=sorted(actual_segment_dirs - set(manifest.values())),
        )

        for collection_name in selected_collections:
            health = self._check_collection_health(
                client=client,
                collection_name=collection_name,
                expected_segment_id=manifest.get(collection_name),
                actual_segment_dirs=actual_segment_dirs,
            )
            report.collections.append(health)

        if repair:
            report.quarantined_dirs = self._quarantine_orphans(report.orphan_segment_dirs)
            for health in report.collections:
                if health.healthy:
                    continue
                self._rebuild_collection(
                    client=client,
                    collection_name=health.collection_name,
                    batch_size=batch_size,
                )
                refreshed_manifest = self._load_collection_manifest()
                refreshed_dirs = set(self._load_actual_segment_dirs())
                refreshed = self._check_collection_health(
                    client=client,
                    collection_name=health.collection_name,
                    expected_segment_id=refreshed_manifest.get(health.collection_name),
                    actual_segment_dirs=refreshed_dirs,
                )
                health.expected_segment_id = refreshed.expected_segment_id
                health.actual_segment_dir_present = refreshed.actual_segment_dir_present
                health.count = refreshed.count
                health.get_ok = refreshed.get_ok
                health.query_ok = refreshed.query_ok
                health.whoosh_docs = refreshed.whoosh_docs
                health.healthy = refreshed.healthy
                health.errors = refreshed.errors
                health.repaired = True

        if persist:
            self.write_report(report)

        return report

    def _load_collection_manifest(self) -> dict[str, str]:
        sqlite_path = self.vector_db_path / "chroma.sqlite3"
        with sqlite3.connect(sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT c.name, s.id
                FROM collections c
                JOIN segments s ON s.collection = c.id
                WHERE s.scope = 'VECTOR'
                ORDER BY c.name ASC
                """
            )
            return {name: segment_id for name, segment_id in cursor.fetchall()}

    def _load_actual_segment_dirs(self) -> list[str]:
        return sorted(
            entry.name
            for entry in self.vector_db_path.iterdir()
            if entry.is_dir() and entry.name != "_quarantine"
        )

    def _load_whoosh_documents(self, collection_name: str) -> list[DocumentChunk]:
        index_dir = self.keyword_db_path / collection_name
        if not exists_in(str(index_dir)):
            raise RuntimeError(f"Whoosh index for [{collection_name}] does not exist")

        ix = open_dir(str(index_dir))
        chunks: list[DocumentChunk] = []
        with ix.searcher() as searcher:
            for fields in searcher.all_stored_fields():
                metadata = json.loads(fields.get("metadata") or "{}")
                source_file = fields.get("source_file") or metadata.get("source_file")
                if source_file and "source_file" not in metadata:
                    metadata["source_file"] = source_file
                chunks.append(
                    DocumentChunk(
                        id=fields["id"],
                        content=fields["content"],
                        metadata=metadata,
                    )
                )
        return chunks

    def _check_collection_health(
        self,
        client: chromadb.PersistentClient,
        collection_name: str,
        expected_segment_id: str | None,
        actual_segment_dirs: set[str],
    ) -> KnowledgeBaseCollectionHealth:
        health = KnowledgeBaseCollectionHealth(
            collection_name=collection_name,
            expected_segment_id=expected_segment_id,
            actual_segment_dir_present=bool(
                expected_segment_id and expected_segment_id in actual_segment_dirs
            ),
        )

        try:
            collection = client.get_collection(collection_name)
        except Exception as exc:
            health.errors.append(f"get_collection failed: {exc}")
            return self._finalize_health(health)

        try:
            health.count = int(collection.count())
        except Exception as exc:
            health.errors.append(f"count failed: {exc}")

        try:
            result = collection.get(limit=1, include=["metadatas"])
            health.get_ok = bool(result.get("ids") is not None)
        except Exception as exc:
            health.errors.append(f"get failed: {exc}")

        if self.embedding_model is None:
            health.errors.append("query check skipped: embedding model unavailable")
        else:
            try:
                query_vector = self.embedding_model.embed_text("vector index doctor health check")
                result = collection.query(
                    query_embeddings=[query_vector],
                    n_results=1,
                    include=["metadatas"],
                )
                health.query_ok = bool((result.get("ids") or [[]])[0] is not None)
            except Exception as exc:
                health.errors.append(f"query failed: {exc}")

        try:
            health.whoosh_docs = len(self._load_whoosh_documents(collection_name))
        except Exception as exc:
            health.errors.append(f"whoosh export failed: {exc}")

        return self._finalize_health(health)

    def _finalize_health(
        self,
        health: KnowledgeBaseCollectionHealth,
    ) -> KnowledgeBaseCollectionHealth:
        health.healthy = (
            health.actual_segment_dir_present
            and health.get_ok
            and health.query_ok
            and health.count is not None
            and (health.whoosh_docs is None or health.count == health.whoosh_docs)
        )
        return health

    def _rebuild_collection(
        self,
        client: chromadb.PersistentClient,
        collection_name: str,
        batch_size: int,
    ) -> None:
        if self.embedding_model is None:
            raise RuntimeError("Cannot rebuild vector collection without an embedding model")

        chunks = self._load_whoosh_documents(collection_name)
        if not chunks:
            raise RuntimeError(f"No documents available in Whoosh for [{collection_name}]")

        logger.warning(
            "Rebuilding vector collection [%s] from Whoosh source of truth",
            collection_name,
        )
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        vector_store = ChromaStore()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedding_model.embed_documents([chunk.content for chunk in batch])
            if not vector_store.add_chunks(collection_name, batch, embeddings):
                raise RuntimeError(
                    f"Failed to add batch starting at {start} for [{collection_name}]"
                )

    def _quarantine_orphans(self, orphan_dirs: list[str]) -> list[str]:
        if not orphan_dirs:
            return []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_root = self.vector_db_path / "_quarantine" / timestamp
        quarantine_root.mkdir(parents=True, exist_ok=True)

        moved: list[str] = []
        for orphan in orphan_dirs:
            source = self.vector_db_path / orphan
            if not source.exists():
                continue
            target = quarantine_root / orphan
            shutil.move(str(source), str(target))
            moved.append(str(target))
        return moved
