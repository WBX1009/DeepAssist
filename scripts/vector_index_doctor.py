import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

from backend.common.config import settings
from backend.infrastructure.embeddings.bge_m3_local import BGEM3Local
from backend.infrastructure.databases.vector_index_health import VectorIndexHealthInspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose and repair Chroma vector collections.")
    parser.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help="Optional subset of collection names to inspect.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Attempt repair for unhealthy collections and quarantine orphan segment dirs.",
    )
    parser.add_argument(
        "--output",
        default=settings.VECTOR_HEALTH_REPORT_PATH,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding batch size when rebuilding a collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspector = VectorIndexHealthInspector(
        vector_db_path=settings.VECTOR_DB_PATH,
        keyword_db_path=settings.KEYWORD_DB_PATH,
        report_path=args.output,
        embedding_model=BGEM3Local(),
    )
    report = inspector.inspect(
        collections=args.collections,
        repair=args.repair,
        batch_size=args.batch_size,
        persist=True,
    )
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
