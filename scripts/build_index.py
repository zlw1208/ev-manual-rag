from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.hybrid import (
    DEFAULT_COLLECTION,
    DEFAULT_DENSE_MODEL,
    build_local_index,
    build_remote_index,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build the dense+sparse Qdrant index")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "aito-m5-ev.chunks.jsonl",
    )
    parser.add_argument("--storage", type=Path, default=PROJECT_ROOT / "qdrant_storage")
    parser.add_argument("--qdrant-url", help="Remote Qdrant URL, for example http://qdrant:6333")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Keep a non-empty remote collection instead of rebuilding it",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0,
        help="How long to wait for a remote Qdrant service",
    )
    args = parser.parse_args()

    common_options = {
        "chunks_path": args.chunks.resolve(),
        "collection_name": args.collection,
        "dense_model": args.dense_model,
        "model_cache": args.model_cache,
    }
    if args.qdrant_url:
        count = build_remote_index(
            qdrant_url=args.qdrant_url,
            skip_if_exists=args.if_missing,
            wait_seconds=args.wait_seconds,
            **common_options,
        )
        destination = args.qdrant_url
    else:
        count = build_local_index(
            storage_path=args.storage.resolve(),
            **common_options,
        )
        destination = str(args.storage.resolve())
    print(f"Indexed {count} searchable chunks into collection '{args.collection}'")
    print(f"Qdrant destination: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
