from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.hybrid import DEFAULT_COLLECTION, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


def describe_vector(vector: Any, show_values: bool) -> str:
    if isinstance(vector, list):
        preview = vector if show_values else vector[:8]
        suffix = "" if show_values else ", ..."
        return f"dense(dim={len(vector)}, values={preview}{suffix})"
    if hasattr(vector, "indices") and hasattr(vector, "values"):
        indices = list(vector.indices)
        values = list(vector.values)
        size = len(indices)
        if show_values:
            return f"sparse(non_zero={size}, indices={indices}, values={values})"
        return (
            f"sparse(non_zero={size}, indices={indices[:8]}, "
            f"values={values[:8]}, ...)"
        )
    return repr(vector)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Inspect a Qdrant collection")
    parser.add_argument("--storage", type=Path, default=PROJECT_ROOT / "qdrant_storage")
    parser.add_argument("--qdrant-url", help="Remote Qdrant URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--show-vector", action="store_true")
    args = parser.parse_args()

    client = (
        QdrantClient(url=args.qdrant_url)
        if args.qdrant_url
        else QdrantClient(path=str(args.storage.resolve()))
    )
    try:
        info = client.get_collection(args.collection)
        count = client.count(args.collection, exact=True).count
        dense_config = info.config.params.vectors[DENSE_VECTOR_NAME]
        print(f"Collection : {args.collection}")
        print(f"Status     : {info.status.value}")
        print(f"Points     : {count}")
        print(f"Dense      : {dense_config.size} dimensions / {dense_config.distance.value}")
        print(f"Sparse     : {SPARSE_VECTOR_NAME} / IDF")

        points, _ = client.scroll(
            collection_name=args.collection,
            limit=args.limit,
            with_payload=True,
            with_vectors=True,
        )
        for rank, point in enumerate(points, start=1):
            payload = point.payload or {}
            print(f"\n[{rank}] id={point.id}")
            print(f"chunk_id   : {payload.get('chunk_id')}")
            print(f"vehicle    : {payload.get('brand')} {payload.get('model')}")
            print(f"pages      : {payload.get('page_start')}-{payload.get('page_end')}")
            print(f"section    : {' > '.join(payload.get('section_path', []))}")
            print(f"text       : {str(payload.get('text', ''))[:180]}")
            vectors = point.vector if isinstance(point.vector, dict) else {}
            for name in (DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME):
                if name in vectors:
                    print(f"{name:<10} : {describe_vector(vectors[name], args.show_vector)}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
