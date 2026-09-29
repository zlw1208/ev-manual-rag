from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.hybrid import DEFAULT_COLLECTION, DEFAULT_DENSE_MODEL, HybridRetriever
from app.retrieval.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Search the local vehicle manual index")
    parser.add_argument("query", help="Chinese search query")
    parser.add_argument("--mode", choices=("dense", "sparse", "hybrid"), default="hybrid")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--storage", type=Path, default=PROJECT_ROOT / "qdrant_storage")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--brand")
    parser.add_argument("--model")
    parser.add_argument("--year", type=int)
    parser.add_argument("--power-type")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    args = parser.parse_args()

    with HybridRetriever(
        storage_path=args.storage.resolve(),
        collection_name=args.collection,
        dense_model=args.dense_model,
        model_cache=args.model_cache,
    ) as retriever:
        results = retriever.search(
            args.query,
            mode=args.mode,
            limit=args.candidate_limit if args.rerank else args.limit,
            brand=args.brand,
            model=args.model,
            year=args.year,
            power_type=args.power_type,
        )
        if args.rerank:
            reranker = CrossEncoderReranker(model_name=args.reranker_model)
            results = reranker.rerank(args.query, results, limit=args.limit)

    for rank, result in enumerate(results, start=1):
        section = " > ".join(result.section_path) or "未命名章节"
        page = (
            str(result.page_start)
            if result.page_start == result.page_end
            else f"{result.page_start}-{result.page_end}"
        )
        preview = result.text[:240].replace("\n", " ")
        print(f"[{rank}] score={result.score:.4f} page={page} section={section}")
        print(f"    {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
