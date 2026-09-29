from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.retrieval import evaluate_mode, load_questions, report_as_dict
from app.retrieval.hybrid import DEFAULT_COLLECTION, DEFAULT_DENSE_MODEL, HybridRetriever
from app.retrieval.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate dense, sparse and hybrid retrieval")
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "retrieval_questions.jsonl",
    )
    parser.add_argument("--storage", type=Path, default=PROJECT_ROOT / "qdrant_storage")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("dense", "sparse", "hybrid", "hybrid_rerank"),
        default=("dense", "sparse", "hybrid"),
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--brand", default="AITO")
    parser.add_argument("--model", default="M5")
    parser.add_argument("--power-type", default="纯电")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    questions = load_questions(args.questions.resolve())
    metrics = []
    details = {}
    reranker = (
        CrossEncoderReranker(model_name=args.reranker_model)
        if "hybrid_rerank" in args.modes
        else None
    )
    with HybridRetriever(
        storage_path=args.storage.resolve(),
        collection_name=args.collection,
        dense_model=args.dense_model,
    ) as retriever:
        for mode in args.modes:
            print(f"正在评测 {mode}（{len(questions)} 个问题）...")
            mode_metrics, mode_rows = evaluate_mode(
                retriever,
                questions,
                mode,
                limit=args.limit,
                brand=args.brand,
                model=args.model,
                power_type=args.power_type,
                reranker=reranker,
                candidate_limit=args.candidate_limit,
            )
            metrics.append(mode_metrics)
            details[mode] = mode_rows

    print("\n检索基线结果")
    print("mode     R@1     R@3     R@5     MRR@5   avg_ms   p95_ms")
    for item in metrics:
        print(
            f"{item.mode:<8} "
            f"{item.recall_at_1:>6.1%}  {item.recall_at_3:>6.1%}  "
            f"{item.recall_at_5:>6.1%}  {item.mrr_at_5:>6.3f}  "
            f"{item.average_latency_ms:>7.1f}  {item.p95_latency_ms:>7.1f}"
        )

    for mode, rows in details.items():
        misses = [row for row in rows if row.rank is None]
        if misses:
            print(f"\n{mode} Top-{args.limit} 未命中（{len(misses)}）:")
            for row in misses:
                print(f"- {row.id}: {row.question}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report_as_dict(metrics, details), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n详细结果已写入：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
