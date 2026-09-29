from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.api_client import ApiManualQAClient
from app.evaluation.generation import (
    calculate_generation_metrics,
    error_evaluation,
    evaluate_answer,
    evaluation_report,
    load_generation_questions,
    rescore_saved_evaluation,
)
from app.manual_qa import ManualQAService


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate end-to-end grounded manual QA")
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "generation_questions.jsonl",
    )
    parser.add_argument("--storage", type=Path, default=PROJECT_ROOT / "qdrant_storage")
    parser.add_argument(
        "--api-url",
        help="Evaluate a deployed FastAPI service, for example http://127.0.0.1:8001",
    )
    parser.add_argument("--qdrant-url", help="Use a remote Qdrant service directly")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N questions")
    parser.add_argument(
        "--reuse",
        type=Path,
        help="Rescore responses from an existing report without calling the model again",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "generation_results.json",
    )
    args = parser.parse_args()

    questions = load_generation_questions(args.questions.resolve())
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        questions = questions[: args.limit]

    rows = []
    if args.reuse:
        saved_report = json.loads(args.reuse.resolve().read_text(encoding="utf-8"))
        saved_by_id = {str(item["id"]): item for item in saved_report["details"]}
        missing_ids = [item.id for item in questions if item.id not in saved_by_id]
        if missing_ids:
            parser.error(f"saved report is missing question ids: {', '.join(missing_ids)}")
        rows = [rescore_saved_evaluation(item, saved_by_id[item.id]) for item in questions]
        print(f"已使用最新标注重新评分（{len(rows)} 题），未调用模型。")
    else:
        print(f"正在评测端到端问答（{len(questions)} 题）...")
        if args.api_url:
            service = ApiManualQAClient(args.api_url)
        elif args.qdrant_url:
            service = ManualQAService(qdrant_url=args.qdrant_url)
        else:
            service = ManualQAService(storage_path=args.storage.resolve())
        try:
            for index, question in enumerate(questions, start=1):
                print(f"[{index}/{len(questions)}] {question.id}: {question.question}")
                try:
                    response = service.ask(question.question)
                    row = evaluate_answer(question, response)
                except Exception as error:  # noqa: BLE001 - preserve failures in the report
                    row = error_evaluation(question, error)
                rows.append(row)
                status = "PASS" if row.passed else "FAIL"
                print(
                    f"  {status} | latency={row.latency_ms:.0f}ms | "
                    f"tokens={row.total_tokens} | refused={row.refused}"
                )
        finally:
            service.close()

    metrics = calculate_generation_metrics(rows)
    print("\n端到端问答评测结果")
    print(f"通过率：           {metrics.pass_rate:.1%}")
    print(f"答案要点召回率：   {metrics.answer_point_recall:.1%}")
    print(f"完整答案率：       {metrics.fully_correct_answer_rate:.1%}")
    print(f"引用命中率：       {metrics.citation_hit_rate:.1%}")
    print(f"引用精确率：       {metrics.citation_precision:.1%}")
    print(f"拒答准确率：       {metrics.refusal_accuracy:.1%}")
    print(f"平均/P95 延迟：    {metrics.average_latency_ms:.0f}/{metrics.p95_latency_ms:.0f} ms")
    print(f"总 Token：         {metrics.total_tokens}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = evaluation_report(metrics, rows)
    report["metadata"] = {
        "dataset_path": str(args.questions.resolve()),
        "dataset_sha256": hashlib.sha256(args.questions.resolve().read_bytes()).hexdigest(),
        "reused_responses": bool(args.reuse),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"详细结果已写入：{args.output.resolve()}")
    return 0 if all(row.error is None for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
