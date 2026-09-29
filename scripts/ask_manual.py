from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.context import build_evidence
from app.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from app.generation.qwen_client import QwenChatCompletionsClient
from app.generation.service import AnswerService
from app.retrieval.hybrid import DEFAULT_COLLECTION, DEFAULT_DENSE_MODEL, HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Ask a grounded question about the vehicle manual")
    parser.add_argument("question")
    parser.add_argument("--storage", type=Path, default=PROJECT_ROOT / "qdrant_storage")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--context-limit", type=int, default=3)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with HybridRetriever(
        storage_path=args.storage.resolve(),
        collection_name=args.collection,
        dense_model=args.dense_model,
    ) as retriever:
        results = retriever.search(
            args.question,
            mode="hybrid",
            limit=args.candidate_limit if not args.no_rerank else args.context_limit,
            brand="AITO",
            model="M5",
            power_type="纯电",
        )
        if not args.no_rerank:
            results = CrossEncoderReranker().rerank(
                args.question,
                results,
                limit=args.context_limit,
            )

    evidence = build_evidence(results, max_sources=args.context_limit)
    if args.dry_run:
        print("=== System Prompt ===")
        print(SYSTEM_PROMPT)
        print("\n=== User Prompt ===")
        print(build_user_prompt(args.question, evidence))
        return 0

    try:
        client = QwenChatCompletionsClient.from_env()
    except (ValueError, RuntimeError) as error:
        print(f"配置错误：{error}", file=sys.stderr)
        print(
            "请设置 DASHSCOPE_API_KEY；模型默认使用 qwen-plus。",
            file=sys.stderr,
        )
        return 2

    answer = AnswerService(client, max_sources=args.context_limit).answer(args.question, results)
    print(answer.answer)
    if answer.citations:
        print("\n来源：")
        for index, citation in enumerate(answer.citations, start=1):
            print(
                f"[{index}] {citation.document_title}，{citation.section}，{citation.page_label}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
