from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    question: str
    category: str
    expected_chunk_ids: frozenset[str]
    expected_pages: tuple[int, ...]
    expected_section: str


@dataclass(frozen=True)
class QueryEvaluation:
    id: str
    question: str
    category: str
    rank: int | None
    latency_ms: float
    returned_chunk_ids: list[str]


@dataclass(frozen=True)
class ModeMetrics:
    mode: str
    question_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr_at_5: float
    average_latency_ms: float
    p95_latency_ms: float
    category_recall_at_5: dict[str, float]


def load_questions(path: Path) -> list[EvaluationQuestion]:
    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                question = EvaluationQuestion(
                    id=str(raw["id"]),
                    question=str(raw["question"]),
                    category=str(raw["category"]),
                    expected_chunk_ids=frozenset(map(str, raw["expected_chunk_ids"])),
                    expected_pages=tuple(map(int, raw.get("expected_pages", []))),
                    expected_section=str(raw.get("expected_section", "")),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid evaluation JSON at line {line_number}: {error}") from error
            if question.id in seen_ids:
                raise ValueError(f"Duplicate evaluation question id: {question.id}")
            if not question.expected_chunk_ids:
                raise ValueError(f"Question {question.id} has no expected chunk ids")
            seen_ids.add(question.id)
            questions.append(question)
    if not questions:
        raise ValueError(f"No evaluation questions found in {path}")
    return questions


def first_relevant_rank(returned_ids: Sequence[str], expected_ids: frozenset[str]) -> int | None:
    return next(
        (rank for rank, chunk_id in enumerate(returned_ids, start=1) if chunk_id in expected_ids),
        None,
    )


def percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def calculate_metrics(mode: str, rows: Sequence[QueryEvaluation]) -> ModeMetrics:
    if not rows:
        raise ValueError("Cannot calculate metrics without query results")
    category_rows: dict[str, list[QueryEvaluation]] = defaultdict(list)
    for row in rows:
        category_rows[row.category].append(row)

    def recall_at(items: Iterable[QueryEvaluation], cutoff: int) -> float:
        materialized = list(items)
        return sum(row.rank is not None and row.rank <= cutoff for row in materialized) / len(
            materialized
        )

    latencies = [row.latency_ms for row in rows]
    return ModeMetrics(
        mode=mode,
        question_count=len(rows),
        recall_at_1=recall_at(rows, 1),
        recall_at_3=recall_at(rows, 3),
        recall_at_5=recall_at(rows, 5),
        mrr_at_5=sum(1 / row.rank for row in rows if row.rank is not None and row.rank <= 5)
        / len(rows),
        average_latency_ms=statistics.fmean(latencies),
        p95_latency_ms=percentile_nearest_rank(latencies, 0.95),
        category_recall_at_5={
            category: recall_at(items, 5) for category, items in sorted(category_rows.items())
        },
    )


def evaluate_mode(
    retriever: HybridRetriever,
    questions: Sequence[EvaluationQuestion],
    mode: str,
    limit: int = 5,
    brand: str | None = None,
    model: str | None = None,
    power_type: str | None = None,
    reranker: Reranker | None = None,
    candidate_limit: int = 10,
) -> tuple[ModeMetrics, list[QueryEvaluation]]:
    rows: list[QueryEvaluation] = []
    for item in questions:
        started = time.perf_counter()
        retrieval_mode = "hybrid" if mode == "hybrid_rerank" else mode
        retrieval_limit = candidate_limit if mode == "hybrid_rerank" else limit
        results = retriever.search(
            item.question,
            mode=retrieval_mode,  # type: ignore[arg-type]
            limit=retrieval_limit,
            brand=brand,
            model=model,
            power_type=power_type,
        )
        if mode == "hybrid_rerank":
            if reranker is None:
                raise ValueError("hybrid_rerank mode requires a reranker")
            results = reranker.rerank(item.question, results, limit=limit)
        latency_ms = (time.perf_counter() - started) * 1000
        returned_ids = [result.chunk_id for result in results]
        rows.append(
            QueryEvaluation(
                id=item.id,
                question=item.question,
                category=item.category,
                rank=first_relevant_rank(returned_ids, item.expected_chunk_ids),
                latency_ms=latency_ms,
                returned_chunk_ids=returned_ids,
            )
        )
    return calculate_metrics(mode, rows), rows


def report_as_dict(
    metrics: Sequence[ModeMetrics],
    details: dict[str, list[QueryEvaluation]],
) -> dict[str, Any]:
    return {
        "metrics": [asdict(item) for item in metrics],
        "details": {mode: [asdict(row) for row in rows] for mode, rows in details.items()},
    }
