from pathlib import Path

import pytest

from app.evaluation.retrieval import (
    QueryEvaluation,
    calculate_metrics,
    first_relevant_rank,
    load_questions,
    percentile_nearest_rank,
)


def test_first_relevant_rank() -> None:
    assert first_relevant_rank(["wrong", "gold", "other"], frozenset({"gold"})) == 2
    assert first_relevant_rank(["wrong"], frozenset({"gold"})) is None


def test_percentile_uses_nearest_rank() -> None:
    assert percentile_nearest_rank([1, 2, 3, 4], 0.95) == 4
    assert percentile_nearest_rank([], 0.95) == 0


def test_calculate_metrics() -> None:
    rows = [
        QueryEvaluation("q1", "one", "a", 1, 10.0, ["x"]),
        QueryEvaluation("q2", "two", "a", 3, 20.0, ["y"]),
        QueryEvaluation("q3", "three", "b", None, 30.0, ["z"]),
    ]
    metrics = calculate_metrics("hybrid", rows)
    assert metrics.recall_at_1 == pytest.approx(1 / 3)
    assert metrics.recall_at_3 == pytest.approx(2 / 3)
    assert metrics.recall_at_5 == pytest.approx(2 / 3)
    assert metrics.mrr_at_5 == pytest.approx((1 + 1 / 3) / 3)
    assert metrics.average_latency_ms == 20
    assert metrics.category_recall_at_5 == {"a": 1.0, "b": 0.0}


def test_load_questions_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    row = (
        '{"id":"same","question":"q","category":"c",'
        '"expected_chunk_ids":["chunk"]}\n'
    )
    path.write_text(row + row, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_questions(path)
