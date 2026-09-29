from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from app.manual_qa import ManualQAResponse


@dataclass(frozen=True)
class GenerationQuestion:
    id: str
    question: str
    category: str
    answerable: bool
    answer_point_groups: tuple[tuple[str, ...], ...]
    expected_chunk_ids: frozenset[str]
    expected_pages: frozenset[int]


@dataclass(frozen=True)
class GenerationEvaluation:
    id: str
    question: str
    category: str
    answerable: bool
    refused: bool
    answer: str
    answer_point_recall: float | None
    matched_answer_points: int
    total_answer_points: int
    citation_hit: bool | None
    citation_precision: float | None
    cited_chunk_ids: list[str]
    cited_pages: list[int]
    passed: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str | None = None


@dataclass(frozen=True)
class GenerationMetrics:
    question_count: int
    answerable_count: int
    unanswerable_count: int
    pass_rate: float
    answer_point_recall: float
    fully_correct_answer_rate: float
    citation_hit_rate: float
    citation_precision: float
    refusal_accuracy: float
    average_latency_ms: float
    p95_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int


class QuestionAnswerer(Protocol):
    def ask(self, question: str) -> ManualQAResponse: ...


def load_generation_questions(path: Path) -> list[GenerationQuestion]:
    questions: list[GenerationQuestion] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                answerable = bool(raw["answerable"])
                groups = tuple(
                    tuple(str(term) for term in group)
                    for group in raw.get("answer_point_groups", [])
                )
                question = GenerationQuestion(
                    id=str(raw["id"]),
                    question=str(raw["question"]),
                    category=str(raw["category"]),
                    answerable=answerable,
                    answer_point_groups=groups,
                    expected_chunk_ids=frozenset(map(str, raw.get("expected_chunk_ids", []))),
                    expected_pages=frozenset(map(int, raw.get("expected_pages", []))),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid generation evaluation JSON at line {line_number}") from error
            if question.id in seen_ids:
                raise ValueError(f"Duplicate evaluation question id: {question.id}")
            if answerable and not question.answer_point_groups:
                raise ValueError(f"Answerable question {question.id} has no answer points")
            if answerable and not (question.expected_chunk_ids or question.expected_pages):
                raise ValueError(f"Answerable question {question.id} has no expected citation")
            seen_ids.add(question.id)
            questions.append(question)
    if not questions:
        raise ValueError(f"No generation evaluation questions found in {path}")
    return questions


def normalize_text(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())


_EQUIVALENT_PHRASES = (
    (re.compile(r"请勿|切勿|禁止|不可以"), "不能"),
    (re.compile(r"制动踏板"), "刹车"),
)
_NEGATION_PATTERN = re.compile(r"不能|不会|不可|不应|不得|禁止|无|未|勿")
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def canonicalize_answer_point_text(text: str) -> str:
    normalized = normalize_text(text)
    for pattern, replacement in _EQUIVALENT_PHRASES:
        normalized = pattern.sub(replacement, normalized)
    return normalized.replace("其", "")


def _has_same_facts(expected: str, candidate: str) -> bool:
    """Protect fuzzy matching from changing polarity or numeric facts."""
    if bool(_NEGATION_PATTERN.search(expected)) != bool(_NEGATION_PATTERN.search(candidate)):
        return False
    return _NUMBER_PATTERN.findall(expected) == _NUMBER_PATTERN.findall(candidate)


def _exact_match(answer: str, expected: str) -> bool:
    start = answer.find(expected)
    while start >= 0:
        # Do not accept a positive phrase merely because it is nested in a
        # negated one, for example `可以开启` inside `不可以开启`.
        prefix = answer[max(0, start - 1) : start]
        candidate = prefix + expected if prefix in {"不", "未", "无"} else expected
        if _has_same_facts(expected, candidate):
            return True
        start = answer.find(expected, start + 1)
    return False


def _subsequence_match(answer: str, expected: str) -> bool:
    """Allow a few inserted modifiers while preserving character order."""
    if not expected or len(expected) < 3:
        return False
    max_extra = max(2, len(expected) // 2)
    for start, character in enumerate(answer):
        if character != expected[0]:
            continue
        expected_index = 1
        end = start + 1
        while end < len(answer) and expected_index < len(expected):
            if answer[end] == expected[expected_index]:
                expected_index += 1
            end += 1
        if expected_index == len(expected) and end - start <= len(expected) + max_extra:
            candidate = answer[start:end]
            if _has_same_facts(expected, candidate):
                return True
    return False


def _fuzzy_window_match(answer: str, expected: str) -> bool:
    """Handle small local word-order changes without comparing whole answers."""
    if len(expected) < 6:
        return False
    min_window = max(3, len(expected) - 2)
    max_window = min(len(answer), len(expected) + 3)
    for window_size in range(min_window, max_window + 1):
        for start in range(len(answer) - window_size + 1):
            candidate = answer[start : start + window_size]
            if not _has_same_facts(expected, candidate):
                continue
            if SequenceMatcher(None, expected, candidate).ratio() >= 0.8:
                return True
    return False


def answer_point_matches(answer: str, alternatives: Sequence[str]) -> bool:
    normalized_answer = canonicalize_answer_point_text(answer)
    for term in alternatives:
        normalized_term = canonicalize_answer_point_text(term)
        if (
            _exact_match(normalized_answer, normalized_term)
            or _subsequence_match(normalized_answer, normalized_term)
            or _fuzzy_window_match(normalized_answer, normalized_term)
        ):
            return True
    return False


def citation_is_expected(
    chunk_id: str,
    page_start: int,
    page_end: int,
    question: GenerationQuestion,
) -> bool:
    if chunk_id in question.expected_chunk_ids:
        return True
    cited_pages = set(range(page_start, page_end + 1))
    return bool(cited_pages & question.expected_pages)


def evaluate_answer(
    question: GenerationQuestion,
    response: ManualQAResponse,
) -> GenerationEvaluation:
    answer = response.answer
    citation_matches = [
        citation_is_expected(
            citation.chunk_id,
            citation.page_start,
            citation.page_end,
            question,
        )
        for citation in answer.citations
    ]

    if question.answerable:
        matched_points = sum(
            answer_point_matches(answer.answer, group) for group in question.answer_point_groups
        )
        point_recall = matched_points / len(question.answer_point_groups)
        citation_hit = any(citation_matches)
        citation_precision = (
            sum(citation_matches) / len(citation_matches) if citation_matches else 0.0
        )
        passed = not answer.refused and point_recall == 1.0 and citation_hit
    else:
        matched_points = 0
        point_recall = None
        citation_hit = None
        citation_precision = None
        passed = answer.refused

    return GenerationEvaluation(
        id=question.id,
        question=question.question,
        category=question.category,
        answerable=question.answerable,
        refused=answer.refused,
        answer=answer.answer,
        answer_point_recall=point_recall,
        matched_answer_points=matched_points,
        total_answer_points=len(question.answer_point_groups),
        citation_hit=citation_hit,
        citation_precision=citation_precision,
        cited_chunk_ids=[item.chunk_id for item in answer.citations],
        cited_pages=[item.page_start for item in answer.citations],
        passed=passed,
        latency_ms=response.elapsed_seconds * 1000,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
    )


def error_evaluation(question: GenerationQuestion, error: Exception) -> GenerationEvaluation:
    return GenerationEvaluation(
        id=question.id,
        question=question.question,
        category=question.category,
        answerable=question.answerable,
        refused=False,
        answer="",
        answer_point_recall=0.0 if question.answerable else None,
        matched_answer_points=0,
        total_answer_points=len(question.answer_point_groups),
        citation_hit=False if question.answerable else None,
        citation_precision=0.0 if question.answerable else None,
        cited_chunk_ids=[],
        cited_pages=[],
        passed=False,
        latency_ms=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        error=str(error),
    )


def rescore_saved_evaluation(
    question: GenerationQuestion,
    saved: dict[str, Any],
) -> GenerationEvaluation:
    """Reapply updated labels to a saved model response without another API call."""
    answer = str(saved.get("answer", ""))
    refused = bool(saved.get("refused", False))
    chunk_ids = list(map(str, saved.get("cited_chunk_ids", [])))
    pages = list(map(int, saved.get("cited_pages", [])))
    citation_matches = [
        chunk_id in question.expected_chunk_ids or page in question.expected_pages
        for chunk_id, page in zip(chunk_ids, pages, strict=False)
    ]

    if question.answerable:
        matched_points = sum(
            answer_point_matches(answer, group) for group in question.answer_point_groups
        )
        point_recall = matched_points / len(question.answer_point_groups)
        citation_hit = any(citation_matches)
        citation_precision = (
            sum(citation_matches) / len(citation_matches) if citation_matches else 0.0
        )
        passed = not refused and point_recall == 1.0 and citation_hit
    else:
        matched_points = 0
        point_recall = None
        citation_hit = None
        citation_precision = None
        passed = refused

    return GenerationEvaluation(
        id=question.id,
        question=question.question,
        category=question.category,
        answerable=question.answerable,
        refused=refused,
        answer=answer,
        answer_point_recall=point_recall,
        matched_answer_points=matched_points,
        total_answer_points=len(question.answer_point_groups),
        citation_hit=citation_hit,
        citation_precision=citation_precision,
        cited_chunk_ids=chunk_ids,
        cited_pages=pages,
        passed=passed,
        latency_ms=float(saved.get("latency_ms", 0.0)),
        prompt_tokens=int(saved.get("prompt_tokens", 0)),
        completion_tokens=int(saved.get("completion_tokens", 0)),
        total_tokens=int(saved.get("total_tokens", 0)),
        error=str(saved["error"]) if saved.get("error") else None,
    )


def percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def calculate_generation_metrics(rows: Sequence[GenerationEvaluation]) -> GenerationMetrics:
    if not rows:
        raise ValueError("Cannot calculate metrics without generation results")
    answerable = [row for row in rows if row.answerable]
    unanswerable = [row for row in rows if not row.answerable]
    citation_precisions = [
        row.citation_precision for row in answerable if row.citation_precision is not None
    ]
    latencies = [row.latency_ms for row in rows if row.error is None]
    return GenerationMetrics(
        question_count=len(rows),
        answerable_count=len(answerable),
        unanswerable_count=len(unanswerable),
        pass_rate=sum(row.passed for row in rows) / len(rows),
        answer_point_recall=statistics.fmean(
            row.answer_point_recall or 0.0 for row in answerable
        ),
        fully_correct_answer_rate=(
            sum(row.answer_point_recall == 1.0 and not row.refused for row in answerable)
            / len(answerable)
            if answerable
            else 0.0
        ),
        citation_hit_rate=(
            sum(row.citation_hit is True for row in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        citation_precision=(statistics.fmean(citation_precisions) if citation_precisions else 0.0),
        refusal_accuracy=(
            sum(row.refused for row in unanswerable) / len(unanswerable)
            if unanswerable
            else 0.0
        ),
        average_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
        p95_latency_ms=percentile_nearest_rank(latencies, 0.95),
        total_prompt_tokens=sum(row.prompt_tokens for row in rows),
        total_completion_tokens=sum(row.completion_tokens for row in rows),
        total_tokens=sum(row.total_tokens for row in rows),
    )


def evaluation_report(
    metrics: GenerationMetrics,
    rows: Sequence[GenerationEvaluation],
) -> dict[str, Any]:
    return {"metrics": asdict(metrics), "details": [asdict(row) for row in rows]}
