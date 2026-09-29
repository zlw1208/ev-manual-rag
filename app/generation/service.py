from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from app.generation.context import Evidence, build_evidence
from app.generation.prompt import (
    SYSTEM_PROMPT,
    build_coverage_review_prompt,
    build_user_prompt,
)
from app.generation.schemas import Citation, GroundedAnswer, ModelAnswer
from app.retrieval.hybrid import SearchResult


class StructuredAnswerClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> ModelAnswer: ...


class CitationValidationError(ValueError):
    """Raised when model-provided citation ids do not match supplied evidence."""


_COMPOUND_QUESTION_PATTERN = re.compile(r"又|分别|哪些|以及|同时|并且|并说明|各自")


def needs_coverage_review(question: str) -> bool:
    return bool(_COMPOUND_QUESTION_PATTERN.search(question)) or question.count("？") > 1


def refusal(reason: str) -> GroundedAnswer:
    return GroundedAnswer(
        answer=f"根据当前检索到的说明书资料，无法回答这个问题。{reason}",
        citations=[],
        refused=True,
        refusal_reason=reason,
    )


def validate_and_ground(
    model_answer: ModelAnswer,
    evidence: Sequence[Evidence],
) -> GroundedAnswer:
    evidence_by_id = {item.source_id: item for item in evidence}
    if model_answer.refused:
        reason = model_answer.refusal_reason or "现有资料不足。"
        return refusal(reason)

    unique_ids = list(dict.fromkeys(model_answer.source_ids))
    if not unique_ids:
        raise CitationValidationError("A non-refusal answer must cite at least one source")
    unknown_ids = [source_id for source_id in unique_ids if source_id not in evidence_by_id]
    if unknown_ids:
        raise CitationValidationError(f"Unknown source ids: {', '.join(unknown_ids)}")

    citations = []
    for source_id in unique_ids:
        item = evidence_by_id[source_id]
        citations.append(
            Citation(
                source_id=source_id,
                chunk_id=item.result.chunk_id,
                document_title=item.document_title,
                section=item.section,
                page_start=item.result.page_start,
                page_end=item.result.page_end,
            )
        )
    return GroundedAnswer(
        answer=model_answer.answer.strip(),
        citations=citations,
        refused=False,
        refusal_reason=None,
    )


class AnswerService:
    def __init__(
        self,
        client: StructuredAnswerClient,
        max_sources: int = 3,
        max_context_chars: int = 6000,
    ) -> None:
        self.client = client
        self.max_sources = max_sources
        self.max_context_chars = max_context_chars

    def answer(self, question: str, results: Sequence[SearchResult]) -> GroundedAnswer:
        evidence = build_evidence(
            results,
            max_sources=self.max_sources,
            max_total_chars=self.max_context_chars,
        )
        if not evidence:
            return refusal("没有检索到可用的说明书内容。")
        raw_answer = self.client.generate(
            SYSTEM_PROMPT,
            build_user_prompt(question, evidence),
        )
        try:
            grounded_draft = validate_and_ground(raw_answer, evidence)
        except CitationValidationError:
            return refusal("模型未能提供可验证的说明书引用。")

        if grounded_draft.refused or not needs_coverage_review(question):
            return grounded_draft

        reviewed_answer = self.client.generate(
            SYSTEM_PROMPT,
            build_coverage_review_prompt(question, evidence, grounded_draft.answer),
        )
        try:
            return validate_and_ground(reviewed_answer, evidence)
        except CitationValidationError:
            return grounded_draft
