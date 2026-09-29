from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.generation.schemas import Citation


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500, description="用户的车辆问题")


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    section: str
    page_start: int
    page_end: int
    text: str
    reranker_score: float
    first_stage_score: float | None = None
    reranker_best_passage_index: int | None = None
    reranker_passage_count: int | None = None


class TokenUsageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    evidence: list[EvidenceResponse]
    elapsed_seconds: float
    usage: TokenUsageResponse
