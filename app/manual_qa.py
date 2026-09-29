from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.generation.qwen_client import QwenChatCompletionsClient
from app.generation.schemas import GroundedAnswer
from app.generation.service import AnswerService
from app.retrieval.hybrid import (
    DEFAULT_COLLECTION,
    DEFAULT_DENSE_MODEL,
    HybridRetriever,
    SearchResult,
)
from app.retrieval.reranker import CrossEncoderReranker


@dataclass(frozen=True)
class ManualQAResponse:
    answer: GroundedAnswer
    evidence: list[SearchResult]
    elapsed_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ManualQAService:
    """End-to-end retrieval, reranking and grounded answer generation."""

    def __init__(
        self,
        storage_path: Path | None = None,
        qdrant_url: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        dense_model: str = DEFAULT_DENSE_MODEL,
        candidate_limit: int = 10,
        context_limit: int = 3,
    ) -> None:
        self.candidate_limit = candidate_limit
        self.context_limit = context_limit
        self.retriever = HybridRetriever(
            storage_path=storage_path,
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            dense_model=dense_model,
        )
        self.reranker = CrossEncoderReranker()
        self.generation_client = QwenChatCompletionsClient.from_env()
        self.answer_service = AnswerService(
            self.generation_client,
            max_sources=context_limit,
        )

    def close(self) -> None:
        self.retriever.close()

    def ask(self, question: str) -> ManualQAResponse:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("问题不能为空")

        started_at = perf_counter()
        candidates = self.retriever.search(
            normalized_question,
            mode="hybrid",
            limit=self.candidate_limit,
            brand="AITO",
            model="M5",
            power_type="纯电",
        )
        evidence = self.reranker.rerank(
            normalized_question,
            candidates,
            limit=self.context_limit,
        )
        self.generation_client.reset_usage()
        answer = self.answer_service.answer(normalized_question, evidence)
        return ManualQAResponse(
            answer=answer,
            evidence=evidence,
            elapsed_seconds=perf_counter() - started_at,
            prompt_tokens=self.generation_client.last_usage.prompt_tokens,
            completion_tokens=self.generation_client.last_usage.completion_tokens,
            total_tokens=self.generation_client.last_usage.total_tokens,
        )
