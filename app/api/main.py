from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response

from app.api.schemas import (
    AskRequest,
    AskResponse,
    EvidenceResponse,
    TokenUsageResponse,
)
from app.manual_qa import ManualQAResponse, ManualQAService
from app.observability import create_request_id, log_event, question_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_PATH = PROJECT_ROOT / "qdrant_storage"
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


class QAService(Protocol):
    def ask(self, question: str) -> ManualQAResponse: ...

    def close(self) -> None: ...


def build_qa_service() -> QAService:
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        return ManualQAService(qdrant_url=qdrant_url)
    return ManualQAService(storage_path=STORAGE_PATH)


def _serialize_response(response: ManualQAResponse, request_id: str) -> AskResponse:
    evidence = [
        EvidenceResponse(
            chunk_id=item.chunk_id,
            section=" > ".join(item.section_path) or "未命名章节",
            page_start=item.page_start,
            page_end=item.page_end,
            text=item.text,
            reranker_score=item.score,
            first_stage_score=item.payload.get("first_stage_score"),
            reranker_best_passage_index=item.payload.get(
                "reranker_best_passage_index"
            ),
            reranker_passage_count=item.payload.get("reranker_passage_count"),
        )
        for item in response.evidence
    ]
    return AskResponse(
        request_id=request_id,
        answer=response.answer.answer,
        refused=response.answer.refused,
        refusal_reason=response.answer.refusal_reason,
        citations=response.answer.citations,
        evidence=evidence,
        elapsed_seconds=response.elapsed_seconds,
        usage=TokenUsageResponse(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        ),
    )


def create_app(
    service_factory: Callable[[], QAService] = build_qa_service,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> Iterator[None]:
        service = service_factory()
        application.state.qa_service = service
        try:
            yield
        finally:
            service.close()

    application = FastAPI(
        title="EV Manual RAG API",
        description="新能源汽车说明书智能问答服务",
        version="0.2.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_tracking(request: Request, call_next: Callable) -> Response:
        request_id = create_request_id()
        request.state.request_id = request_id
        started_at = perf_counter()
        response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            response = await call_next(request)
            response_status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            log_event(
                logger,
                "http_request_completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response_status,
                elapsed_ms=round((perf_counter() - started_at) * 1000, 2),
            )

    @application.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        """Return a lightweight process health status."""
        return {"status": "ok", "service": "ev-manual-rag"}

    @application.post(
        "/ask",
        response_model=AskResponse,
        tags=["question-answering"],
        summary="询问新能源汽车说明书问题",
    )
    def ask_manual(payload: AskRequest, request: Request) -> AskResponse:
        question = payload.question.strip()
        if not question:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="问题不能为空",
            )
        service: QAService = request.app.state.qa_service
        try:
            response = service.ask(question)
            log_event(
                logger,
                "qa_completed",
                request_id=request.state.request_id,
                question_chars=len(question),
                question_fingerprint=question_fingerprint(question),
                refused=response.answer.refused,
                evidence_count=len(response.evidence),
                elapsed_ms=round(response.elapsed_seconds * 1000, 2),
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            )
            return _serialize_response(response, request.state.request_id)
        except ValueError as error:
            log_event(
                logger,
                "qa_validation_failed",
                level=logging.WARNING,
                request_id=request.state.request_id,
                question_chars=len(question),
                error_type=type(error).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        except Exception as error:
            log_event(
                logger,
                "qa_failed",
                level=logging.ERROR,
                request_id=request.state.request_id,
                question_chars=len(question),
                question_fingerprint=question_fingerprint(question),
                error_type=type(error).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="问答服务暂时不可用，请稍后重试",
            ) from error

    return application


app = create_app()
