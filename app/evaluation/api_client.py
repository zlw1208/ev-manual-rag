from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.api.schemas import AskResponse
from app.generation.schemas import GroundedAnswer
from app.manual_qa import ManualQAResponse
from app.retrieval.hybrid import SearchResult


def api_response_to_manual_qa(response: AskResponse) -> ManualQAResponse:
    evidence = [
        SearchResult(
            score=item.reranker_score,
            chunk_id=item.chunk_id,
            section_path=item.section.split(" > ") if item.section else [],
            page_start=item.page_start,
            page_end=item.page_end,
            text=item.text,
            payload={
                "first_stage_score": item.first_stage_score,
                "reranker_best_passage_index": item.reranker_best_passage_index,
                "reranker_passage_count": item.reranker_passage_count,
            },
        )
        for item in response.evidence
    ]
    return ManualQAResponse(
        answer=GroundedAnswer(
            answer=response.answer,
            citations=response.citations,
            refused=response.refused,
            refusal_reason=response.refusal_reason,
        ),
        evidence=evidence,
        elapsed_seconds=response.elapsed_seconds,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )


class ApiManualQAClient:
    """Question answerer that evaluates the deployed FastAPI service over HTTP."""

    def __init__(self, base_url: str, timeout_seconds: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def ask(self, question: str) -> ManualQAResponse:
        request = Request(
            f"{self.base_url}/ask",
            data=json.dumps({"question": question}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as raw_response:
                response = AskResponse.model_validate_json(raw_response.read())
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"QA API returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Cannot connect to QA API at {self.base_url}") from error
        return api_response_to_manual_qa(response)

    def close(self) -> None:
        return None
