import json
import logging

from fastapi.testclient import TestClient

from app.api.main import app, create_app
from app.generation.schemas import Citation, GroundedAnswer
from app.manual_qa import ManualQAResponse
from app.retrieval.hybrid import SearchResult

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ev-manual-rag"}
    assert len(response.headers["X-Request-ID"]) == 32


class FakeQAService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.questions: list[str] = []
        self.closed = False

    def ask(self, question: str) -> ManualQAResponse:
        self.questions.append(question)
        if self.error:
            raise self.error
        citation = Citation(
            source_id="S1",
            chunk_id="chunk-1",
            document_title="问界 M5 纯电版用户手册",
            section="充电",
            page_start=10,
            page_end=11,
        )
        evidence = SearchResult(
            score=3.2,
            chunk_id="chunk-1",
            section_path=["充电"],
            page_start=10,
            page_end=11,
            text="充电说明",
            payload={
                "first_stage_score": 0.8,
                "reranker_best_passage_index": 1,
                "reranker_passage_count": 3,
            },
        )
        return ManualQAResponse(
            answer=GroundedAnswer(
                answer="请按照说明书操作。",
                citations=[citation],
                refused=False,
                refusal_reason=None,
            ),
            evidence=[evidence],
            elapsed_seconds=1.25,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )

    def close(self) -> None:
        self.closed = True


def test_ask_returns_grounded_answer_and_closes_service(caplog) -> None:
    service = FakeQAService()
    test_app = create_app(lambda: service)

    with (
        caplog.at_level(logging.INFO, logger="uvicorn.error"),
        TestClient(test_app) as test_client,
    ):
        response = test_client.post("/ask", json={"question": "  如何充电？  "})

    assert response.status_code == 200
    assert service.questions == ["如何充电？"]
    assert service.closed is True
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["answer"] == "请按照说明书操作。"
    assert body["citations"][0]["chunk_id"] == "chunk-1"
    assert body["evidence"][0]["reranker_score"] == 3.2
    assert body["evidence"][0]["first_stage_score"] == 0.8
    assert body["evidence"][0]["reranker_passage_count"] == 3
    assert body["usage"]["total_tokens"] == 120
    events = [json.loads(record.message) for record in caplog.records]
    qa_event = next(event for event in events if event["event"] == "qa_completed")
    assert qa_event["request_id"] == body["request_id"]
    assert qa_event["evidence_count"] == 1
    assert qa_event["total_tokens"] == 120
    assert "如何充电" not in caplog.text


def test_ask_rejects_blank_question() -> None:
    service = FakeQAService()
    test_app = create_app(lambda: service)

    with TestClient(test_app) as test_client:
        response = test_client.post("/ask", json={"question": "   "})

    assert response.status_code == 422
    assert response.json()["detail"] == "问题不能为空"
    assert service.questions == []


def test_ask_hides_internal_provider_errors() -> None:
    service = FakeQAService(RuntimeError("secret provider detail"))
    test_app = create_app(lambda: service)

    with TestClient(test_app) as test_client:
        response = test_client.post("/ask", json={"question": "如何充电？"})

    assert response.status_code == 503
    assert response.json()["detail"] == "问答服务暂时不可用，请稍后重试"
    assert "secret provider detail" not in response.text
