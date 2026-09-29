from __future__ import annotations

from types import SimpleNamespace

from app.generation.context import build_evidence
from app.generation.openai_client import OpenAIResponsesClient
from app.generation.prompt import SYSTEM_PROMPT, extract_question_conditions
from app.generation.qwen_client import QwenChatCompletionsClient
from app.generation.schemas import ModelAnswer
from app.generation.service import AnswerService, needs_coverage_review, validate_and_ground
from app.retrieval.hybrid import SearchResult


def result(chunk_id: str = "chunk-1", page: int = 175) -> SearchResult:
    return SearchResult(
        score=8.0,
        chunk_id=chunk_id,
        section_path=["跨接启动"],
        page_start=page,
        page_end=page,
        text="低压蓄电池馈电时，可以通过跨接启动启动车辆。",
        payload={"document_title": "测试手册"},
    )


class FakeClient:
    def __init__(self, answer: ModelAnswer) -> None:
        self.answer = answer

    def generate(self, system_prompt: str, user_prompt: str) -> ModelAnswer:
        assert "只能依据" in system_prompt
        assert "[S1]" in user_prompt
        return self.answer


def test_grounded_answer_uses_server_side_citation_metadata() -> None:
    evidence = build_evidence([result()])
    grounded = validate_and_ground(
        ModelAnswer(
            answer="可以进行跨接启动。",
            source_ids=["S1"],
            refused=False,
            refusal_reason=None,
        ),
        evidence,
    )
    assert grounded.citations[0].page_start == 175
    assert grounded.citations[0].section == "跨接启动"
    assert grounded.citations[0].document_title == "测试手册"


def test_system_prompt_requires_all_subquestions_to_be_answered() -> None:
    assert "逐项核对" in SYSTEM_PROMPT
    assert "全部并列要求" in SYSTEM_PROMPT
    assert "必须全部回答" in SYSTEM_PROMPT


def test_compound_question_is_reviewed_once_for_coverage() -> None:
    answers = iter(
        [
            ModelAnswer(
                answer="每月充满一次。",
                source_ids=["S1"],
                refused=False,
                refusal_reason=None,
            ),
            ModelAnswer(
                answer="每月至少使用一次并充满一次；日常建议每周充满一次。",
                source_ids=["S1"],
                refused=False,
                refusal_reason=None,
            ),
        ]
    )

    class ReviewClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, system_prompt: str, user_prompt: str) -> ModelAnswer:
            self.prompts.append(user_prompt)
            return next(answers)

    client = ReviewClient()
    answer = AnswerService(client).answer(
        "长期停放每月要做什么？日常又多久充满一次？",
        [result()],
    )

    assert len(client.prompts) == 2
    assert "初稿答案" in client.prompts[1]
    assert "每月至少使用一次" in answer.answer


def test_simple_question_does_not_add_coverage_review() -> None:
    assert needs_coverage_review("低压蓄电池没电怎么办？") is False
    assert needs_coverage_review("高度分别怎么调？") is True


def test_question_conditions_extract_time_threshold_and_gear() -> None:
    assert extract_question_conditions("每月至少一次，低于20%或挂P挡时怎么办？") == (
        "每月",
        "低于20%",
        "P挡",
    )


def test_answer_service_rejects_hallucinated_source_id() -> None:
    client = FakeClient(
        ModelAnswer(
            answer="无效回答",
            source_ids=["S99"],
            refused=False,
            refusal_reason=None,
        )
    )
    answer = AnswerService(client).answer("怎么搭电？", [result()])
    assert answer.refused is True
    assert answer.citations == []


def test_answer_service_refuses_when_no_evidence() -> None:
    client = FakeClient(
        ModelAnswer(answer="unused", source_ids=[], refused=True, refusal_reason="资料不足")
    )
    answer = AnswerService(client).answer("明年会降价吗？", [])
    assert answer.refused is True
    assert "没有检索到" in (answer.refusal_reason or "")


def test_build_evidence_respects_source_and_character_limits() -> None:
    results = [result("one", 1), result("two", 2), result("three", 3)]
    evidence = build_evidence(results, max_sources=2, max_total_chars=20)
    assert len(evidence) == 1
    assert evidence[0].source_id == "S1"
    assert evidence[0].result.text.endswith("……")


def test_openai_client_requests_structured_output() -> None:
    expected = ModelAnswer(
        answer="可以跨接启动。",
        source_ids=["S1"],
        refused=False,
        refusal_reason=None,
    )

    class FakeResponses:
        def parse(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["model"] == "test-model"
            assert kwargs["text_format"] is ModelAnswer
            return SimpleNamespace(output_parsed=expected)

    fake_client = SimpleNamespace(responses=FakeResponses())
    client = OpenAIResponsesClient(
        model="test-model",
        api_key="test-key",
        client=fake_client,
    )
    assert client.generate("system", "user") == expected


def test_qwen_client_requests_schema_and_parses_chat_completion() -> None:
    content = """```json
{"answer":"可以跨接启动。","source_ids":["S1"],"refused":false,"refusal_reason":null}
```"""

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["model"] == "qwen-plus"
            assert kwargs["temperature"] == 0
            response_format = kwargs["response_format"]
            assert response_format["type"] == "json_schema"
            assert response_format["json_schema"]["strict"] is True
            assert kwargs["extra_body"] == {"enable_thinking": False}
            messages = kwargs["messages"]
            assert isinstance(messages, list)
            assert "JSON" in messages[1]["content"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = QwenChatCompletionsClient(
        model="qwen-plus",
        api_key="test-key",
        client=fake_client,
    )
    answer = client.generate("system", "user")
    assert answer.answer == "可以跨接启动。"
    assert answer.source_ids == ["S1"]


def test_qwen_client_retries_invalid_json_once() -> None:
    contents = iter(
        [
            "不是 JSON",
            (
                '{"answer":"资料不足。","source_ids":[],"refused":true,'
                '"refusal_reason":"资料不足"}'
            ),
        ]
    )

    class FakeCompletions:
        calls = 0

        def create(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            if self.calls == 2:
                messages = kwargs["messages"]
                assert isinstance(messages, list)
                assert len(messages) == 4
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(contents)))]
            )

    completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = QwenChatCompletionsClient(
        model="qwen-plus",
        api_key="test-key",
        client=fake_client,
    )
    answer = client.generate("system", "user")
    assert completions.calls == 2
    assert answer.refused is True
