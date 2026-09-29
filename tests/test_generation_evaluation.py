from __future__ import annotations

from app.api.schemas import AskResponse, EvidenceResponse, TokenUsageResponse
from app.evaluation.api_client import api_response_to_manual_qa
from app.evaluation.generation import (
    GenerationQuestion,
    answer_point_matches,
    calculate_generation_metrics,
    evaluate_answer,
)
from app.generation.schemas import Citation, GroundedAnswer
from app.manual_qa import ManualQAResponse


def question(answerable: bool = True) -> GenerationQuestion:
    return GenerationQuestion(
        id="q1",
        question="胎压何时报警？",
        category="安全",
        answerable=answerable,
        answer_point_groups=(("2.2bar", "2.2巴"), ("3.5bar", "3.5巴")) if answerable else (),
        expected_chunk_ids=frozenset({"chunk-1"}) if answerable else frozenset(),
        expected_pages=frozenset({118}) if answerable else frozenset(),
    )


def response(answer: GroundedAnswer) -> ManualQAResponse:
    return ManualQAResponse(
        answer=answer,
        evidence=[],
        elapsed_seconds=1.25,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )


def test_evaluate_answer_checks_points_and_citation() -> None:
    grounded = GroundedAnswer(
        answer="胎压不高于2.2 bar或不低于3.5 bar时报警。",
        refused=False,
        refusal_reason=None,
        citations=[
            Citation(
                source_id="S1",
                chunk_id="chunk-1",
                document_title="手册",
                section="胎压",
                page_start=118,
                page_end=118,
            )
        ],
    )
    row = evaluate_answer(question(), response(grounded))
    assert row.passed is True
    assert row.answer_point_recall == 1.0
    assert row.citation_precision == 1.0
    assert row.total_tokens == 120


def test_answer_point_matching_handles_common_chinese_paraphrases() -> None:
    assert answer_point_matches("雨刮必须处于“OFF”档位", ("雨刮处于OFF档位",))
    assert answer_point_matches("踩下刹车踏板即可退出", ("踩刹车",))
    assert answer_point_matches("液罐空时不能继续使用洗涤器", ("不能使用洗涤器",))
    assert answer_point_matches("切勿让前舱盖自由落下", ("请勿让其自由落下",))
    assert answer_point_matches(
        "主驾座椅不会执行迎宾功能",
        ("不会执行座椅迎宾功能",),
    )


def test_answer_point_matching_does_not_reverse_polarity_or_numbers() -> None:
    assert not answer_point_matches("传送带洗车时不可以开启", ("可以开启",))
    assert not answer_point_matches("胎压不低于2.5bar", ("胎压不低于3.5bar",))


def test_unanswerable_question_passes_only_when_refused() -> None:
    grounded = GroundedAnswer(
        answer="无法回答。",
        refused=True,
        refusal_reason="资料不足",
        citations=[],
    )
    row = evaluate_answer(question(answerable=False), response(grounded))
    assert row.passed is True
    assert row.answer_point_recall is None
    assert row.citation_hit is None


def test_generation_metrics_aggregate_answer_and_refusal_rows() -> None:
    answered = GroundedAnswer(
        answer="2.2bar和3.5bar",
        refused=False,
        refusal_reason=None,
        citations=[
            Citation(
                source_id="S1",
                chunk_id="chunk-1",
                document_title="手册",
                section="胎压",
                page_start=118,
                page_end=118,
            )
        ],
    )
    refused = GroundedAnswer(
        answer="无法回答。",
        refused=True,
        refusal_reason="资料不足",
        citations=[],
    )
    rows = [
        evaluate_answer(question(), response(answered)),
        evaluate_answer(question(answerable=False), response(refused)),
    ]
    metrics = calculate_generation_metrics(rows)
    assert metrics.pass_rate == 1.0
    assert metrics.refusal_accuracy == 1.0
    assert metrics.total_tokens == 240


def test_api_response_is_adapted_for_existing_evaluator() -> None:
    api_response = AskResponse(
        request_id="request-1",
        answer="胎压说明",
        refused=False,
        refusal_reason=None,
        citations=[
            Citation(
                source_id="S1",
                chunk_id="chunk-1",
                document_title="手册",
                section="轮胎 > 胎压",
                page_start=118,
                page_end=118,
            )
        ],
        evidence=[
            EvidenceResponse(
                chunk_id="chunk-1",
                section="轮胎 > 胎压",
                page_start=118,
                page_end=118,
                text="胎压说明原文",
                reranker_score=3.5,
                first_stage_score=0.8,
                reranker_best_passage_index=1,
                reranker_passage_count=2,
            )
        ],
        elapsed_seconds=1.2,
        usage=TokenUsageResponse(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
    )

    adapted = api_response_to_manual_qa(api_response)

    assert adapted.answer.answer == "胎压说明"
    assert adapted.evidence[0].section_path == ["轮胎", "胎压"]
    assert adapted.evidence[0].payload["first_stage_score"] == 0.8
    assert adapted.total_tokens == 120
