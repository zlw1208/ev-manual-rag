from pathlib import Path

import pytest

from app.retrieval.hybrid import HybridRetriever, build_filter, point_id_for_chunk
from app.retrieval.reranker import (
    CrossEncoderReranker,
    rerank_with_passage_scores,
    rerank_with_scores,
    split_reranker_passages,
)
from app.retrieval.sparse import ChineseSparseEncoder


def test_sparse_encoder_preserves_chinese_and_error_codes() -> None:
    encoder = ChineseSparseEncoder()
    tokens = encoder.tokenize("M5充电时出现BMS故障，SOC低于20%")

    assert "m5" in tokens
    assert "bms" in tokens
    assert "20%" in tokens
    assert "充电" in tokens
    assert "故障" in tokens


def test_sparse_vectors_are_sorted_and_non_empty() -> None:
    encoder = ChineseSparseEncoder()
    encoder.fit(["车辆无法充电", "动力电池保养"])

    vector = encoder.encode_document("车辆无法充电")

    assert vector.indices == sorted(vector.indices)
    assert len(vector.indices) == len(vector.values)
    assert vector.indices


def test_point_ids_are_stable() -> None:
    assert point_id_for_chunk("chunk-1") == point_id_for_chunk("chunk-1")
    assert point_id_for_chunk("chunk-1") != point_id_for_chunk("chunk-2")


def test_empty_metadata_produces_no_filter() -> None:
    assert build_filter(brand=None, model=None) is None


def test_retriever_requires_exactly_one_qdrant_connection_mode() -> None:
    with pytest.raises(ValueError, match="storage_path or qdrant_url is required"):
        HybridRetriever()

    with pytest.raises(ValueError, match="either storage_path or qdrant_url"):
        HybridRetriever(storage_path=Path("local"), qdrant_url="http://qdrant:6333")


def test_reranker_reorders_results_and_preserves_first_stage_score() -> None:
    from app.retrieval.hybrid import SearchResult

    results = [
        SearchResult(0.9, "first", ["A"], 1, 1, "first", {}),
        SearchResult(0.8, "second", ["B"], 2, 2, "second", {}),
    ]
    reranked = rerank_with_scores(results, [-1.0, 2.0], limit=1)
    assert reranked[0].chunk_id == "second"
    assert reranked[0].score == 2.0
    assert reranked[0].payload["first_stage_score"] == 0.8


def test_long_chunks_are_split_with_overlap() -> None:
    text = "甲" * 799

    passages = split_reranker_passages(
        text,
        max_chars=400,
        overlap_chars=80,
        split_threshold=500,
    )

    assert [len(passage) for passage in passages] == [400, 400, 159]
    assert passages[0][-80:] == passages[1][:80]
    assert passages[1][-80:] == passages[2][:80]


def test_passage_scores_use_max_pooling_for_original_chunk() -> None:
    from app.retrieval.hybrid import SearchResult

    long_result = SearchResult(0.8, "long", ["轮胎"], 1, 1, "long", {})
    short_result = SearchResult(0.9, "short", ["轮胎"], 2, 2, "short", {})
    passages = [["long-1", "long-2", "long-3"], ["short-1"]]

    reranked = rerank_with_passage_scores(
        [long_result, short_result],
        passages,
        [-3.0, -1.0, 6.5, 2.0],
        limit=2,
    )

    assert [result.chunk_id for result in reranked] == ["long", "short"]
    assert reranked[0].score == 6.5
    assert reranked[0].payload["reranker_best_passage"] == "long-3"
    assert reranked[0].payload["reranker_best_passage_index"] == 2
    assert reranked[0].payload["reranker_passage_count"] == 3


def test_cross_encoder_reranker_scores_flattened_passages() -> None:
    from app.retrieval.hybrid import SearchResult

    class FakeEncoder:
        documents: list[str]

        def rerank(
            self, query: str, documents: list[str], batch_size: int
        ) -> list[float]:
            assert query == "防滑链装在哪里"
            assert batch_size == 8
            self.documents = documents
            return [0.1, 0.2, 5.0, 1.0]

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.batch_size = 8
    reranker.split_threshold = 500
    reranker.passage_chars = 400
    reranker.overlap_chars = 80
    reranker.encoder = FakeEncoder()
    results = [
        SearchResult(
            0.8,
            "long",
            ["轮胎与车轮"],
            1,
            1,
            "无关" * 350 + "防滑链应安装在后轮",
            {"brand": "AITO", "model": "M5"},
        ),
        SearchResult(0.9, "short", ["轮胎与车轮"], 2, 2, "普通内容", {}),
    ]

    reranked = reranker.rerank("防滑链装在哪里", results, limit=2)

    assert reranked[0].chunk_id == "long"
    assert reranked[0].payload["reranker_passage_count"] == 3
    assert "防滑链应安装在后轮" in reranked[0].payload["reranker_best_passage"]
    assert len(reranker.encoder.documents) == 4
