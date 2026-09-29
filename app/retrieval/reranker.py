from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.retrieval.hybrid import DEFAULT_MODEL_CACHE, SearchResult

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_RERANKER_FILE = "onnx/model_quint8_avx2.onnx"
DEFAULT_RERANKER_SPLIT_THRESHOLD = 500
DEFAULT_RERANKER_PASSAGE_CHARS = 400
DEFAULT_RERANKER_OVERLAP_CHARS = 80


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        limit: int,
    ) -> list[SearchResult]: ...


def rerank_with_scores(
    results: Sequence[SearchResult],
    scores: Sequence[float],
    limit: int,
) -> list[SearchResult]:
    if len(results) != len(scores):
        raise ValueError("Reranker returned a different number of scores than candidates")
    scored = []
    for result, score in zip(results, scores, strict=True):
        payload = {
            **result.payload,
            "first_stage_score": result.score,
            "reranker_score": float(score),
        }
        scored.append(replace(result, score=float(score), payload=payload))
    return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]


def split_reranker_passages(
    text: str,
    max_chars: int = DEFAULT_RERANKER_PASSAGE_CHARS,
    overlap_chars: int = DEFAULT_RERANKER_OVERLAP_CHARS,
    split_threshold: int = DEFAULT_RERANKER_SPLIT_THRESHOLD,
) -> list[str]:
    """Split only long chunks into overlapping windows for cross-encoder scoring."""
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")
    if split_threshold < max_chars:
        raise ValueError("split_threshold must be greater than or equal to max_chars")
    if len(text) <= split_threshold:
        return [text]

    step = max_chars - overlap_chars
    return [text[start : start + max_chars] for start in range(0, len(text), step)]


def rerank_with_passage_scores(
    results: Sequence[SearchResult],
    passages_by_result: Sequence[Sequence[str]],
    scores: Sequence[float],
    limit: int,
) -> list[SearchResult]:
    """Max-pool passage scores back to their original chunks."""
    if len(results) != len(passages_by_result):
        raise ValueError("Each candidate must have one passage group")
    if sum(len(passages) for passages in passages_by_result) != len(scores):
        raise ValueError("Reranker returned a different number of scores than passages")

    scored: list[SearchResult] = []
    score_offset = 0
    for result, passages in zip(results, passages_by_result, strict=True):
        if not passages:
            raise ValueError("Each candidate must contain at least one passage")
        passage_scores = [
            float(score)
            for score in scores[score_offset : score_offset + len(passages)]
        ]
        score_offset += len(passages)
        best_index = max(range(len(passage_scores)), key=passage_scores.__getitem__)
        best_score = passage_scores[best_index]
        payload = {
            **result.payload,
            "first_stage_score": result.score,
            "reranker_score": best_score,
            "reranker_best_passage": passages[best_index],
            "reranker_best_passage_index": best_index,
            "reranker_passage_count": len(passages),
        }
        scored.append(replace(result, score=best_score, payload=payload))

    return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        cache_dir: Path | None = None,
        threads: int | None = None,
        batch_size: int = 8,
        split_threshold: int = DEFAULT_RERANKER_SPLIT_THRESHOLD,
        passage_chars: int = DEFAULT_RERANKER_PASSAGE_CHARS,
        overlap_chars: int = DEFAULT_RERANKER_OVERLAP_CHARS,
    ) -> None:
        if not 0 <= overlap_chars < passage_chars:
            raise ValueError("overlap_chars must be non-negative and smaller than passage_chars")
        if split_threshold < passage_chars:
            raise ValueError("split_threshold must be greater than or equal to passage_chars")
        self.model_name = model_name
        self.batch_size = batch_size
        self.split_threshold = split_threshold
        self.passage_chars = passage_chars
        self.overlap_chars = overlap_chars
        resolved_cache = (cache_dir or DEFAULT_MODEL_CACHE).resolve()
        resolved_cache.mkdir(parents=True, exist_ok=True)
        self._register_default_model()

        cached_file = next(resolved_cache.rglob(Path(DEFAULT_RERANKER_FILE).name), None)
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        if cached_file is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            self.encoder = TextCrossEncoder(
                model_name=model_name,
                cache_dir=str(resolved_cache),
                threads=threads,
            )
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline

    @staticmethod
    def _register_default_model() -> None:
        supported = {item["model"] for item in TextCrossEncoder.list_supported_models()}
        if DEFAULT_RERANKER_MODEL not in supported:
            TextCrossEncoder.add_custom_model(
                model=DEFAULT_RERANKER_MODEL,
                sources=ModelSource(hf=DEFAULT_RERANKER_MODEL),
                model_file=DEFAULT_RERANKER_FILE,
                description="Multilingual MiniLM reranker trained on mMARCO",
                license="apache-2.0",
                size_in_gb=0.14,
            )

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        limit: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []
        raw_passages_by_result = [
            split_reranker_passages(
                result.text,
                max_chars=self.passage_chars,
                overlap_chars=self.overlap_chars,
                split_threshold=self.split_threshold,
            )
            for result in results
        ]
        passages_by_result = [
            [self._passage_with_metadata(result, passage) for passage in passages]
            for result, passages in zip(results, raw_passages_by_result, strict=True)
        ]
        documents = [
            passage for passages in passages_by_result for passage in passages
        ]
        scores = list(
            self.encoder.rerank(query, documents, batch_size=self.batch_size)
        )
        return rerank_with_passage_scores(results, passages_by_result, scores, limit)

    @staticmethod
    def _passage_with_metadata(result: SearchResult, passage: str) -> str:
        payload = result.payload
        section_label = " > ".join(result.section_path)
        metadata = [
            f"品牌：{payload.get('brand')}" if payload.get("brand") else "",
            f"车型：{payload.get('model')}" if payload.get("model") else "",
            (
                f"动力类型：{payload.get('power_type')}"
                if payload.get("power_type")
                else ""
            ),
            f"章节：{section_label}" if section_label else "",
        ]
        prefix = "\n".join(item for item in metadata if item)
        return f"{prefix}\n正文：{passage}" if prefix else passage


def reranker_metadata(result: SearchResult) -> dict[str, Any]:
    return {
        "first_stage_score": result.payload.get("first_stage_score"),
        "reranker_score": result.payload.get("reranker_score"),
        "reranker_best_passage_index": result.payload.get(
            "reranker_best_passage_index"
        ),
        "reranker_passage_count": result.payload.get("reranker_passage_count"),
    }
