from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from qdrant_client import QdrantClient, models

from app.retrieval.sparse import ChineseSparseEncoder

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DEFAULT_DENSE_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION = "ev_manual_chunks"
DEFAULT_MODEL_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "fastembed"


def configure_model_cache(
    cache_path: Path | None = None,
    dense_model: str = DEFAULT_DENSE_MODEL,
) -> Path:
    """Keep FastEmbed model files in a stable, project-local cache by default."""
    resolved = (cache_path or DEFAULT_MODEL_CACHE).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(resolved))
    default_model_file = resolved / "fast-bge-small-zh-v1.5" / "model_optimized.onnx"
    if dense_model == DEFAULT_DENSE_MODEL and default_model_file.is_file():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return resolved


@dataclass(frozen=True)
class SearchResult:
    score: float
    chunk_id: str
    section_path: list[str]
    page_start: int
    page_end: int
    text: str
    payload: dict[str, Any]


def load_searchable_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid chunk JSON at line {line_number}: {error}") from error
            if chunk.get("searchable", True):
                chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No searchable chunks found in {path}")
    return chunks


def point_id_for_chunk(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ev-manual-rag:{chunk_id}"))


def build_filter(**values: str | int | None) -> models.Filter | None:
    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in values.items()
        if value is not None
    ]
    return models.Filter(must=conditions) if conditions else None


def build_local_index(
    chunks_path: Path,
    storage_path: Path,
    collection_name: str = DEFAULT_COLLECTION,
    dense_model: str = DEFAULT_DENSE_MODEL,
    model_cache: Path | None = None,
) -> int:
    configure_model_cache(model_cache, dense_model)
    storage_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(storage_path), local_inference_batch_size=16)
    try:
        return _build_index(
            client,
            chunks_path=chunks_path,
            collection_name=collection_name,
            dense_model=dense_model,
        )
    finally:
        client.close()


def build_remote_index(
    chunks_path: Path,
    qdrant_url: str,
    collection_name: str = DEFAULT_COLLECTION,
    dense_model: str = DEFAULT_DENSE_MODEL,
    model_cache: Path | None = None,
    skip_if_exists: bool = False,
    wait_seconds: float = 0,
) -> int:
    configure_model_cache(model_cache, dense_model)
    client = QdrantClient(url=qdrant_url, local_inference_batch_size=16)
    try:
        _wait_for_qdrant(client, qdrant_url, wait_seconds)
        if skip_if_exists and client.collection_exists(collection_name):
            existing_count = client.count(collection_name, exact=True).count
            if existing_count:
                return existing_count
        return _build_index(
            client,
            chunks_path=chunks_path,
            collection_name=collection_name,
            dense_model=dense_model,
        )
    finally:
        client.close()


def _wait_for_qdrant(
    client: QdrantClient,
    qdrant_url: str,
    wait_seconds: float,
) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            client.get_collections()
            return
        except Exception as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Qdrant is unavailable at {qdrant_url}") from error
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _build_index(
    client: QdrantClient,
    chunks_path: Path,
    collection_name: str,
    dense_model: str,
) -> int:
    chunks = load_searchable_chunks(chunks_path)
    retrieval_texts = [str(chunk["retrieval_text"]) for chunk in chunks]
    sparse_encoder = ChineseSparseEncoder()
    sparse_encoder.fit(retrieval_texts)

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=client.get_embedding_size(dense_model),
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    points = [
        models.PointStruct(
            id=point_id_for_chunk(str(chunk["chunk_id"])),
            vector={
                DENSE_VECTOR_NAME: models.Document(
                    text=str(chunk["retrieval_text"]),
                    model=dense_model,
                ),
                SPARSE_VECTOR_NAME: sparse_encoder.encode_document(
                    str(chunk["retrieval_text"])
                ),
            },
            payload=chunk,
        )
        for chunk in chunks
    ]
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=32,
        parallel=1,
    )
    return len(points)


class HybridRetriever:
    def __init__(
        self,
        storage_path: Path | None = None,
        qdrant_url: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        dense_model: str = DEFAULT_DENSE_MODEL,
        model_cache: Path | None = None,
    ) -> None:
        if storage_path is not None and qdrant_url is not None:
            raise ValueError("Configure either storage_path or qdrant_url, not both")
        if storage_path is None and qdrant_url is None:
            raise ValueError("storage_path or qdrant_url is required")
        configure_model_cache(model_cache, dense_model)
        self.collection_name = collection_name
        self.dense_model = dense_model
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, local_inference_batch_size=8)
        else:
            self.client = QdrantClient(
                path=str(storage_path), local_inference_batch_size=8
            )
        self.sparse_encoder = ChineseSparseEncoder()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(
        self,
        query: str,
        mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        limit: int = 5,
        candidate_limit: int = 20,
        brand: str | None = None,
        model: str | None = None,
        year: int | None = None,
        power_type: str | None = None,
    ) -> list[SearchResult]:
        query_filter = build_filter(
            brand=brand,
            model=model,
            year=year,
            power_type=power_type,
        )
        dense_query = models.Document(text=query, model=self.dense_model)
        sparse_query = self.sparse_encoder.encode_query(query)

        if mode == "dense":
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_query,
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        elif mode == "sparse":
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_query,
                using=SPARSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using=DENSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=candidate_limit,
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using=SPARSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=candidate_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )

        return [self._to_result(point) for point in response.points]

    @staticmethod
    def _to_result(point: models.ScoredPoint) -> SearchResult:
        payload = dict(point.payload or {})
        return SearchResult(
            score=float(point.score),
            chunk_id=str(payload.get("chunk_id", point.id)),
            section_path=list(payload.get("section_path", [])),
            page_start=int(payload.get("page_start", 0)),
            page_end=int(payload.get("page_end", 0)),
            text=str(payload.get("text", "")),
            payload=payload,
        )
