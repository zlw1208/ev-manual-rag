FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts
COPY data/processed/*.chunks.jsonl ./data/processed/

ENV PIP_DEFAULT_TIMEOUT=1000

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install ".[retrieval,generation,ui]"

EXPOSE 8001 8501
