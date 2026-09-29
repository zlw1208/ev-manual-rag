from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any


def create_request_id() -> str:
    return uuid.uuid4().hex


def question_fingerprint(question: str) -> str:
    """Return a short stable identifier without logging the question itself."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
