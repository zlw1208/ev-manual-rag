from __future__ import annotations

import os
from typing import Any

from app.generation.schemas import ModelAnswer


class OpenAIResponsesClient:
    """Structured answer client backed by the OpenAI Responses API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("LLM model name is required")
        if not api_key:
            raise ValueError("LLM API key is required")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    'Install generation dependencies with: pip install -e ".[generation]"'
                ) from error
            kwargs: dict[str, str] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
        self.client = client
        self.model = model

    @classmethod
    def from_env(cls) -> OpenAIResponsesClient:
        return cls(
            model=os.getenv("LLM_MODEL", "").strip(),
            api_key=(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip(),
            base_url=(os.getenv("LLM_BASE_URL") or "").strip() or None,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> ModelAnswer:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=ModelAnswer,
        )
        if response.output_parsed is None:
            raise RuntimeError("The model did not return a structured answer")
        return response.output_parsed
