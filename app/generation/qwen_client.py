from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from app.generation.schemas import ModelAnswer

DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TokenUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _extract_json(text: str) -> str:
    """Extract one JSON object from a plain or Markdown-fenced model response."""
    content = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        content = fenced.group(1).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end < start:
        raise ValueError("模型响应中没有 JSON 对象")
    return content[start : end + 1]


def _format_instruction() -> str:
    schema = json.dumps(ModelAnswer.model_json_schema(), ensure_ascii=False)
    return f"""请只输出一个 JSON 对象，不要输出 Markdown 代码块或其他说明。
JSON 必须符合以下 Schema：
{schema}

字段约束：
- 能回答时 refused=false、refusal_reason=null，并填写直接支持答案的 source_ids。
- 资料不足时 refused=true、source_ids=[]，并填写 refusal_reason。
"""


class QwenChatCompletionsClient:
    """Qwen client using Alibaba Model Studio's OpenAI-compatible Chat API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        client: Any | None = None,
        max_format_retries: int = 1,
    ) -> None:
        if not model:
            raise ValueError("LLM model name is required")
        if not api_key:
            raise ValueError("DashScope API key is required")
        if not base_url:
            raise ValueError("LLM base URL is required")
        if max_format_retries < 0:
            raise ValueError("max_format_retries must be non-negative")

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    'Install generation dependencies with: pip install -e ".[generation]"'
                ) from error
            client = OpenAI(api_key=api_key, base_url=base_url)

        self.client = client
        self.model = model
        self.max_format_retries = max_format_retries
        self.last_usage = TokenUsage()

    def reset_usage(self) -> None:
        self.last_usage = TokenUsage()

    @classmethod
    def from_env(cls) -> QwenChatCompletionsClient:
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            model=(os.getenv("LLM_MODEL") or DEFAULT_QWEN_MODEL).strip(),
            api_key=(os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY") or "").strip(),
            base_url=(os.getenv("LLM_BASE_URL") or DEFAULT_QWEN_BASE_URL).strip(),
        )

    def generate(self, system_prompt: str, user_prompt: str) -> ModelAnswer:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n\n{_format_instruction()}"},
        ]
        last_error: Exception | None = None
        for attempt in range(self.max_format_retries + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "manual_answer",
                        "strict": True,
                        "schema": ModelAnswer.model_json_schema(),
                    },
                },
                extra_body={"enable_thinking": False},
            )
            content = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.last_usage.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.last_usage.completion_tokens += int(
                    getattr(usage, "completion_tokens", 0) or 0
                )
            if not isinstance(content, str) or not content.strip():
                last_error = ValueError("模型返回了空响应")
            else:
                try:
                    return ModelAnswer.model_validate_json(_extract_json(content))
                except (ValueError, ValidationError) as error:
                    last_error = error

            if attempt < self.max_format_retries:
                messages.extend(
                    [
                        {"role": "assistant", "content": content or ""},
                        {
                            "role": "user",
                            "content": (
                                "上一次输出不符合要求。请严格按照指定 Schema 重新输出，"
                                "只返回一个合法 JSON 对象。"
                            ),
                        },
                    ]
                )

        raise RuntimeError(f"模型未返回有效的结构化答案：{last_error}") from last_error
