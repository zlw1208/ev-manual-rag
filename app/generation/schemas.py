from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelAnswer(BaseModel):
    """Structured output requested from the language model."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Chinese answer grounded only in the supplied evidence")
    source_ids: list[str] = Field(description="Evidence ids that directly support the answer")
    refused: bool = Field(description="True when the evidence cannot answer the question")
    refusal_reason: str | None = Field(
        description="Why the evidence is insufficient; null when refused is false"
    )


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    chunk_id: str
    document_title: str
    section: str
    page_start: int
    page_end: int

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"第{self.page_start}页"
        return f"第{self.page_start}–{self.page_end}页"


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: str | None
