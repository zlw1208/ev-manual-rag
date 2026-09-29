from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.retrieval.hybrid import SearchResult

DEFAULT_DOCUMENT_TITLE = "《问界M5纯电版用户手册》"


@dataclass(frozen=True)
class Evidence:
    source_id: str
    result: SearchResult

    @property
    def section(self) -> str:
        return " > ".join(self.result.section_path) or "未命名章节"

    @property
    def document_title(self) -> str:
        return str(self.result.payload.get("document_title") or DEFAULT_DOCUMENT_TITLE)


def build_evidence(
    results: Sequence[SearchResult],
    max_sources: int = 3,
    max_total_chars: int = 6000,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    used_chars = 0
    for result in results[:max_sources]:
        remaining = max_total_chars - used_chars
        if remaining <= 0:
            break
        text = result.text.strip()
        if not text:
            continue
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "……"
            result = SearchResult(
                score=result.score,
                chunk_id=result.chunk_id,
                section_path=result.section_path,
                page_start=result.page_start,
                page_end=result.page_end,
                text=text,
                payload=result.payload,
            )
        evidence.append(Evidence(source_id=f"S{len(evidence) + 1}", result=result))
        used_chars += len(text)
    return evidence


def format_evidence(evidence: Sequence[Evidence]) -> str:
    blocks = []
    for item in evidence:
        page = (
            str(item.result.page_start)
            if item.result.page_start == item.result.page_end
            else f"{item.result.page_start}-{item.result.page_end}"
        )
        blocks.append(
            "\n".join(
                [
                    f"[{item.source_id}]",
                    f"文档：{item.document_title}",
                    f"章节：{item.section}",
                    f"页码：{page}",
                    f"内容：{item.result.text}",
                ]
            )
        )
    return "\n\n".join(blocks)
