from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from app.ingestion.sources import ManualSource

LIST_ITEM_PATTERN = re.compile(r"^(?:[●•▪◦*-]|\d{1,2}[.、])\s*", re.MULTILINE)
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[。！？；])")
SUBHEADING_PATTERN = re.compile(
    r"(?:说明|警告|设置|状态|故障|设备|接口|模式|方法|方式|介绍|概述|保养|操作|启动|关闭|功能|检查|维护|救援|控制|调节|须知|系统)$"
)
TERMINAL_PUNCTUATION = ("。", "！", "？", "；", ":", "：")
SAFETY_LEVELS = {"危险": "danger", "警告": "warning", "注意": "caution", "提示": "notice"}


@dataclass(frozen=True)
class PageRecord:
    document_id: str
    page_number: int
    headings: list[str]
    text: str
    safety_markers: list[str]
    extraction_mode: str


@dataclass(frozen=True)
class TextBlock:
    page_number: int
    section_path: tuple[str, ...]
    text: str
    searchable: bool


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    brand: str
    model: str
    year: int | None
    power_type: str
    document_type: str
    section_path: list[str]
    page_start: int
    page_end: int
    chunk_type: str
    safety_level: str
    searchable: bool
    text: str
    retrieval_text: str
    char_count: int
    source_url: str


def load_page_records(path: Path) -> list[PageRecord]:
    records: list[PageRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(PageRecord(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid page record at line {line_number}: {error}") from error
    if not records:
        raise ValueError(f"No page records found in {path}")
    return records


def _join_wrapped_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left[-1].isascii() and right[0].isascii() and left[-1].isalnum() and right[0].isalnum():
        return f"{left} {right}"
    return f"{left}{right}"


def _looks_like_subheading(line: str) -> bool:
    return bool(
        2 <= len(line) <= 18
        and line not in SAFETY_LEVELS
        and not line.startswith("*")
        and not LIST_ITEM_PATTERN.match(line)
        and not any(mark in line for mark in ("，", ",", "、", "：", ":"))
        and not line.endswith(TERMINAL_PUNCTUATION)
        and SUBHEADING_PATTERN.search(line)
    )


def reconstruct_paragraphs(text: str, headings: Sequence[str]) -> list[str]:
    """Join visual line wraps while preserving headings and list item boundaries."""
    heading_set = set(headings)
    paragraphs: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            paragraphs.append(current.strip())
        current = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line in heading_set:
            flush()
            continue
        if _looks_like_subheading(line):
            flush()
            paragraphs.append(line)
            continue
        if LIST_ITEM_PATTERN.match(line):
            flush()
            current = line
        else:
            current = _join_wrapped_text(current, line)
        if line.endswith(TERMINAL_PUNCTUATION):
            flush()

    flush()
    return paragraphs


def pages_to_blocks(pages: Sequence[PageRecord]) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    current_section = "未命名章节"
    current_subsection: str | None = None

    for page in pages:
        if page.headings:
            next_section = page.headings[0]
            if next_section != current_section:
                current_subsection = None
            current_section = next_section
        searchable = current_section != "目录"
        for paragraph in reconstruct_paragraphs(page.text, page.headings):
            if current_section != "缩略语" and _looks_like_subheading(paragraph):
                current_subsection = paragraph
                continue
            if len(paragraph) < 4 and not re.fullmatch(r"[A-Z0-9-]{2,12}", paragraph):
                continue
            section_path = (
                (current_section, current_subsection)
                if current_subsection
                else (current_section,)
            )
            blocks.append(
                TextBlock(
                    page_number=page.page_number,
                    section_path=section_path,
                    text=paragraph,
                    searchable=searchable,
                )
            )
    return blocks


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    sentences = [part for part in SENTENCE_BOUNDARY_PATTERN.split(text) if part]
    if len(sentences) <= 1:
        step = max_chars - overlap_chars
        return [text[start : start + max_chars] for start in range(0, len(text), step)]

    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            parts.append(current)
            overlap = current[-overlap_chars:] if overlap_chars else ""
            current = overlap + sentence
            if len(current) > max_chars:
                parts.extend(_split_long_text(current, max_chars, overlap_chars)[:-1])
                current = _split_long_text(current, max_chars, overlap_chars)[-1]
        else:
            current += sentence
    if current:
        parts.append(current)
    return parts


def _safety_level(text: str) -> str:
    for marker in ("危险", "警告", "注意", "提示"):
        for raw_line in text.splitlines():
            line = raw_line.strip()
            starts_with_marker = line.startswith(marker)
            if marker == "注意" and line.startswith(("注意力", "注意到")):
                starts_with_marker = False
            short_heading_ends_with_marker = len(line) <= 20 and line.endswith(marker)
            if starts_with_marker or short_heading_ends_with_marker:
                return SAFETY_LEVELS[marker]
    return "normal"


def _chunk_type(text: str, searchable: bool) -> str:
    if not searchable:
        return "table_of_contents"
    if _safety_level(text) != "normal":
        return "safety_notice"
    if len(LIST_ITEM_PATTERN.findall(text)) >= 2 or re.search(r"步骤|操作如下|按以下", text):
        return "procedure"
    return "text"


def _stable_chunk_id(
    document_id: str, page_start: int, page_end: int, section_path: Sequence[str], text: str
) -> str:
    payload = "|".join(
        [document_id, str(page_start), str(page_end), *section_path, text]
    ).encode("utf-8")
    suffix = hashlib.sha1(payload).hexdigest()[:12]
    return f"{document_id}-p{page_start:03d}-{page_end:03d}-{suffix}"


def _build_chunk(source: ManualSource, blocks: Sequence[TextBlock]) -> ChunkRecord:
    text = "\n".join(block.text for block in blocks).strip()
    page_start = min(block.page_number for block in blocks)
    page_end = max(block.page_number for block in blocks)
    section_path = list(blocks[0].section_path)
    searchable = all(block.searchable for block in blocks)
    section_label = " > ".join(section_path)
    classification_text = f"{section_label}\n{text}"
    retrieval_text = (
        f"品牌：{source.brand}\n车型：{source.model}\n动力类型：{source.power_type}\n"
        f"章节：{section_label}\n正文：{text}"
    )
    return ChunkRecord(
        chunk_id=_stable_chunk_id(source.id, page_start, page_end, section_path, text),
        document_id=source.id,
        brand=source.brand,
        model=source.model,
        year=source.year,
        power_type=source.power_type,
        document_type=source.document_type,
        section_path=section_path,
        page_start=page_start,
        page_end=page_end,
        chunk_type=_chunk_type(classification_text, searchable),
        safety_level=_safety_level(classification_text),
        searchable=searchable,
        text=text,
        retrieval_text=retrieval_text,
        char_count=len(text),
        source_url=source.source_url,
    )


def create_chunks(
    pages: Sequence[PageRecord],
    source: ManualSource,
    target_chars: int = 450,
    min_chars: int = 180,
    max_chars: int = 800,
    overlap_chars: int = 80,
) -> Iterator[ChunkRecord]:
    """Create section-aware chunks without crossing section or searchability boundaries."""
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")
    if not 0 < min_chars <= target_chars <= max_chars:
        raise ValueError("Expected 0 < min_chars <= target_chars <= max_chars")

    pending: list[TextBlock] = []

    def flush() -> Iterator[ChunkRecord]:
        nonlocal pending
        if pending:
            yield _build_chunk(source, pending)
            pending = []

    for block in pages_to_blocks(pages):
        if len(block.text) > max_chars:
            yield from flush()
            for part in _split_long_text(block.text, max_chars, overlap_chars):
                yield _build_chunk(
                    source,
                    [
                        TextBlock(
                            page_number=block.page_number,
                            section_path=block.section_path,
                            text=part,
                            searchable=block.searchable,
                        )
                    ],
                )
            continue

        boundary_changed = bool(pending) and (
            pending[0].section_path != block.section_path
            or pending[0].searchable != block.searchable
        )
        pending_length = sum(len(item.text) for item in pending) + max(0, len(pending) - 1)
        would_exceed = bool(pending) and pending_length + 1 + len(block.text) > max_chars
        target_reached = pending_length >= target_chars and len(block.text) >= min_chars

        if boundary_changed or would_exceed or target_reached:
            yield from flush()
        pending.append(block)

    yield from flush()


def write_chunks(records: Iterable[ChunkRecord], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count
