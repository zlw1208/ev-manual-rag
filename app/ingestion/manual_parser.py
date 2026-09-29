from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SAFETY_MARKERS = ("危险", "警告", "注意", "提示")
FOOTER_PATTERN = re.compile(r"^(?:[\u4e00-\u9fffA-Za-z· ]{1,20}\s+)?\d{1,3}$")


@dataclass(frozen=True)
class ParsedPage:
    document_id: str
    page_number: int
    headings: list[str]
    text: str
    safety_markers: list[str]
    extraction_mode: str


def normalize_text(text: str) -> str:
    """Normalize extracted PDF text while retaining paragraph boundaries."""
    cleaned_lines: list[str] = []
    for raw_line in text.replace("\x00", "").splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line or FOOTER_PATTERN.fullmatch(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _is_two_column(page: Any) -> bool:
    """Detect a central gutter without depending on a PDF outline or tags."""
    words = page.extract_words() or []
    if len(words) < 12:
        return False

    midpoint = float(page.width) / 2
    margin = float(page.width) * 0.035
    content_words = [
        word
        for word in words
        if float(word["top"]) > 25 and float(word["bottom"]) < float(page.height) - 25
    ]
    crossing = [
        word
        for word in content_words
        if float(word["x0"]) < midpoint - margin and float(word["x1"]) > midpoint + margin
    ]
    left = [word for word in content_words if float(word["x1"]) <= midpoint + margin]
    right = [word for word in content_words if float(word["x0"]) >= midpoint - margin]

    return len(crossing) < 3 and len(left) >= 5 and len(right) >= 5


def extract_page_text(page: Any) -> tuple[str, str]:
    """Extract text in a stable reading order for one- or two-column pages."""
    if not _is_two_column(page):
        return normalize_text(page.extract_text(x_tolerance=2, y_tolerance=3) or ""), "single"

    midpoint = float(page.width) / 2
    left = page.crop((0, 0, midpoint, float(page.height)))
    right = page.crop((midpoint, 0, float(page.width), float(page.height)))
    left_text = normalize_text(left.extract_text(x_tolerance=2, y_tolerance=3) or "")
    right_text = normalize_text(right.extract_text(x_tolerance=2, y_tolerance=3) or "")
    return "\n".join(part for part in (left_text, right_text) if part), "two-column"


def extract_headings(page: Any) -> list[str]:
    """Infer visible headings from font size and line position."""
    chars = [char for char in page.chars if char.get("text", "").strip()]
    if not chars:
        return []

    lines: dict[int, list[dict[str, Any]]] = {}
    for char in chars:
        if float(char["top"]) > float(page.height) - 30:
            continue
        line_key = round(float(char["top"]) / 2)
        lines.setdefault(line_key, []).append(char)

    candidates: list[tuple[float, float, str]] = []
    for line_chars in lines.values():
        ordered = sorted(line_chars, key=lambda char: float(char["x0"]))
        segments: list[list[dict[str, Any]]] = [[]]
        for char in ordered:
            if segments[-1]:
                previous = segments[-1][-1]
                gap = float(char["x0"]) - float(previous["x1"])
                if gap > 18:
                    segments.append([])
            segments[-1].append(char)

        for segment in segments:
            text = re.sub(
                r"\s+", "", "".join(str(char.get("text", "")) for char in segment)
            )
            max_size = max(float(char.get("size", 0)) for char in segment)
            top = min(float(char["top"]) for char in segment)
            if not text or len(text) > 32 or max_size < 11:
                continue
            candidates.append((top, -max_size, text))

    headings: list[str] = []
    for _, _, text in sorted(candidates):
        if text not in headings:
            headings.append(text)
    return headings


def parse_pdf(pdf_path: Path, document_id: str) -> Iterable[ParsedPage]:
    """Yield page-level records suitable for later chunking and indexing."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text, extraction_mode = extract_page_text(page)
            headings = extract_headings(page)
            missing_headings = [heading for heading in headings if heading not in text[:120]]
            if missing_headings:
                text = "\n".join([*missing_headings, text])
            yield ParsedPage(
                document_id=document_id,
                page_number=index,
                headings=headings,
                text=text,
                safety_markers=[marker for marker in SAFETY_MARKERS if marker in text],
                extraction_mode=extraction_mode,
            )


def write_jsonl(records: Iterable[ParsedPage], output_path: Path) -> int:
    """Write parsed pages as UTF-8 JSON Lines and return the record count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count
