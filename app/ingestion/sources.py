from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManualSource:
    id: str
    brand: str
    model: str
    power_type: str
    document_type: str
    source_url: str
    local_filename: str
    year: int | None = None
    sha256: str | None = None
    status: str = "planned"


def load_sources(manifest_path: Path) -> list[ManualSource]:
    """Load and validate the manual source manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources: list[ManualSource] = []

    for item in payload.get("sources", []):
        sources.append(ManualSource(**item))

    if not sources:
        raise ValueError(f"No sources found in {manifest_path}")

    ids = [source.id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("Source IDs must be unique")

    return sources


def get_source(sources: list[ManualSource], source_id: str) -> ManualSource:
    """Return one source by ID."""
    for source in sources:
        if source.id == source_id:
            return source
    raise KeyError(f"Unknown source ID: {source_id}")

