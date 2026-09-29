import json
from pathlib import Path

import pytest

from app.ingestion.manual_parser import normalize_text
from app.ingestion.sources import get_source, load_sources


def test_normalize_text_removes_blank_lines_and_simple_footer() -> None:
    text = "  充电说明  \n\n动力电池使用注意事项\n充电  156\n"

    assert normalize_text(text) == "充电说明\n动力电池使用注意事项"


def test_load_sources_and_lookup(tmp_path: Path) -> None:
    manifest = {
        "sources": [
            {
                "id": "demo",
                "brand": "AITO",
                "model": "M5",
                "power_type": "纯电",
                "document_type": "使用说明书",
                "source_url": "https://example.com/manual.pdf",
                "local_filename": "demo.pdf"
            }
        ]
    }
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    source = get_source(load_sources(manifest_path), "demo")

    assert source.model == "M5"
    assert source.local_filename == "demo.pdf"


def test_get_source_rejects_unknown_id(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "known",
                        "brand": "AITO",
                        "model": "M5",
                        "power_type": "纯电",
                        "document_type": "使用说明书",
                        "source_url": "https://example.com/manual.pdf",
                        "local_filename": "known.pdf"
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        get_source(load_sources(manifest_path), "missing")

