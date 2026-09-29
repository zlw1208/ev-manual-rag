from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.manual_parser import parse_pdf, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a vehicle manual into page-level JSONL")
    parser.add_argument("input", type=Path, help="Path to the source PDF")
    parser.add_argument("--document-id", required=True, help="Stable document identifier")
    parser.add_argument("--output", type=Path, help="Output JSONL path")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output or (
        PROJECT_ROOT / "data" / "processed" / f"{args.document_id}.pages.jsonl"
    )

    if not input_path.is_file():
        parser.error(f"Input PDF does not exist: {input_path}")

    count = write_jsonl(parse_pdf(input_path, args.document_id), output_path.resolve())
    print(f"Parsed {count} pages -> {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
