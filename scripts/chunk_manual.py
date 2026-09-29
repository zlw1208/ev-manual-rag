from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.chunker import create_chunks, load_page_records, write_chunks
from app.ingestion.sources import get_source, load_sources


def main() -> int:
    parser = argparse.ArgumentParser(description="Create retrieval chunks from parsed manual pages")
    parser.add_argument("input", type=Path, help="Page-level JSONL produced by parse_manual.py")
    parser.add_argument("--source-id", required=True, help="ID from data/sources.json")
    parser.add_argument("--output", type=Path, help="Output chunk JSONL path")
    parser.add_argument("--target-chars", type=int, default=450)
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap-chars", type=int, default=80)
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        parser.error(f"Input JSONL does not exist: {input_path}")

    source = get_source(load_sources(PROJECT_ROOT / "data" / "sources.json"), args.source_id)
    output_path = args.output or (
        PROJECT_ROOT / "data" / "processed" / f"{source.id}.chunks.jsonl"
    )
    chunks = create_chunks(
        load_page_records(input_path),
        source,
        target_chars=args.target_chars,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    count = write_chunks(chunks, output_path.resolve())
    print(f"Created {count} chunks -> {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

