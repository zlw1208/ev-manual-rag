from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.sources import get_source, load_sources

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "sources.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(source_id: str, force: bool = False) -> Path:
    sources = load_sources(DEFAULT_MANIFEST)
    source = get_source(sources, source_id)
    output_path = DEFAULT_OUTPUT_DIR / source.local_filename

    if output_path.exists() and not force:
        digest = sha256_file(output_path)
        if source.sha256 and digest != source.sha256:
            raise ValueError(f"Checksum mismatch for existing file: {output_path}")
        print(f"Already downloaded: {output_path}")
        print(f"SHA256: {digest}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(
        source.source_url,
        headers={"User-Agent": "ev-manual-rag/0.1 (educational project)"},
    )

    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary_path.open("wb") as stream,
        ):
            while block := response.read(1024 * 1024):
                stream.write(block)

        if temporary_path.read_bytes()[:5] != b"%PDF-":
            raise ValueError("Downloaded content is not a PDF")

        digest = sha256_file(temporary_path)
        if source.sha256 and digest != source.sha256:
            raise ValueError(f"Checksum mismatch: expected {source.sha256}, got {digest}")

        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Downloaded: {output_path}")
    print(f"SHA256: {sha256_file(output_path)}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one official vehicle manual")
    parser.add_argument("source_id", help="ID from data/sources.json")
    parser.add_argument("--force", action="store_true", help="Replace an existing local file")
    args = parser.parse_args()

    try:
        download(args.source_id, force=args.force)
    except (KeyError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
