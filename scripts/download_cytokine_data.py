"""Helper script to stream GEO/SRA data to the raw folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.download import download_cytokine_accession


def parse_args():
    parser = argparse.ArgumentParser(description="Stream cytokine datasets from GEO/SRA.")
    parser.add_argument("--accession", required=True, help="GEO accession (e.g., GSEXXXXXX).")
    parser.add_argument("--raw_dir", default="data/raw", help="Directory to write streamed files.")
    parser.add_argument("--chunk_size", type=int, default=262144, help="Chunk size in bytes.")
    parser.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds.")
    parser.add_argument("--max_records", type=int, default=None, help="Limit number of runs (for testing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_path, files = download_cytokine_accession(
        accession=args.accession,
        raw_dir=raw_dir,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        max_records=args.max_records,
    )
    print(f"Metadata: {metadata_path}")
    print(f"Downloaded {len(files)} run files to {raw_dir}")


if __name__ == "__main__":
    main()
