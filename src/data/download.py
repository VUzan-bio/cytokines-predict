"""Streaming utilities to pull cytokine datasets from GEO/SRA without exhausting memory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
import requests
from pysradb.sraweb import SRAweb
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fetch_sra_metadata(accession: str, output_dir: str | Path) -> Path:
    """Download SRA metadata table for a GEO accession."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sra = SRAweb()
    logger.info("Querying SRA metadata for %s", accession)
    metadata = sra.sra_metadata(accession, detailed=True)
    if metadata is None or metadata.empty:
        raise ValueError(f"No metadata returned for accession {accession}")
    out_path = output_dir / f"{accession}_sra_metadata.csv"
    metadata.to_csv(out_path, index=False)
    return out_path


def _iter_download_urls(metadata: pd.DataFrame) -> Iterable[Tuple[str, str]]:
    """Yield (run_accession, url) tuples from pysradb metadata."""
    url_candidates = [
        "download_url",
        "run_download_url",
        "fastq_ftp",
        "fastq_http",
    ]
    for _, row in metadata.iterrows():
        url = None
        for col in url_candidates:
            val = row.get(col)
            if isinstance(val, str) and val:
                url = val.split(";")[0]
                break
        run = row.get("run_accession") or row.get("run")
        if url and run:
            yield str(run), url


def stream_runs(metadata_path: str | Path, output_dir: str | Path, *, chunk_size: int = 262_144, timeout: int = 30, max_records: Optional[int] = None) -> list[Path]:
    """Stream FASTQ/BAM files specified in an SRA metadata CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    files: list[Path] = []
    for idx, (run, url) in enumerate(_iter_download_urls(metadata)):
        if max_records is not None and idx >= max_records:
            break
        dest = output_dir / f"{run}.fastq.gz"
        if dest.exists():
            files.append(dest)
            continue
        logger.info("Streaming %s -> %s", url, dest)
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(dest, "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)
        files.append(dest)
    return files


def download_cytokine_accession(accession: str, raw_dir: str | Path, *, chunk_size: int = 262_144, timeout: int = 30, max_records: Optional[int] = None) -> tuple[Path, list[Path]]:
    """Convenience wrapper to fetch metadata and stream runs."""
    metadata_path = fetch_sra_metadata(accession, raw_dir)
    files = stream_runs(metadata_path, raw_dir, chunk_size=chunk_size, timeout=timeout, max_records=max_records)
    return metadata_path, files
