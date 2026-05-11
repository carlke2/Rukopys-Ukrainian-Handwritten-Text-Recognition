"""
src/utils/jsonl.py
──────────────────
Helpers for reading and writing JSONL (JSON Lines) files.

WHY JSONL:
  The RUKOPYS dataset uses metadata.jsonl — one JSON object per line.
  It is streaming-friendly: you can read line by line without loading
  the entire file into memory, which matters when you have thousands
  of annotated pages.

USAGE:
  from src.utils.jsonl import read_jsonl, write_jsonl, stream_jsonl

  records = read_jsonl("data/processed/train_split.jsonl")
  write_jsonl(records, "data/processed/train_split.jsonl")

  # Memory-efficient streaming:
  for record in stream_jsonl("data/raw/train/metadata.jsonl"):
      process(record)
"""

import json
from pathlib import Path
from typing import Any, Generator, Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    """
    Read an entire JSONL file into a list of dicts.

    Use this when the file fits comfortably in memory (< a few GB).
    For very large files, use stream_jsonl() instead.

    Args:
        path: path to the .jsonl file

    Returns:
        list of dicts, one per line
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e
    return records


def stream_jsonl(path: str | Path) -> Generator[dict, None, None]:
    """
    Stream a JSONL file one record at a time (memory-efficient).

    Use this when iterating over large files without loading everything.

    Args:
        path: path to the .jsonl file

    Yields:
        one dict per valid non-empty line
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e


def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    """
    Write an iterable of dicts to a JSONL file (overwrites existing).

    Args:
        records: iterable of dicts
        path:    destination path

    Returns:
        number of records written
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(record: dict, path: str | Path) -> None:
    """
    Append a single record to an existing (or new) JSONL file.
    Useful for streaming writes during long processing jobs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_lines(path: str | Path) -> int:
    """
    Count non-empty lines in a JSONL file without loading it fully.
    Fast way to know how many records exist.
    """
    path = Path(path)
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os

    sample = [
        {"id": 1, "text": "Доброго ранку"},
        {"id": 2, "text": "Привіт світ"},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        tmp_path = tmp.name

    write_jsonl(sample, tmp_path)
    loaded = read_jsonl(tmp_path)
    assert loaded == sample, "Round-trip failed!"
    print(f"✅ JSONL round-trip OK: {loaded}")
    os.unlink(tmp_path)
