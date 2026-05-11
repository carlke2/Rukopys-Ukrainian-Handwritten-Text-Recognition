"""
src/data/load_dataset.py
────────────────────────
Stage 4 — Load the RUKOPYS dataset from Hugging Face.

WHY THIS FILE EXISTS:
  The dataset lives on Hugging Face Hub. This module gives you two
  loading modes:
    1. Full download  — caches the whole dataset locally (recommended
                        when you have disk space and a fast connection).
    2. Streaming      — reads records one-by-one without downloading
                        (useful on machines with limited storage).

  We also provide a function to save raw metadata as local JSONL files
  so subsequent scripts never need internet access again.

USAGE:
  # Load full train split
  from src.data.load_dataset import load_split, save_split_as_jsonl
  ds = load_split("train")
  save_split_as_jsonl(ds, "train")

  # Stream (memory-efficient)
  from src.data.load_dataset import stream_split
  for sample in stream_split("train"):
      print(sample["file_name"])

RUN DIRECTLY:
  python -m src.data.load_dataset
"""

import json
import os
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from src.utils.paths import DATA_CFG, PROJECT_ROOT, ensure_dir, get_path

# ── Dataset identifier on Hugging Face ───────────────────────────
HF_DATASET_ID = DATA_CFG["hf_dataset_id"]


def load_split(split: str, cache_dir: str | None = None):
    """
    Download and cache a full dataset split from Hugging Face.

    This uses the `datasets` library, which automatically caches
    the data in ~/.cache/huggingface/datasets/ (or cache_dir).
    After the first download, subsequent calls are instant.

    Args:
        split:     "train", "silver", or "test"
        cache_dir: optional custom cache directory

    Returns:
        HuggingFace Dataset object (indexable, filterable)

    Example:
        ds = load_split("train")
        print(ds[0]["file_name"])
        print(len(ds))
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "The 'datasets' package is not installed.\n"
            "Run: pip install datasets"
        )

    print(f"[load_split] Loading '{split}' split from {HF_DATASET_ID} …")
    print("  (First run downloads data; subsequent runs use local cache.)")

    ds = hf_load(
        HF_DATASET_ID,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    print(f"[load_split] ✅ Loaded {len(ds):,} samples from '{split}'")
    return ds


def stream_split(split: str) -> Iterator[dict]:
    """
    Stream a dataset split one record at a time WITHOUT downloading.

    Use this on machines with limited disk space.
    Note: streaming datasets cannot be indexed with ds[i].

    Args:
        split: "train", "silver", or "test"

    Yields:
        one sample dict per document page
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(f"[stream_split] Streaming '{split}' from {HF_DATASET_ID} …")
    ds = hf_load(
        HF_DATASET_ID,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    yield from ds


def save_split_as_jsonl(ds_or_split, split_name: str) -> Path:
    """
    Save a dataset split to a local JSONL file so future scripts
    can work offline (without re-loading from Hugging Face).

    The image column is NOT saved (images are large binary blobs).
    Only metadata fields are saved: file_name, image_width,
    image_height, source, annotation_source, regions.

    Args:
        ds_or_split: HuggingFace Dataset OR a string like "train"
                     (if string, we call load_split() for you)
        split_name:  "train", "silver", or "test"

    Returns:
        Path to the saved JSONL file
    """
    if isinstance(ds_or_split, str):
        ds_or_split = load_split(ds_or_split)

    out_dir = ensure_dir(get_path("processed_root"))
    out_path = out_dir / f"{split_name}_raw.jsonl"

    print(f"[save_split_as_jsonl] Writing metadata → {out_path}")

    # Columns that exist in all splits
    meta_cols = [
        "file_name", "image_width", "image_height",
        "source", "annotation_source", "regions",
    ]

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in tqdm(ds_or_split, desc=f"Saving {split_name} metadata"):
            record = {}
            for col in meta_cols:
                if col in sample:
                    val = sample[col]
                    # regions is already a list-of-dicts from HF datasets
                    record[col] = val
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"[save_split_as_jsonl] ✅ Saved {count:,} records → {out_path}")
    return out_path


def load_from_jsonl(split_name: str) -> list[dict]:
    """
    Load metadata that was previously saved with save_split_as_jsonl().
    Much faster than re-loading from HuggingFace for offline work.

    Args:
        split_name: "train", "silver", or "test"

    Returns:
        list of metadata dicts
    """
    from src.utils.jsonl import read_jsonl

    path = get_path("processed_root") / f"{split_name}_raw.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Local metadata not found: {path}\n"
            f"Run save_split_as_jsonl('{split_name}') first."
        )
    records = read_jsonl(path)
    print(f"[load_from_jsonl] ✅ Loaded {len(records):,} records from {path}")
    return records


# ── CLI entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and cache RUKOPYS dataset splits."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "silver", "test"],
        help="Which splits to download (default: train test)",
    )
    parser.add_argument(
        "--save-jsonl",
        action="store_true",
        help="Also save metadata as local JSONL files",
    )
    args = parser.parse_args()

    for split in args.splits:
        ds = load_split(split)
        print(f"  Split '{split}': {len(ds):,} samples")
        print(f"  Columns: {ds.column_names}")
        if args.save_jsonl:
            save_split_as_jsonl(ds, split)

    print("\n✅ Done. Dataset is cached locally.")
    print("   Next step: python -m src.data.inspect_dataset")
