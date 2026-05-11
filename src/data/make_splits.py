"""
src/data/make_splits.py  —  Stage 7: Train / Validation split.

WHY VALIDATION IS CRITICAL:
  - The Kaggle leaderboard only shows your score after submission.
    That is slow and expensive feedback. Local validation gives you
    instant feedback after every training run.
  - Without validation you cannot tell if the model is memorising
    the training data (overfitting) or genuinely learning.
  - Stratification by 'source' (dictation, archive, university, school)
    ensures both splits see the same variety of document styles.

STRATEGY:
  - Use 85% of train for local training, 15% for local validation.
  - Stratify by the 'source' field so no source is concentrated in
    one split.
  - Use a fixed random seed so results are reproducible.

OUTPUT:
  data/processed/train_split.jsonl   — local training set
  data/processed/val_split.jsonl     — local validation set

USAGE:
  python -m src.data.make_splits
  python -m src.data.make_splits --from-jsonl data/processed/train_raw.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.utils.paths import DATA_CFG, ensure_dir, get_path
from src.utils.jsonl import write_jsonl


def stratified_split(
    records: list[dict],
    val_fraction: float = 0.15,
    seed: int = 42,
    stratify_key: str = "source",
) -> tuple[list[dict], list[dict]]:
    """
    Split records into train and val lists, stratified by stratify_key.

    Stratification means we split each group independently so both
    train and val have similar proportions of each source type.

    Args:
        records:       list of metadata dicts
        val_fraction:  fraction going to validation (0.15 = 15%)
        seed:          random seed for reproducibility
        stratify_key:  field name to stratify on (default: "source")

    Returns:
        (train_records, val_records)
    """
    rng = random.Random(seed)

    # Group by the stratify key
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        key = rec.get(stratify_key, "unknown")
        groups[key].append(rec)

    train_records: list[dict] = []
    val_records:   list[dict] = []

    print(f"\n[make_splits] Stratifying by '{stratify_key}':")
    for key, group in sorted(groups.items()):
        rng.shuffle(group)
        n_val   = max(1, round(len(group) * val_fraction))
        n_train = len(group) - n_val
        val_records.extend(group[:n_val])
        train_records.extend(group[n_val:])
        print(f"  {key:<16}: {len(group):>5} total  →  "
              f"train={n_train}  val={n_val}")

    # Shuffle final lists so they are not grouped by source
    rng.shuffle(train_records)
    rng.shuffle(val_records)

    return train_records, val_records


def run(records: list[dict]) -> tuple[Path, Path]:
    """
    Execute the split and save both files.

    Returns:
        (train_path, val_path)
    """
    val_fraction = DATA_CFG.get("val_fraction", 0.15)
    seed         = DATA_CFG.get("random_seed", 42)

    print(f"[make_splits] Total records: {len(records):,}")
    print(f"[make_splits] val_fraction={val_fraction}  seed={seed}")

    train_recs, val_recs = stratified_split(
        records, val_fraction=val_fraction, seed=seed
    )

    ensure_dir(get_path("processed_root"))
    train_path = get_path("train_split")
    val_path   = get_path("val_split")

    n_train = write_jsonl(train_recs, train_path)
    n_val   = write_jsonl(val_recs,   val_path)

    print(f"\n[make_splits] ✅ Train split: {n_train:,} records → {train_path}")
    print(f"[make_splits] ✅ Val   split: {n_val:,}   records → {val_path}")

    # Sanity check: no image appears in both splits
    train_names = {r.get("file_name") for r in train_recs}
    val_names   = {r.get("file_name") for r in val_recs}
    overlap     = train_names & val_names
    if overlap:
        print(f"\n  ⚠️  WARNING: {len(overlap)} images appear in BOTH splits!")
        print(f"     Example: {list(overlap)[:3]}")
    else:
        print(f"  ✅ No overlap between train and val splits.")

    return train_path, val_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-jsonl", type=str, default=None,
                    help="Local metadata JSONL (else loads from HF)")
    ap.add_argument("--split", default="train",
                    choices=["train", "silver", "test"])
    args = ap.parse_args()

    if args.from_jsonl:
        from src.utils.jsonl import read_jsonl
        records = read_jsonl(args.from_jsonl)
    else:
        from src.data.load_dataset import load_split
        ds = load_split(args.split)
        records = list(ds)

    run(records)
    print("\n✅ Done. Next: python -m src.visualization.draw_bboxes")
