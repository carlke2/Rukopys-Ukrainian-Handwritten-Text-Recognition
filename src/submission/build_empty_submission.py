"""
src/submission/build_empty_submission.py  —  Stage 11: Empty baseline CSV.

WHY AN EMPTY SUBMISSION FIRST:
  Before training any model, confirm that:
    1. You can read the list of test images.
    2. You can produce a correctly-formatted CSV.
    3. The validator accepts it.
    4. You understand the full submission loop end-to-end.

  An empty submission (all regions = []) scores 0 on the leaderboard
  but proves your pipeline works. Every model submission later follows
  the same format — only the region lists change.

SUBMISSION FORMAT:
  image,regions
  doc_001.jpg,"[]"
  doc_002.jpg,"[]"
  ...

USAGE:
  python -m src.submission.build_empty_submission
  python -m src.submission.build_empty_submission --out data/submissions/empty_v1.csv
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from src.utils.paths import ensure_dir, get_path


def get_test_filenames() -> list[str]:
    """
    Collect test image filenames from available sources.

    Tries in order:
      1. Local test metadata JSONL (if downloaded)
      2. sample_submission.csv (always available from Kaggle)
      3. Local test images directory

    Returns:
        list of filenames (e.g. ["doc_001.jpg", "doc_002.jpg", ...])
    """
    # Source 1: local test metadata
    meta_path = get_path("raw_test_meta")
    if meta_path.exists():
        from src.utils.jsonl import read_jsonl
        records = read_jsonl(meta_path)
        names = [r["file_name"] for r in records if "file_name" in r]
        if names:
            print(f"  Source: test metadata JSONL ({len(names):,} images)")
            return names

    # Source 2: sample_submission.csv
    sample_path = get_path("sample_submission")
    if sample_path.exists():
        names = []
        with open(sample_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("image", "").strip()
                if fname:
                    names.append(fname)
        if names:
            print(f"  Source: sample_submission.csv ({len(names):,} images)")
            return names

    # Source 3: test images directory
    img_dir = get_path("raw_test_images")
    if img_dir.exists():
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        names = sorted(
            f.name for f in img_dir.iterdir()
            if f.suffix.lower() in exts
        )
        if names:
            print(f"  Source: test images directory ({len(names):,} images)")
            return names

    print("  ⚠️  No test image source found. Creating a demo submission.")
    print("     Download test data with: python -m src.data.load_dataset --splits test")
    # Return a small demo list so the script still produces a valid CSV
    return [f"demo_{i:04d}.jpg" for i in range(5)]


def build_empty_submission(out_path: Path | None = None) -> Path:
    """
    Build a submission CSV with empty region lists for every test image.

    Args:
        out_path: where to save the CSV (default: data/submissions/empty_<timestamp>.csv)

    Returns:
        path to saved CSV
    """
    if out_path is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_dir  = ensure_dir(get_path("submissions_dir"))
        out_path = sub_dir / f"empty_{ts}.csv"

    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    print("[build_empty_submission] Collecting test image names …")
    filenames = get_test_filenames()

    print(f"[build_empty_submission] Writing {len(filenames):,} rows → {out_path}")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "regions"])
        for fname in filenames:
            writer.writerow([fname, json.dumps([])])

    print(f"[build_empty_submission] ✅ Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV path (default: auto-timestamped)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else None
    csv_path = build_empty_submission(out_path)

    # Auto-validate the result
    print("\n[build_empty_submission] Validating output …")
    from src.submission.validate_submission import validate_submission
    is_valid, errors = validate_submission(csv_path)
    if is_valid:
        print("✅ Validation passed — submission format is correct.")
    else:
        print(f"❌ Validation failed ({len(errors)} errors):")
        for e in errors:
            print(f"  {e}")

    print(f"\n✅ Empty baseline submission ready: {csv_path}")
    print("   Next: python -m src.evaluation.local_metric --help")
