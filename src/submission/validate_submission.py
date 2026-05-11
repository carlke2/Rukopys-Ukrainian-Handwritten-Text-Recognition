"""
src/submission/validate_submission.py  —  Stage 10: Validate CSV format.

WHY VALIDATION BEFORE SUBMITTING:
  Kaggle will silently reject or error on malformed submissions.
  Running this locally catches problems before you waste a submission slot.

  Checks performed:
    1. CSV has exactly the columns: image, regions
    2. Every 'regions' value is valid JSON
    3. Each region has: bbox (list of 4 numbers), type (valid string), text (string)
    4. bbox satisfies x2 > x1 and y2 > y1
    5. type is one of the 7 allowed region types
    6. image/graph regions may have empty text; others must have string text
    7. (Optional) All test image filenames are present

USAGE:
  python -m src.submission.validate_submission --csv data/submissions/empty.csv
  python -m src.submission.validate_submission --csv my_sub.csv --check-completeness
"""

import argparse
import csv
import json
from pathlib import Path

from src.utils.paths import get_path, get_region_types, DATA_CFG

VALID_TYPES  = set(get_region_types().keys())
NON_TEXT     = set(DATA_CFG.get("non_text_types", ["image", "graph"]))


def validate_region(region: dict, row_idx: int, reg_idx: int) -> list[str]:
    """
    Validate one region dict. Returns list of error strings (empty = valid).
    """
    errors = []
    prefix = f"  row {row_idx}, region {reg_idx}"

    # Must have bbox, type, text
    for key in ("bbox", "type", "text"):
        if key not in region:
            errors.append(f"{prefix}: missing key '{key}'")

    if errors:
        return errors

    # Validate bbox
    bbox = region["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        errors.append(f"{prefix}: bbox must be a list of 4 numbers, got {bbox}")
    else:
        x1, y1, x2, y2 = bbox
        for v in bbox:
            if not isinstance(v, (int, float)):
                errors.append(f"{prefix}: bbox values must be numbers, got {v}")
                break
        else:
            if x2 <= x1:
                errors.append(f"{prefix}: bbox x2={x2} must be > x1={x1}")
            if y2 <= y1:
                errors.append(f"{prefix}: bbox y2={y2} must be > y1={y1}")

    # Validate type
    rtype = region["type"]
    if rtype not in VALID_TYPES:
        errors.append(
            f"{prefix}: unknown type '{rtype}'. "
            f"Valid: {sorted(VALID_TYPES)}"
        )

    # Validate text
    text = region["text"]
    if not isinstance(text, str):
        errors.append(f"{prefix}: text must be a string, got {type(text)}")

    return errors


def validate_submission(
    csv_path: str | Path,
    test_images: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate a full submission CSV.

    Args:
        csv_path:    path to the submission CSV
        test_images: optional set of expected image filenames;
                     if given, checks all test images are present

    Returns:
        (is_valid: bool, errors: list of error strings)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False, [f"File not found: {csv_path}"]

    errors = []
    seen_images = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        try:
            reader = csv.DictReader(f)
        except Exception as e:
            return False, [f"Cannot parse CSV: {e}"]

        # Check required columns
        required = {"image", "regions"}
        if reader.fieldnames is None:
            return False, ["CSV appears empty or has no header row."]

        missing_cols = required - set(reader.fieldnames)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
            return False, errors

        for row_idx, row in enumerate(reader, start=2):  # row 1 = header
            fname = row.get("image", "").strip()
            if not fname:
                errors.append(f"  row {row_idx}: empty image filename")
                continue

            seen_images.add(fname)

            # Parse regions JSON
            raw = row.get("regions", "").strip()
            try:
                regions = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(
                    f"  row {row_idx} ({fname}): invalid JSON in regions: {e}"
                )
                continue

            if not isinstance(regions, list):
                errors.append(
                    f"  row {row_idx} ({fname}): regions must be a JSON array"
                )
                continue

            # Validate each region
            for reg_idx, region in enumerate(regions):
                if not isinstance(region, dict):
                    errors.append(
                        f"  row {row_idx}, region {reg_idx}: "
                        f"must be a dict, got {type(region)}"
                    )
                    continue
                errs = validate_region(region, row_idx, reg_idx)
                errors.extend(errs)

    # Check completeness
    if test_images is not None:
        missing = test_images - seen_images
        extra   = seen_images - test_images
        if missing:
            errors.append(
                f"  Missing {len(missing)} test images from submission. "
                f"Examples: {list(missing)[:5]}"
            )
        if extra:
            errors.append(
                f"  {len(extra)} extra images not in test set. "
                f"Examples: {list(extra)[:5]}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to submission CSV")
    ap.add_argument("--check-completeness", action="store_true",
                    help="Also verify all test images are present")
    args = ap.parse_args()

    test_images = None
    if args.check_completeness:
        meta_path = get_path("raw_test_meta")
        if meta_path.exists():
            from src.utils.jsonl import read_jsonl
            test_records = read_jsonl(meta_path)
            test_images = {r["file_name"] for r in test_records}
            print(f"  Loaded {len(test_images):,} test image names for completeness check.")
        else:
            print(f"  ⚠️  Test metadata not found at {meta_path}. Skipping completeness check.")

    is_valid, errors = validate_submission(args.csv, test_images=test_images)

    if is_valid:
        print(f"\n✅ Submission {args.csv} is VALID.")
    else:
        print(f"\n❌ Submission {args.csv} has {len(errors)} error(s):")
        for e in errors[:30]:
            print(e)
        if len(errors) > 30:
            print(f"  … and {len(errors)-30} more errors.")
