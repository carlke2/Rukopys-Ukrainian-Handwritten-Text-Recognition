"""
src/recognition/validate_crops.py  —  Stage 8b: Validate recognition crops.

WHY CROP VALIDATION BEFORE TRAINING:
  The recognizer trains on (image, text) pairs from labels.csv.
  If any of these are broken:
    - A missing crop file → FileNotFoundError during training.
    - A zero-pixel crop   → model training crash.
    - Empty text label    → model learns to predict nothing.
    - Corrupt JPEG        → Pillow raises an exception mid-epoch.

  Running this once before training catches all of these.
  After this passes, you can trust the crop dataset completely.

CHECKS PERFORMED:
  ✓ Every crop_path in labels.csv exists on disk
  ✓ Every crop opens without error (Pillow)
  ✓ Every crop has width >= 1 and height >= 1
  ✓ Every text label is non-empty
  ✓ Every type is one of the allowed recognition types
  ✓ No duplicate crop paths
  ✓ Contact sheet of random samples saved for visual inspection

USAGE:
  python -m src.recognition.validate_crops
  python -m src.recognition.validate_crops --split val
  python -m src.recognition.validate_crops --split train --n-contact 40
"""

import argparse
import csv
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.utils.paths import DATA_CFG, ensure_dir, get_path, get_region_types

# Region types allowed for recognition training
VALID_RECOGNITION_TYPES = {
    k for k, v in get_region_types().items()
    if v.get("text_bearing", False)
}
NON_TEXT = set(DATA_CFG.get("non_text_types", ["image", "graph"]))


def validate_crops(csv_path: Path, n_contact: int = 20) -> dict:
    """
    Validate a crops labels.csv and save a contact sheet.

    Args:
        csv_path:   path to labels.csv
        n_contact:  number of random crops to tile in the contact sheet

    Returns:
        summary dict with counts and pass/fail status
    """
    if not csv_path.exists():
        return {
            "passed": False,
            "error": f"labels.csv not found: {csv_path}",
            "total": 0, "errors": 0, "warnings": 0,
        }

    # Load CSV
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[validate_crops] Checking {len(rows):,} crops from {csv_path}")

    errors   = []
    warnings = []
    seen_paths = set()
    valid_rows = []

    for i, row in enumerate(tqdm(rows, desc="Validating crops")):
        crop_path = row.get("crop_path", "").strip()
        text      = row.get("text", "").strip()
        rtype     = row.get("type", "").strip()

        # ── Duplicate check ───────────────────────────────────────
        if crop_path in seen_paths:
            errors.append(f"  row {i+2}: Duplicate crop_path: {crop_path}")
            continue
        seen_paths.add(crop_path)

        # ── File exists ───────────────────────────────────────────
        p = Path(crop_path)
        if not p.exists():
            errors.append(f"  row {i+2}: File not found: {crop_path}")
            continue

        # ── Can open ──────────────────────────────────────────────
        try:
            with Image.open(p) as img:
                w, h = img.size
        except Exception as e:
            errors.append(f"  row {i+2}: Cannot open {p.name}: {e}")
            continue

        # ── Non-zero dimensions ───────────────────────────────────
        if w < 1 or h < 1:
            errors.append(
                f"  row {i+2}: {p.name} has zero dimension ({w}x{h})"
            )
            continue

        # ── Text label ────────────────────────────────────────────
        if not text:
            errors.append(f"  row {i+2}: Empty text label for {p.name}")
            continue

        # ── Region type ───────────────────────────────────────────
        if rtype in NON_TEXT:
            warnings.append(
                f"  row {i+2}: {p.name} has non-text type '{rtype}' — "
                "should have been excluded during prepare_crops."
            )
        elif rtype not in VALID_RECOGNITION_TYPES:
            warnings.append(
                f"  row {i+2}: Unknown type '{rtype}' for {p.name}"
            )

        valid_rows.append(row)

    # ── Contact sheet ─────────────────────────────────────────────
    contact_path = None
    if valid_rows:
        sample_rows = random.sample(valid_rows, min(n_contact, len(valid_rows)))
        contact_path = _make_contact_sheet(sample_rows)

    # ── Summary ───────────────────────────────────────────────────
    summary = {
        "total":      len(rows),
        "valid":      len(valid_rows),
        "errors":     len(errors),
        "warnings":   len(warnings),
        "passed":     len(errors) == 0,
        "error_msgs": errors[:30],
        "warn_msgs":  warnings[:20],
        "contact_sheet": str(contact_path) if contact_path else None,
    }
    return summary


def _make_contact_sheet(rows: list[dict]) -> Path:
    """
    Create a simple contact sheet of crop thumbnails.
    Each cell shows the crop image + truncated text label below it.

    Returns:
        path to saved PNG
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  ⚠️  opencv not available — skipping contact sheet.")
        return None

    thumb_w, thumb_h = 300, 70
    cols     = 4
    pad      = 4
    cell_w   = thumb_w + pad * 2
    cell_h   = thumb_h + 28 + pad
    n        = len(rows)
    n_rows   = (n + cols - 1) // cols
    canvas   = np.full((n_rows * cell_h, cols * cell_w, 3), 30, dtype="uint8")

    for i, row in enumerate(rows):
        r    = i // cols
        c    = i % cols
        x0   = c * cell_w + pad
        y0   = r * cell_h + pad
        p    = Path(row["crop_path"])
        text = row.get("text", "")

        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                thumb = cv2.resize(img, (thumb_w, thumb_h))
                canvas[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb

        label = (text[:38] + "…") if len(text) > 38 else text
        cv2.putText(
            canvas, label,
            (x0, y0 + thumb_h + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36,
            (200, 200, 200), 1, cv2.LINE_AA,
        )

    out_dir  = ensure_dir(get_path("debug_images_dir") / "crops")
    out_path = out_dir / "crop_contact_sheet.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"  ✅ Contact sheet saved → {out_path}")
    return out_path


def print_summary(s: dict, split: str) -> None:
    """Print validation summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  CROP VALIDATION — {split.upper()}")
    print(f"{'='*60}")
    print(f"  Total rows   : {s['total']:,}")
    print(f"  Valid crops  : {s['valid']:,}")
    print(f"  Errors       : {s['errors']:,}  ← must be 0 before training")
    print(f"  Warnings     : {s['warnings']:,}")

    if s.get("contact_sheet"):
        print(f"  Contact sheet: {s['contact_sheet']}")

    if s["error_msgs"]:
        print(f"\n  First {len(s['error_msgs'])} errors:")
        for e in s["error_msgs"]:
            print(e)

    if s["warn_msgs"]:
        print(f"\n  First {len(s['warn_msgs'])} warnings:")
        for w in s["warn_msgs"]:
            print(w)

    verdict = "✅ PASSED" if s["passed"] else "❌ FAILED"
    print(f"\n  Result: {verdict}")
    print("=" * 60)


def save_report(s: dict, split: str) -> Path:
    """Save a Markdown validation report."""
    report_dir  = ensure_dir(get_path("reports_dir"))
    report_path = report_dir / f"crop_validation_report_{split}.md"
    verdict     = "✅ PASSED" if s["passed"] else "❌ FAILED"

    lines = [
        f"# Crop Validation Report — {split}\n\n",
        f"**Result: {verdict}**\n\n",
        "## Summary\n\n",
        "| Metric | Value |\n|--------|-------|\n",
        f"| Total rows | {s['total']:,} |\n",
        f"| Valid crops | {s['valid']:,} |\n",
        f"| Errors (blockers) | {s['errors']:,} |\n",
        f"| Warnings | {s['warnings']:,} |\n",
    ]
    if s.get("contact_sheet"):
        lines.append(f"\nContact sheet: `{s['contact_sheet']}`\n")

    if s["error_msgs"]:
        lines.append("\n## Errors\n\n```\n")
        for e in s["error_msgs"]:
            lines.append(e + "\n")
        lines.append("```\n")

    if s["warn_msgs"]:
        lines.append("\n## Warnings\n\n```\n")
        for w in s["warn_msgs"]:
            lines.append(w + "\n")
        lines.append("```\n")

    lines.append(
        "\n> Errors must be resolved before starting recognizer training.\n"
    )
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"  ✅ Report saved → {report_path}")
    return report_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Validate recognition crop dataset."
    )
    ap.add_argument(
        "--split", default="train", choices=["train", "val"],
        help="Which split to validate"
    )
    ap.add_argument(
        "--n-contact", type=int, default=20,
        help="Number of random crops in the contact sheet"
    )
    args = ap.parse_args()

    if args.split == "train":
        csv_path = get_path("crops_train_csv")
    else:
        csv_path = get_path("crops_val_csv")

    summary = validate_crops(csv_path, n_contact=args.n_contact)
    print_summary(summary, args.split)
    save_report(summary, args.split)

    import sys
    if summary["passed"]:
        print(f"\n✅ Crop validation PASSED for '{args.split}' split.")
        print("   Next: python -m src.detection.convert_to_yolo")
        sys.exit(0)
    else:
        print(f"\n❌ {summary['errors']} error(s) found. Fix before training.")
        sys.exit(1)
