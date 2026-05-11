"""
src/data/bbox_validation.py  —  Stage 7b: Validate all bounding boxes.

WHY BBOX VALIDATION IS CRITICAL:
  A bad bounding box causes silent failures downstream:
    - A crop with x2 <= x1 produces a zero-pixel image → crash.
    - A box outside the image boundary → the model trains on black bars.
    - A box of type "handwritten" with no text → mislabels the recognizer.
    - A box that is suspiciously tiny → likely a mis-click annotation.
  Catching these now prevents hours of debugging during training.

WHAT IS CHECKED:
  ✓ Bbox has exactly 4 numbers
  ✓ x2 > x1   (positive width)
  ✓ y2 > y1   (positive height)
  ✓ x1 >= 0   (inside image)
  ✓ y1 >= 0
  ✓ x2 <= image_width
  ✓ y2 <= image_height
  ✓ Not suspiciously tiny (< MIN_AREA pixels²)
  ✓ Not suspiciously huge (> 95% of image area)
  ✓ Region type is one of the 7 known types
  ✓ Text-bearing regions have non-empty text (warning, not error)
  ✓ image/graph regions do not have text (informational)

USAGE:
  python -m src.data.bbox_validation
  python -m src.data.bbox_validation --from-jsonl data/processed/train_split.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.utils.paths import DATA_CFG, ensure_dir, get_path, get_region_types

# ── Validation thresholds (tunable) ──────────────────────────────
MIN_AREA        = 50        # px²  — anything smaller is probably noise
MAX_AREA_RATIO  = 0.95      # fraction of total image area — full-page box is suspicious
VALID_TYPES     = set(get_region_types().keys())
NON_TEXT_TYPES  = set(DATA_CFG.get("non_text_types", ["image", "graph"]))


class BboxIssue:
    """Represents one detected issue with a bounding box."""
    __slots__ = ("severity", "category", "file_name", "region_idx",
                 "region_type", "bbox", "message")

    def __init__(self, severity, category, file_name, region_idx,
                 region_type, bbox, message):
        self.severity    = severity     # "error" | "warning" | "info"
        self.category    = category
        self.file_name   = file_name
        self.region_idx  = region_idx
        self.region_type = region_type
        self.bbox        = bbox
        self.message     = message

    def __str__(self):
        icon = {"error": "❌", "warning": "⚠️ ", "info": "ℹ️ "}.get(
            self.severity, "?"
        )
        return (
            f"{icon} [{self.severity.upper()}] {self.category} | "
            f"{self.file_name}  region[{self.region_idx}] "
            f"type={self.region_type}  bbox={self.bbox} | "
            f"{self.message}"
        )


def validate_record_bboxes(rec: dict) -> list[BboxIssue]:
    """
    Validate all bbox annotations in one metadata record.

    Args:
        rec: one metadata dict (file_name, image_width, image_height, regions)

    Returns:
        list of BboxIssue objects (empty = all clean)
    """
    issues: list[BboxIssue] = []
    fname  = rec.get("file_name", "unknown")
    img_w  = rec.get("image_width",  0) or 0
    img_h  = rec.get("image_height", 0) or 0

    regions = rec.get("regions", [])
    if isinstance(regions, str):
        regions = json.loads(regions)

    for idx, region in enumerate(regions):
        bbox  = region.get("bbox", None)
        rtype = region.get("type", "unknown")
        text  = region.get("text", "")
        lang  = region.get("language", "")
        leg   = region.get("legibility", "")

        # ── Unknown type ─────────────────────────────────────────
        if rtype not in VALID_TYPES:
            issues.append(BboxIssue(
                "error", "UNKNOWN_TYPE",
                fname, idx, rtype, bbox,
                f"'{rtype}' is not in the class map."
            ))

        # ── Bbox format ──────────────────────────────────────────
        if bbox is None or not isinstance(bbox, list):
            issues.append(BboxIssue(
                "error", "MISSING_BBOX",
                fname, idx, rtype, bbox,
                "bbox is None or not a list."
            ))
            continue

        if len(bbox) != 4:
            issues.append(BboxIssue(
                "error", "BBOX_LENGTH",
                fname, idx, rtype, bbox,
                f"bbox has {len(bbox)} elements (need 4)."
            ))
            continue

        x1, y1, x2, y2 = bbox

        # ── Numeric check ────────────────────────────────────────
        if not all(isinstance(v, (int, float)) for v in bbox):
            issues.append(BboxIssue(
                "error", "BBOX_NOT_NUMERIC",
                fname, idx, rtype, bbox,
                "bbox values must be numbers."
            ))
            continue

        # ── Positive dimensions ──────────────────────────────────
        if x2 <= x1:
            issues.append(BboxIssue(
                "error", "DEGENERATE_WIDTH",
                fname, idx, rtype, bbox,
                f"x2={x2} must be > x1={x1}."
            ))

        if y2 <= y1:
            issues.append(BboxIssue(
                "error", "DEGENERATE_HEIGHT",
                fname, idx, rtype, bbox,
                f"y2={y2} must be > y1={y1}."
            ))

        if x2 <= x1 or y2 <= y1:
            continue  # skip area/boundary checks for degenerate boxes

        # ── Within image bounds ──────────────────────────────────
        if img_w > 0 and img_h > 0:
            if x1 < 0:
                issues.append(BboxIssue(
                    "error", "OUTSIDE_BOUNDS",
                    fname, idx, rtype, bbox,
                    f"x1={x1} < 0."
                ))
            if y1 < 0:
                issues.append(BboxIssue(
                    "error", "OUTSIDE_BOUNDS",
                    fname, idx, rtype, bbox,
                    f"y1={y1} < 0."
                ))
            if x2 > img_w:
                issues.append(BboxIssue(
                    "error", "OUTSIDE_BOUNDS",
                    fname, idx, rtype, bbox,
                    f"x2={x2} > image_width={img_w}."
                ))
            if y2 > img_h:
                issues.append(BboxIssue(
                    "error", "OUTSIDE_BOUNDS",
                    fname, idx, rtype, bbox,
                    f"y2={y2} > image_height={img_h}."
                ))

        # ── Size checks ──────────────────────────────────────────
        area = (x2 - x1) * (y2 - y1)

        if area < MIN_AREA:
            issues.append(BboxIssue(
                "warning", "TINY_BOX",
                fname, idx, rtype, bbox,
                f"Area={area}px² is very small (< {MIN_AREA}px²). "
                "Check if this is a mis-click annotation."
            ))

        if img_w > 0 and img_h > 0:
            img_area = img_w * img_h
            ratio    = area / img_area
            if ratio > MAX_AREA_RATIO:
                issues.append(BboxIssue(
                    "warning", "HUGE_BOX",
                    fname, idx, rtype, bbox,
                    f"Box covers {ratio*100:.1f}% of the image — suspiciously large."
                ))

        # ── Text content checks ──────────────────────────────────
        if rtype not in NON_TEXT_TYPES and lang == "uk" and leg == "legible":
            if not text.strip():
                issues.append(BboxIssue(
                    "warning", "EMPTY_TEXT",
                    fname, idx, rtype, bbox,
                    f"Legible Ukrainian '{rtype}' region has empty text."
                ))

        if rtype in NON_TEXT_TYPES and text.strip():
            issues.append(BboxIssue(
                "info", "TEXT_IN_VISUAL",
                fname, idx, rtype, bbox,
                f"'{rtype}' region has text '{text[:30]}' — unusual but not fatal."
            ))

    return issues


def run_validation(records: list[dict]) -> dict:
    """
    Validate all records and return a summary dict.

    Returns:
        {
          "total_images": int,
          "total_regions": int,
          "errors": int,
          "warnings": int,
          "infos": int,
          "issues_by_category": {category: count},
          "sample_issues": [str, ...],   ← first 20 issues for report
          "passed": bool,
        }
    """
    summary = {
        "total_images":       len(records),
        "total_regions":      0,
        "errors":             0,
        "warnings":           0,
        "infos":              0,
        "issues_by_category": defaultdict(int),
        "sample_issues":      [],
        "passed":             False,
    }

    all_issues: list[BboxIssue] = []

    for rec in records:
        regions = rec.get("regions", [])
        if isinstance(regions, str):
            regions = json.loads(regions)
        summary["total_regions"] += len(regions)

        issues = validate_record_bboxes(rec)
        all_issues.extend(issues)

    for issue in all_issues:
        if issue.severity == "error":
            summary["errors"]   += 1
        elif issue.severity == "warning":
            summary["warnings"] += 1
        else:
            summary["infos"]    += 1
        summary["issues_by_category"][issue.category] += 1

    summary["issues_by_category"] = dict(summary["issues_by_category"])
    summary["sample_issues"]      = [str(i) for i in all_issues[:20]]
    summary["passed"]             = summary["errors"] == 0

    return summary


def print_summary(s: dict) -> None:
    """Print validation results to stdout."""
    print("\n" + "=" * 65)
    print("  BBOX VALIDATION RESULTS")
    print("=" * 65)
    print(f"  Images checked  : {s['total_images']:,}")
    print(f"  Regions checked : {s['total_regions']:,}")
    print(f"  Errors          : {s['errors']:,}     ← must be 0 before training")
    print(f"  Warnings        : {s['warnings']:,}   ← review, not blockers")
    print(f"  Infos           : {s['infos']:,}")

    if s["issues_by_category"]:
        print("\n  Issues by category:")
        for cat, count in sorted(
            s["issues_by_category"].items(), key=lambda x: -x[1]
        ):
            print(f"    {cat:<25}: {count:,}")

    if s["sample_issues"]:
        print(f"\n  First {len(s['sample_issues'])} issues (sample):")
        for line in s["sample_issues"]:
            print(f"    {line}")

    verdict = "✅ PASSED" if s["passed"] else "❌ FAILED — fix errors before training"
    print(f"\n  Result: {verdict}")
    print("=" * 65)


def save_report(s: dict) -> Path:
    """Save a Markdown validation report."""
    report_dir = ensure_dir(get_path("reports_dir"))
    report_path = report_dir / "bbox_validation_report.md"

    verdict = "✅ PASSED" if s["passed"] else "❌ FAILED"

    lines = [
        "# Bbox Validation Report\n\n",
        f"**Result: {verdict}**\n\n",
        "## Summary\n\n",
        "| Metric | Count |\n|--------|-------|\n",
        f"| Total images | {s['total_images']:,} |\n",
        f"| Total regions | {s['total_regions']:,} |\n",
        f"| Errors (blockers) | {s['errors']:,} |\n",
        f"| Warnings (review) | {s['warnings']:,} |\n",
        f"| Infos | {s['infos']:,} |\n\n",
        "## Issues by Category\n\n",
        "| Category | Count |\n|----------|-------|\n",
    ]
    for cat, count in sorted(
        s["issues_by_category"].items(), key=lambda x: -x[1]
    ):
        lines.append(f"| {cat} | {count:,} |\n")

    lines.append("\n## Sample Issues (first 20)\n\n```\n")
    for line in s["sample_issues"]:
        lines.append(line + "\n")
    lines.append("```\n\n")
    lines.append(
        "> Errors MUST be resolved before running `prepare_crops.py` "
        "or `convert_to_yolo.py`.\n"
    )

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"\n  ✅ Report saved → {report_path}")
    return report_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Validate all bounding boxes in dataset records."
    )
    ap.add_argument(
        "--from-jsonl", type=str, default=None,
        help="Local JSONL file (else loads from HF train split)"
    )
    ap.add_argument(
        "--split", default="train",
        choices=["train", "silver", "test"]
    )
    args = ap.parse_args()

    if args.from_jsonl:
        from src.utils.jsonl import read_jsonl
        records = read_jsonl(args.from_jsonl)
        print(f"[bbox_validation] Loaded {len(records):,} records from {args.from_jsonl}")
    else:
        # Try local train_split.jsonl first (faster, no HF needed)
        local_path = get_path("train_split")
        if local_path.exists():
            from src.utils.jsonl import read_jsonl
            records = read_jsonl(local_path)
            print(f"[bbox_validation] Loaded {len(records):,} records from {local_path}")
        else:
            from src.data.load_dataset import load_split
            ds = load_split(args.split)
            records = list(ds)

    summary = run_validation(records)
    print_summary(summary)
    save_report(summary)

    import sys
    if summary["passed"]:
        print("\n✅ Done. Bboxes are clean. Next steps:")
        print("   python -m src.recognition.prepare_crops")
        print("   python -m src.detection.convert_to_yolo")
        sys.exit(0)
    else:
        print(f"\n❌ Fix {summary['errors']} error(s) before proceeding.")
        sys.exit(1)
