"""
src/detection/validate_yolo.py  —  Stage 9b: Validate YOLO detection dataset.

WHY YOLO VALIDATION IS CRITICAL:
  Ultralytics YOLO training fails silently on bad label files:
    - A label file with the wrong number of values → NaN losses.
    - A class_id out of range → IndexError in the loss function.
    - Coordinates outside [0,1] → corrupted anchor assignments.
    - Image without a label file → YOLO skips it silently (wasted data).
    - Label file without an image → crash during training.

  This validator catches ALL of these before you waste GPU time.

CHECKS PERFORMED:
  ✓ Every image in yolo/images/train/ has a .txt label in yolo/labels/train/
  ✓ Every label file has a corresponding image
  ✓ Every label row has exactly 5 values
  ✓ class_id is an integer in [0, num_classes-1]
  ✓ x_center, y_center, width, height are floats in (0, 1]
  ✓ width > 0  and  height > 0
  ✓ data/yolo/data.yaml exists and is valid
  ✓ Sample labels are drawn back onto images for visual verification

USAGE:
  python -m src.detection.validate_yolo
  python -m src.detection.validate_yolo --split val
  python -m src.detection.validate_yolo --split train --n-visual 12
"""

import argparse
from pathlib import Path

import yaml

from src.utils.paths import DATA_CFG, ensure_dir, get_path, get_id_to_class

NUM_CLASSES   = DATA_CFG.get("num_classes", 7)
ID_TO_CLASS   = get_id_to_class()
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ─────────────────────────────────────────────────────────────────
# Label-level validation
# ─────────────────────────────────────────────────────────────────

def validate_label_file(lbl_path: Path) -> list[str]:
    """
    Validate one YOLO .txt label file.

    Returns:
        list of error strings (empty = valid)
    """
    errors = []
    try:
        content = lbl_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return [f"{lbl_path.name}: cannot read file: {e}"]

    if not content:
        return []  # empty label = background image, allowed

    for line_num, line in enumerate(content.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(
                f"{lbl_path.name} line {line_num}: "
                f"expected 5 values, got {len(parts)} → '{line}'"
            )
            continue

        try:
            class_id = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), \
                              float(parts[3]), float(parts[4])
        except ValueError as e:
            errors.append(f"{lbl_path.name} line {line_num}: not numeric: {e}")
            continue

        if class_id < 0 or class_id >= NUM_CLASSES:
            errors.append(
                f"{lbl_path.name} line {line_num}: "
                f"class_id={class_id} out of range [0, {NUM_CLASSES-1}]"
            )

        for name, val in [("x_center", xc), ("y_center", yc),
                          ("width", bw), ("height", bh)]:
            if not (0.0 <= val <= 1.0):
                errors.append(
                    f"{lbl_path.name} line {line_num}: "
                    f"{name}={val:.6f} not in [0, 1]"
                )

        if bw <= 0:
            errors.append(
                f"{lbl_path.name} line {line_num}: width={bw} must be > 0"
            )
        if bh <= 0:
            errors.append(
                f"{lbl_path.name} line {line_num}: height={bh} must be > 0"
            )

    return errors


# ─────────────────────────────────────────────────────────────────
# Dataset-level validation
# ─────────────────────────────────────────────────────────────────

def validate_split(
    img_dir: Path,
    lbl_dir: Path,
    split_name: str,
) -> dict:
    """
    Validate one split (train or val) of the YOLO dataset.

    Returns:
        summary dict
    """
    errors   = []
    warnings = []

    # Collect images
    img_files = {
        f.stem: f for f in img_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    } if img_dir.exists() else {}

    # Collect labels
    lbl_files = {
        f.stem: f for f in lbl_dir.iterdir()
        if f.suffix == ".txt"
    } if lbl_dir.exists() else {}

    if not img_files:
        errors.append(f"No images found in {img_dir}")
    if not lbl_files:
        errors.append(f"No label files found in {lbl_dir}")

    # Images without labels
    missing_lbls = set(img_files) - set(lbl_files)
    if missing_lbls:
        for stem in sorted(missing_lbls)[:10]:
            errors.append(f"Image has no label: {stem}")
        if len(missing_lbls) > 10:
            errors.append(f"… and {len(missing_lbls)-10} more images without labels")

    # Labels without images
    orphan_lbls = set(lbl_files) - set(img_files)
    if orphan_lbls:
        for stem in sorted(orphan_lbls)[:5]:
            warnings.append(f"Label has no image: {stem}.txt")

    # Validate each label file
    n_label_errors = 0
    for stem, lbl_path in lbl_files.items():
        file_errors = validate_label_file(lbl_path)
        if file_errors:
            n_label_errors += len(file_errors)
            errors.extend(file_errors[:3])   # cap per-file errors in report

    return {
        "split":         split_name,
        "n_images":      len(img_files),
        "n_labels":      len(lbl_files),
        "errors":        len(errors),
        "warnings":      len(warnings),
        "passed":        len(errors) == 0,
        "error_msgs":    errors[:30],
        "warn_msgs":     warnings[:10],
    }


def validate_data_yaml(yaml_path: Path) -> list[str]:
    """
    Check that data/yolo/data.yaml is valid and internally consistent.

    Returns:
        list of error strings (empty = valid)
    """
    errors = []
    if not yaml_path.exists():
        return [f"data.yaml not found: {yaml_path}"]

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        return [f"Cannot parse data.yaml: {e}"]

    for key in ("nc", "names", "train", "val"):
        if key not in cfg:
            errors.append(f"data.yaml missing key '{key}'")

    if "nc" in cfg and "names" in cfg:
        if cfg["nc"] != len(cfg["names"]):
            errors.append(
                f"data.yaml nc={cfg['nc']} but names has "
                f"{len(cfg['names'])} entries"
            )
        if cfg["nc"] != NUM_CLASSES:
            errors.append(
                f"data.yaml nc={cfg['nc']} doesn't match "
                f"config num_classes={NUM_CLASSES}"
            )

    return errors


# ─────────────────────────────────────────────────────────────────
# Visual debug: draw YOLO labels back onto images
# ─────────────────────────────────────────────────────────────────

def save_yolo_debug_images(
    img_dir: Path,
    lbl_dir: Path,
    out_dir: Path,
    n: int = 8,
) -> None:
    """
    Draw YOLO labels back onto images and save to out_dir for visual check.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  ⚠️  opencv not installed — skipping YOLO debug images.")
        return

    ensure_dir(out_dir)
    img_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    ])[:n]

    id_to_name = get_id_to_class()

    # Colors: one per class
    palette = [
        (0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 255),
        (0, 255, 255), (128, 128, 128), (64, 0, 128),
    ]

    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        content = lbl_path.read_text(encoding="utf-8").strip()

        for line in content.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError:
                continue

            # Convert YOLO → pixel coords
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color    = palette[cls % len(palette)]
            cls_name = id_to_name.get(cls, str(cls))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, cls_name, (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        out_path = out_dir / f"yolo_debug_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"    Saved debug → {out_path.name}")


# ─────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────

def print_summary(train_s: dict, val_s: dict, yaml_errs: list) -> None:
    """Print combined validation summary."""
    print("\n" + "=" * 65)
    print("  YOLO DATASET VALIDATION")
    print("=" * 65)

    for s in (train_s, val_s):
        verdict = "✅ PASS" if s["passed"] else "❌ FAIL"
        print(f"\n  [{verdict}] {s['split'].upper()} split")
        print(f"    Images : {s['n_images']:,}")
        print(f"    Labels : {s['n_labels']:,}")
        print(f"    Errors : {s['errors']:,}")
        print(f"    Warns  : {s['warnings']:,}")
        if s["error_msgs"]:
            print(f"    Sample errors:")
            for e in s["error_msgs"][:5]:
                print(f"      ❌ {e}")

    if yaml_errs:
        print(f"\n  data.yaml errors:")
        for e in yaml_errs:
            print(f"    ❌ {e}")
    else:
        print(f"\n  data.yaml: ✅ OK")

    overall = train_s["passed"] and val_s["passed"] and not yaml_errs
    verdict  = "✅ ALL PASSED" if overall else "❌ FAILED"
    print(f"\n  Overall result: {verdict}")
    print("=" * 65)


def save_report(train_s: dict, val_s: dict, yaml_errs: list) -> Path:
    """Save Markdown report."""
    report_dir  = ensure_dir(get_path("reports_dir"))
    report_path = report_dir / "yolo_validation_report.md"

    overall = train_s["passed"] and val_s["passed"] and not yaml_errs
    verdict = "✅ PASSED" if overall else "❌ FAILED"

    lines = [
        "# YOLO Dataset Validation Report\n\n",
        f"**Overall: {verdict}**\n\n",
        "## Per-Split Results\n\n",
        "| Split | Images | Labels | Errors | Warnings | Status |\n"
        "|-------|--------|--------|--------|----------|--------|\n",
    ]
    for s in (train_s, val_s):
        status = "✅ Pass" if s["passed"] else "❌ Fail"
        lines.append(
            f"| {s['split']} | {s['n_images']:,} | {s['n_labels']:,} | "
            f"{s['errors']:,} | {s['warnings']:,} | {status} |\n"
        )

    lines.append("\n## data.yaml\n\n")
    if yaml_errs:
        lines.append("```\n")
        for e in yaml_errs:
            lines.append(f"❌ {e}\n")
        lines.append("```\n")
    else:
        lines.append("✅ Valid\n")

    for s in (train_s, val_s):
        if s["error_msgs"]:
            lines.append(f"\n## {s['split'].capitalize()} Errors (sample)\n\n```\n")
            for e in s["error_msgs"]:
                lines.append(f"{e}\n")
            lines.append("```\n")

    lines.append(
        "\n> All errors must be fixed before running detector training.\n"
    )
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"  ✅ Report saved → {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Validate YOLO detection dataset."
    )
    ap.add_argument(
        "--split", default="both", choices=["train", "val", "both"],
        help="Which split(s) to validate (default: both)"
    )
    ap.add_argument(
        "--n-visual", type=int, default=8,
        help="How many debug images to save (default: 8)"
    )
    args = ap.parse_args()

    # Validate data.yaml
    yaml_path = get_path("yolo_data_yaml")
    yaml_errs = validate_data_yaml(yaml_path)
    if yaml_errs:
        print("⚠️  data.yaml issues:")
        for e in yaml_errs:
            print(f"  {e}")

    splits_to_check = (
        ["train", "val"] if args.split == "both" else [args.split]
    )

    results = {}
    for split in splits_to_check:
        img_dir = get_path(f"yolo_{split}_images")
        lbl_dir = get_path(f"yolo_{split}_labels")
        results[split] = validate_split(img_dir, lbl_dir, split)

        # Save visual debug images
        debug_dir = ensure_dir(
            get_path("debug_images_dir") / "yolo_checks" / split
        )
        if img_dir.exists():
            print(f"  Saving {args.n_visual} debug images for '{split}'…")
            save_yolo_debug_images(img_dir, lbl_dir, debug_dir, n=args.n_visual)

    # Fill in empty result if only one split was checked
    dummy = {"split": "n/a", "n_images": 0, "n_labels": 0,
             "errors": 0, "warnings": 0, "passed": True,
             "error_msgs": [], "warn_msgs": []}
    train_s = results.get("train", dummy)
    val_s   = results.get("val",   dummy)

    print_summary(train_s, val_s, yaml_errs)
    save_report(train_s, val_s, yaml_errs)

    import sys
    overall_ok = train_s["passed"] and val_s["passed"] and not yaml_errs
    if overall_ok:
        print("\n✅ YOLO dataset validated. Ready for detector training.")
        sys.exit(0)
    else:
        print("\n❌ Fix errors before training the detector.")
        sys.exit(1)
