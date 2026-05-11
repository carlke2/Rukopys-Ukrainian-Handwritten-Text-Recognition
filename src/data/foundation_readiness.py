"""
src/data/foundation_readiness.py  —  Final gate before model training.

PURPOSE:
  This script is the LAST thing you run before installing
  requirements-training.txt and starting model training.

  It checks every condition from both readiness gates:
    • Detector training readiness gate
    • Recognizer training readiness gate

  If any gate fails you get a clear ❌ with the exact fix needed.
  When all gates pass you get a signed-off checklist and clear
  instructions for starting training.

HOW TO RUN:
  python -m src.data.foundation_readiness

EXPECTED OUTPUT WHEN READY:
  All 25 checklist items show ✅
  Final banner: "FOUNDATION COMPLETE — SAFE TO BEGIN TRAINING"
"""

import json
from pathlib import Path

from src.utils.paths import DATA_CFG, get_path, ensure_dir

# ── Helper ────────────────────────────────────────────────────────

def _check(label: str, condition: bool, fix: str = "") -> tuple[bool, str]:
    icon   = "✅" if condition else "❌"
    msg    = f"  {icon}  {label}"
    if not condition and fix:
        msg += f"\n      FIX: {fix}"
    return condition, msg


def _file_exists(key: str) -> bool:
    try:
        return get_path(key).exists()
    except KeyError:
        return False


def _dir_has_files(key: str, exts: set | None = None) -> bool:
    try:
        d = get_path(key)
        if not d.exists():
            return False
        if exts:
            return any(f.suffix.lower() in exts for f in d.iterdir())
        return any(True for _ in d.iterdir())
    except (KeyError, StopIteration):
        return False


def _jsonl_nonempty(key: str) -> bool:
    try:
        p = get_path(key)
        if not p.exists():
            return False
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return True
        return False
    except Exception:
        return False


def _csv_nonempty(key: str) -> bool:
    try:
        p = get_path(key)
        if not p.exists():
            return False
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        return len(lines) > 1   # header + at least one data row
    except Exception:
        return False


def _report_exists(filename: str) -> bool:
    try:
        return (get_path("reports_dir") / filename).exists()
    except Exception:
        return False


def _source_exists(module_path: str) -> bool:
    """Check that a Python source file exists in src/."""
    from src.utils.paths import PROJECT_ROOT
    p = PROJECT_ROOT / module_path
    return p.exists()


# ── Gate checks ───────────────────────────────────────────────────

def run_checklist() -> list[tuple[bool, str]]:
    """
    Run all 25 foundation checklist items.

    Returns:
        list of (passed: bool, message: str)
    """
    results = []

    # ── 1. Project concept & configs ─────────────────────────────
    results.append(_check(
        "configs/data.yaml exists",
        _file_exists("processed_root"),   # if paths resolve, yaml loaded OK
        "Create configs/data.yaml from the project template."
    ))
    results.append(_check(
        "configs/detector.yaml exists",
        _source_exists("configs/detector.yaml"),
        "Create configs/detector.yaml."
    ))
    results.append(_check(
        "configs/recognizer.yaml exists",
        _source_exists("configs/recognizer.yaml"),
        "Create configs/recognizer.yaml."
    ))

    # ── 2. Dataset loading modules ────────────────────────────────
    results.append(_check(
        "src/data/load_dataset.py exists",
        _source_exists("src/data/load_dataset.py"),
        "python -m src.data.load_dataset --splits train test"
    ))
    results.append(_check(
        "src/data/inspect_dataset.py exists",
        _source_exists("src/data/inspect_dataset.py"),
        "python -m src.data.inspect_dataset"
    ))

    # ── 3. Dataset stats ──────────────────────────────────────────
    results.append(_check(
        "data/processed/dataset_stats.json exists",
        _file_exists("dataset_stats"),
        "python -m src.data.dataset_stats --from-jsonl data/processed/train_raw.jsonl"
    ))
    results.append(_check(
        "outputs/reports/dataset_stats_train.md exists",
        _report_exists("dataset_stats_train.md"),
        "python -m src.data.dataset_stats"
    ))

    # ── 4. Train / val split ──────────────────────────────────────
    results.append(_check(
        "data/processed/train_split.jsonl exists & non-empty",
        _jsonl_nonempty("train_split"),
        "python -m src.data.make_splits"
    ))
    results.append(_check(
        "data/processed/val_split.jsonl exists & non-empty",
        _jsonl_nonempty("val_split"),
        "python -m src.data.make_splits"
    ))

    # ── 5. Bbox validation ────────────────────────────────────────
    results.append(_check(
        "src/data/bbox_validation.py exists",
        _source_exists("src/data/bbox_validation.py"),
        "This file should already exist."
    ))
    results.append(_check(
        "outputs/reports/bbox_validation_report.md exists",
        _report_exists("bbox_validation_report.md"),
        "python -m src.data.bbox_validation --from-jsonl data/processed/train_split.jsonl"
    ))

    # ── 6. Annotation visualization ───────────────────────────────
    results.append(_check(
        "src/visualization/draw_bboxes.py exists",
        _source_exists("src/visualization/draw_bboxes.py"),
        "This file should already exist."
    ))
    results.append(_check(
        "outputs/debug_images/ has annotation images",
        _dir_has_files("debug_images_dir", {".png", ".jpg"}),
        "python -m src.visualization.draw_bboxes --n-samples 10"
    ))

    # ── 7. Recognition crops ──────────────────────────────────────
    results.append(_check(
        "data/crops/train/images/ has crop images",
        _dir_has_files("crops_train_img", {".jpg", ".jpeg", ".png"}),
        "python -m src.recognition.prepare_crops --split train"
    ))
    results.append(_check(
        "data/crops/train/labels.csv exists & non-empty",
        _csv_nonempty("crops_train_csv"),
        "python -m src.recognition.prepare_crops --split train"
    ))
    results.append(_check(
        "data/crops/val/labels.csv exists & non-empty",
        _csv_nonempty("crops_val_csv"),
        "python -m src.recognition.prepare_crops --split val"
    ))
    results.append(_check(
        "outputs/reports/crop_validation_report_train.md exists",
        _report_exists("crop_validation_report_train.md"),
        "python -m src.recognition.validate_crops --split train"
    ))

    # ── 8. YOLO detection dataset ─────────────────────────────────
    results.append(_check(
        "data/yolo/images/train/ has images",
        _dir_has_files("yolo_train_images", {".jpg", ".jpeg", ".png"}),
        "python -m src.detection.convert_to_yolo"
    ))
    results.append(_check(
        "data/yolo/labels/train/ has .txt files",
        _dir_has_files("yolo_train_labels", {".txt"}),
        "python -m src.detection.convert_to_yolo"
    ))
    results.append(_check(
        "data/yolo/data.yaml exists",
        _file_exists("yolo_data_yaml"),
        "python -m src.detection.convert_to_yolo"
    ))
    results.append(_check(
        "outputs/reports/yolo_validation_report.md exists",
        _report_exists("yolo_validation_report.md"),
        "python -m src.detection.validate_yolo"
    ))

    # ── 9. Submission pipeline ────────────────────────────────────
    results.append(_check(
        "data/submissions/ has an empty baseline CSV",
        _dir_has_files("submissions_dir", {".csv"}),
        "python -m src.submission.build_empty_submission"
    ))
    results.append(_check(
        "outputs/reports/submission_validation_report.md exists",
        _report_exists("submission_validation_report.md"),
        "python -m src.submission.validate_submission --csv <your_csv>"
    ))

    # ── 10. Evaluation & postprocessing utilities ─────────────────
    results.append(_check(
        "src/evaluation/cer.py exists",
        _source_exists("src/evaluation/cer.py"),
        "This file should already exist."
    ))
    results.append(_check(
        "src/evaluation/iou.py exists",
        _source_exists("src/evaluation/iou.py"),
        "This file should already exist."
    ))
    results.append(_check(
        "src/postprocessing/reading_order.py exists",
        _source_exists("src/postprocessing/reading_order.py"),
        "This file should already exist."
    ))

    return results


def print_checklist(results: list[tuple[bool, str]]) -> int:
    """Print checklist and return number of failures."""
    print("\n" + "=" * 65)
    print("  FOUNDATION READINESS CHECKLIST")
    print("=" * 65)

    failures = 0
    for i, (passed, msg) in enumerate(results, start=1):
        print(f"  [{i:02d}] {msg.strip()}")
        if not passed:
            failures += 1

    print("\n" + "-" * 65)
    passed_count = len(results) - failures
    print(f"  {passed_count}/{len(results)} checks passed")

    if failures == 0:
        print("\n  🎉  FOUNDATION COMPLETE — SAFE TO BEGIN TRAINING 🎉")
        print("\n  Next steps:")
        print("    1. pip install -r requirements-training.txt")
        print("    2. Detector:   see configs/detector.yaml for training cmd")
        print("    3. Recognizer: see configs/recognizer.yaml for training cmd")
    else:
        print(f"\n  ❌  {failures} check(s) failed. Fix them before training.")

    print("=" * 65)
    return failures


def save_readiness_report(results: list[tuple[bool, str]]) -> Path:
    """Save the readiness checklist as a Markdown report."""
    report_dir  = ensure_dir(get_path("reports_dir"))
    report_path = report_dir / "foundation_readiness_report.md"

    failures = sum(1 for ok, _ in results if not ok)
    overall  = "✅ READY FOR TRAINING" if failures == 0 else f"❌ {failures} CHECKS FAILING"

    lines = [
        "# Foundation Readiness Report\n\n",
        f"**Overall Status: {overall}**\n\n",
        "## Checklist\n\n",
    ]

    for i, (passed, msg) in enumerate(results, start=1):
        icon = "✅" if passed else "❌"
        # strip the icon from msg since we're adding it separately
        clean_msg = msg.strip().lstrip("✅❌ ")
        lines.append(f"- [{icon}] **{i:02d}.** {clean_msg}\n")

    lines.append("\n---\n\n")
    if failures == 0:
        lines.append(
            "## 🎉 Foundation Complete\n\n"
            "All readiness gates passed. Install training requirements:\n\n"
            "```bash\n"
            "pip install -r requirements-training.txt\n"
            "```\n\n"
            "Then follow `configs/detector.yaml` and `configs/recognizer.yaml` "
            "for training commands.\n"
        )
    else:
        lines.append(
            "## ❌ Action Required\n\n"
            "Fix the failing checks above before starting training.\n"
            "Each ❌ item includes a FIX command.\n"
        )

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"\n  ✅ Readiness report saved → {report_path}")
    return report_path


if __name__ == "__main__":
    results  = run_checklist()
    failures = print_checklist(results)
    save_readiness_report(results)

    import sys
    sys.exit(0 if failures == 0 else 1)
