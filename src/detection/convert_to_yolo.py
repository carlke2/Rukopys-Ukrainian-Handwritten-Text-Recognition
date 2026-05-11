"""
src/detection/convert_to_yolo.py  —  Stage 9: Build YOLO detection dataset.

WHY YOLO FORMAT:
  The detector (YOLOv8/v11) requires a specific data layout:
    data/yolo/images/train/<img>.jpg
    data/yolo/labels/train/<img>.txt   ← one row per region

  Each label row: <class_id> <x_center> <y_center> <width> <height>
  All coordinates are NORMALISED to [0, 1] relative to image dimensions.

  Example: a region at pixel [100, 200, 500, 400] in a 1000x1200 image:
    x_center = (100+500)/2 / 1000 = 0.3
    y_center = (200+400)/2 / 1200 = 0.25
    width    = (500-100) / 1000   = 0.4
    height   = (400-200) / 1200   = 0.167

CLASS MAP (from data.yaml):
  handwritten → 0
  printed     → 1
  formula     → 2
  table       → 3
  annotation  → 4
  image       → 5
  graph       → 6

OUTPUT:
  data/yolo/images/train/   — symlinked or copied images
  data/yolo/images/val/
  data/yolo/labels/train/   — .txt label files
  data/yolo/labels/val/
  data/yolo/data.yaml       — dataset config for Ultralytics

USAGE:
  python -m src.detection.convert_to_yolo
"""

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from src.utils.paths import (
    DATA_CFG, ensure_dir, get_class_to_id, get_path, PROJECT_ROOT
)
from src.utils.jsonl import read_jsonl


def bbox_to_yolo(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """
    Convert absolute [x1,y1,x2,y2] bbox to YOLO normalised format.

    Returns:
        (x_center, y_center, width, height) each in [0, 1]
    """
    x_center = ((x1 + x2) / 2) / img_w
    y_center  = ((y1 + y2) / 2) / img_h
    width     = (x2 - x1) / img_w
    height    = (y2 - y1) / img_h
    # Clamp to [0, 1] to avoid rounding errors
    x_center  = max(0.0, min(1.0, x_center))
    y_center  = max(0.0, min(1.0, y_center))
    width     = max(0.0, min(1.0, width))
    height    = max(0.0, min(1.0, height))
    return x_center, y_center, width, height


def convert_records(
    records: list[dict],
    images_src: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    class_map: dict[str, int],
    copy_images: bool = True,
) -> tuple[int, int, int]:
    """
    Convert a list of metadata records to YOLO format.

    Args:
        records:      metadata dicts
        images_src:   directory where source images live
        out_img_dir:  YOLO images output directory
        out_lbl_dir:  YOLO labels output directory
        class_map:    region_type → class_id
        copy_images:  if True, copy images to yolo dir;
                      if False, create symlinks (saves disk space on Linux)

    Returns:
        (images_processed, regions_written, images_skipped)
    """
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    n_images   = 0
    n_regions  = 0
    n_skipped  = 0

    for rec in tqdm(records, desc=f"→ {out_img_dir.parent.name}/{out_img_dir.name}"):
        fname    = rec.get("file_name", "")
        img_w    = rec.get("image_width", 0)
        img_h    = rec.get("image_height", 0)
        regions  = rec.get("regions", [])
        if isinstance(regions, str):
            regions = json.loads(regions)

        src_img = images_src / fname
        if not src_img.exists():
            n_skipped += 1
            continue

        if img_w <= 0 or img_h <= 0:
            # Fall back to reading from PIL if dimensions are missing
            try:
                from PIL import Image
                with Image.open(src_img) as im:
                    img_w, img_h = im.size
            except Exception:
                n_skipped += 1
                continue

        # Build label lines
        label_lines = []
        for region in regions:
            rtype = region.get("type", "")
            bbox  = region.get("bbox", [])

            if rtype not in class_map:
                continue
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue

            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = class_map[rtype]
            xc, yc, bw, bh = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
            label_lines.append(
                f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )
            n_regions += 1

        # Write label file (even if empty — YOLO needs it to know
        # this image has no regions, i.e. "background" sample)
        stem = Path(fname).stem
        lbl_path = out_lbl_dir / f"{stem}.txt"
        lbl_path.write_text("\n".join(label_lines), encoding="utf-8")

        # Copy or symlink image
        dst_img = out_img_dir / fname
        if not dst_img.exists():
            if copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                dst_img.symlink_to(src_img.resolve())

        n_images += 1

    return n_images, n_regions, n_skipped


def write_yolo_data_yaml(
    train_img_dir: Path,
    val_img_dir: Path,
    class_map: dict[str, int],
    out_path: Path,
) -> None:
    """
    Write the data.yaml file that Ultralytics YOLO needs.

    This file tells YOLO where images are, how many classes there are,
    and what each class is called.
    """
    # YOLO wants paths relative to the yaml file location
    id_to_name = {v: k for k, v in class_map.items()}
    names = [id_to_name[i] for i in range(len(id_to_name))]

    content = (
        f"# YOLO dataset config — auto-generated by convert_to_yolo.py\n"
        f"path: {out_path.parent.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\nnc: {len(names)}\n"
        f"names: {names}\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"[convert_to_yolo] ✅ data.yaml → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-copy", action="store_true",
                    help="Use symlinks instead of copying images (Linux only)")
    args = ap.parse_args()

    class_map   = get_class_to_id()
    images_root = get_path("raw_train_images")

    for split_name, jsonl_key, img_key, lbl_key in [
        ("train", "train_split", "yolo_train_images", "yolo_train_labels"),
        ("val",   "val_split",   "yolo_val_images",   "yolo_val_labels"),
    ]:
        jsonl_path = get_path(jsonl_key)
        if not jsonl_path.exists():
            print(f"  ⚠️  {jsonl_path} not found — run make_splits.py first.")
            continue

        records = read_jsonl(jsonl_path)
        n_img, n_reg, n_skip = convert_records(
            records,
            images_src  = images_root,
            out_img_dir = get_path(img_key),
            out_lbl_dir = get_path(lbl_key),
            class_map   = class_map,
            copy_images = not args.no_copy,
        )
        print(f"  {split_name}: {n_img:,} images, {n_reg:,} regions, "
              f"{n_skip:,} skipped")

    write_yolo_data_yaml(
        train_img_dir = get_path("yolo_train_images"),
        val_img_dir   = get_path("yolo_val_images"),
        class_map     = class_map,
        out_path      = get_path("yolo_data_yaml"),
    )

    print("\n✅ YOLO dataset ready.")
    print("   Next: python -m src.submission.build_empty_submission")
