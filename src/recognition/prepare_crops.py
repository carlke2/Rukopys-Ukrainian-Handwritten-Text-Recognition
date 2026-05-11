"""
src/recognition/prepare_crops.py  —  Stage 8: Extract recognition crops.

WHY CROPS INSTEAD OF FULL PAGES:
  The text recognition model reads ONE text line or region at a time,
  not the entire page. Feeding a full A4 page into TrOCR would:
    - Exceed its input resolution
    - Force it to learn spatial layout instead of just text reading
    - Waste compute on blank margins and non-text regions

  We crop each annotated region from the full page image and save it
  as a separate small image. The recognizer trains on these crops.

WHAT IS KEPT (scorable recognition regions):
  - language   = "uk"        (Ukrainian — what we are trying to read)
  - legibility = "legible"   (illegible text cannot be labelled correctly)
  - type NOT IN image, graph (those are visual; no text to read)
  - text is non-empty        (we need a label to train on)
  - bbox is valid            (x2>x1, y2>y1, positive size)
  - crop size >= minimum     (tiny crops contain no readable content)

OUTPUT:
  data/crops/train/images/<parent_stem>_<region_idx>.jpg
  data/crops/train/labels.csv  with columns:
    crop_path, text, type, source, parent_image, bbox

USAGE:
  python -m src.recognition.prepare_crops
  python -m src.recognition.prepare_crops --split val
"""

import argparse
import json
import csv
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.utils.paths import DATA_CFG, ensure_dir, get_path

NON_TEXT = set(DATA_CFG.get("non_text_types", ["image", "graph"]))
MIN_W    = DATA_CFG.get("min_crop_width",  16)
MIN_H    = DATA_CFG.get("min_crop_height", 8)


def is_valid_crop(region: dict) -> tuple[bool, str]:
    """
    Check whether a region should be used as a recognition crop.

    Returns:
        (is_valid: bool, reason: str)  — reason explains why if invalid
    """
    bbox  = region.get("bbox", [])
    rtype = region.get("type", "")
    lang  = region.get("language", "")
    leg   = region.get("legibility", "")
    text  = region.get("text", "")

    if rtype in NON_TEXT:
        return False, f"non-text type ({rtype})"
    if lang != "uk":
        return False, f"language={lang} (need uk)"
    if leg != "legible":
        return False, f"legibility={leg}"
    if not text.strip():
        return False, "empty text"
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return False, "invalid bbox format"

    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False, f"degenerate bbox {bbox}"
    if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
        return False, f"crop too small ({x2-x1}x{y2-y1})"

    return True, "ok"


def crop_region(page_img: Image.Image, bbox: list) -> Image.Image:
    """
    Crop a region from the full page image using the bbox.

    Args:
        page_img: full-page PIL Image
        bbox:     [x1, y1, x2, y2] in absolute pixels

    Returns:
        cropped PIL Image
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]
    # Clamp to image bounds
    w, h = page_img.size
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    return page_img.crop((x1, y1, x2, y2))


def prepare_crops_from_records(
    records: list[dict],
    images_root: Path,
    out_img_dir: Path,
    out_csv: Path,
) -> int:
    """
    Process a list of metadata records and extract all valid crops.

    Args:
        records:      list of metadata dicts (file_name + regions)
        images_root:  directory where full-page images live
        out_img_dir:  where to save crop PNGs/JPGs
        out_csv:      path for the labels CSV

    Returns:
        number of crops saved
    """
    ensure_dir(out_img_dir)
    ensure_dir(out_csv.parent)

    skipped_img = 0
    skipped_reg = 0
    total_crops = 0

    csv_rows = []

    for rec in tqdm(records, desc="Extracting crops"):
        fname = rec.get("file_name", "")
        img_path = images_root / fname

        if not img_path.exists():
            skipped_img += 1
            continue

        try:
            page_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ⚠️  Cannot open {img_path}: {e}")
            skipped_img += 1
            continue

        regions = rec.get("regions", [])
        if isinstance(regions, str):
            regions = json.loads(regions)

        source = rec.get("source", "unknown")
        stem   = Path(fname).stem

        for region_idx, region in enumerate(regions):
            valid, reason = is_valid_crop(region)
            if not valid:
                skipped_reg += 1
                continue

            bbox  = region["bbox"]
            rtype = region["type"]
            text  = region["text"]

            crop = crop_region(page_img, bbox)
            crop_name = f"{stem}_r{region_idx:04d}.jpg"
            crop_path = out_img_dir / crop_name

            # Save as JPEG (smaller files; quality=95 keeps detail)
            crop.save(str(crop_path), "JPEG", quality=95)

            csv_rows.append({
                "crop_path":    str(crop_path),
                "text":         text,
                "type":         rtype,
                "source":       source,
                "parent_image": fname,
                "bbox":         json.dumps(bbox),
            })
            total_crops += 1

    # Write labels CSV
    if csv_rows:
        fieldnames = ["crop_path", "text", "type", "source",
                      "parent_image", "bbox"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n[prepare_crops] Results:")
    print(f"  Images skipped  : {skipped_img:,}")
    print(f"  Regions skipped : {skipped_reg:,}  (wrong lang/leg/type/size)")
    print(f"  Crops saved     : {total_crops:,}")
    print(f"  Labels CSV      : {out_csv}")

    return total_crops


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val"])
    args = ap.parse_args()

    from src.utils.jsonl import read_jsonl

    if args.split == "train":
        jsonl_path  = get_path("train_split")
        out_img_dir = get_path("crops_train_img")
        out_csv     = get_path("crops_train_csv")
        images_root = get_path("raw_train_images")
    else:
        jsonl_path  = get_path("val_split")
        out_img_dir = get_path("crops_val_img")
        out_csv     = get_path("crops_val_csv")
        images_root = get_path("raw_train_images")  # val comes from train images

    records = read_jsonl(jsonl_path)
    n = prepare_crops_from_records(records, images_root, out_img_dir, out_csv)
    print(f"\n✅ Done — {n:,} crops ready for recognizer training.")
    print("   Next: python -m src.detection.convert_to_yolo")
