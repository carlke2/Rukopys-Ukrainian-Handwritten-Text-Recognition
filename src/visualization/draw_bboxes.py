"""
src/visualization/draw_bboxes.py  —  Stage 5: Visualize annotations.

WHY VISUALIZATION IS MANDATORY BEFORE TRAINING:
  1. It confirms the annotations are correctly aligned with the image.
  2. It shows how complex and varied the documents are.
  3. It lets you spot annotation errors (wrong region types, shifted boxes).
  4. It reveals how much overlap and nesting regions have.
  5. If boxes look wrong here, your detector will learn wrong things.

WHAT THIS PRODUCES:
  For each sample, one PNG with coloured bounding boxes drawn on the
  document image. Each box is labelled with its region type.
  Saved to outputs/debug_images/

COLOR SCHEME (from data.yaml):
  handwritten → green
  printed     → red
  formula     → orange
  table       → magenta
  annotation  → cyan
  image       → gray
  graph       → purple

USAGE:
  python -m src.visualization.draw_bboxes
  python -m src.visualization.draw_bboxes --n-samples 20 --split train
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.paths import ensure_dir, get_path, get_region_types, DATA_CFG


def get_color(region_type: str) -> tuple[int, int, int]:
    """Return BGR color for a region type (from data.yaml)."""
    rt = get_region_types()
    if region_type in rt:
        return tuple(rt[region_type]["color"])  # already BGR
    return (200, 200, 200)  # gray fallback for unknown types


def draw_bboxes_on_image(
    image: Image.Image,
    regions: list[dict],
    thickness: int = 3,
    font_scale: float = 0.7,
) -> np.ndarray:
    """
    Draw bounding boxes on a PIL image.

    Args:
        image:     PIL Image (any mode)
        regions:   list of region dicts with 'bbox', 'type', 'text'
        thickness: box border thickness in pixels
        font_scale: text label size

    Returns:
        numpy array (BGR) with boxes drawn
    """
    # Convert PIL → OpenCV BGR
    img_np = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for region in regions:
        bbox  = region.get("bbox", [])
        rtype = region.get("type", "unknown")
        text  = region.get("text", "")
        lang  = region.get("language", "")
        leg   = region.get("legibility", "")

        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue

        x1, y1, x2, y2 = [int(c) for c in bbox]
        color = get_color(rtype)

        # Draw the box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        # Build label: type + legibility indicator
        label = rtype
        if leg == "illegible":
            label += " [!]"

        # Draw filled rectangle behind text so it is readable
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(
            img_bgr,
            (x1, label_y - th - baseline),
            (x1 + tw + 4, label_y + baseline),
            color, -1
        )
        cv2.putText(
            img_bgr, label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

    return img_bgr


def visualize_sample(
    sample: dict,
    out_dir: Path,
    sample_idx: int = 0,
    max_dim: int = 1600,
) -> Path:
    """
    Draw boxes on one sample and save to disk.

    Args:
        sample:     dict with 'image', 'file_name', 'regions'
        out_dir:    directory to save the debug PNG
        sample_idx: used in the output filename
        max_dim:    resize image if larger (keeps aspect ratio)

    Returns:
        path to saved image
    """
    ensure_dir(out_dir)

    # Get image
    pil_img = sample.get("image")
    if pil_img is None:
        # Try loading from disk if we have file_name
        fname = sample.get("file_name", f"sample_{sample_idx}.jpg")
        img_path = get_path("raw_train_images") / fname
        if not img_path.exists():
            print(f"  ⚠️  Image not found: {img_path} — skipping.")
            return None
        pil_img = Image.open(img_path)

    regions = sample.get("regions", [])
    if isinstance(regions, str):
        regions = json.loads(regions)

    # Resize large images for display (keeps all box coordinates intact
    # since we resize AFTER drawing, not before)
    drawn = draw_bboxes_on_image(pil_img, regions)

    h, w = drawn.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        drawn = cv2.resize(drawn, (int(w*scale), int(h*scale)))

    fname = sample.get("file_name", f"sample_{sample_idx}")
    stem  = Path(fname).stem
    out_path = out_dir / f"{sample_idx:04d}_{stem}_annotated.png"
    cv2.imwrite(str(out_path), drawn)
    return out_path


def run(records: list[dict], n_samples: int = 10, images_root: Path = None) -> None:
    """
    Visualize the first n_samples records and save debug images.

    Args:
        records:     list of metadata dicts
        n_samples:   how many to visualize
        images_root: directory where the raw images live
    """
    out_dir = ensure_dir(get_path("debug_images_dir"))

    if images_root is None:
        images_root = get_path("raw_train_images")

    print(f"[draw_bboxes] Visualising {n_samples} samples → {out_dir}")

    saved = 0
    for idx, rec in enumerate(records[:n_samples]):
        # Attach image from disk if not already in dict
        if "image" not in rec or rec["image"] is None:
            fname = rec.get("file_name", "")
            img_path = images_root / fname
            if img_path.exists():
                rec = dict(rec)  # don't mutate original
                rec["image"] = Image.open(img_path)
            else:
                print(f"  [{idx}]   Image not on disk: {img_path}")
                print(f"         (Download images with load_dataset first)")
                # Draw on a blank canvas so we can still visualize boxes
                w = rec.get("image_width", 800)
                h = rec.get("image_height", 1000)
                rec["image"] = Image.new("RGB", (w, h), color=(30, 30, 30))

        out_path = visualize_sample(rec, out_dir, sample_idx=idx)
        if out_path:
            print(f"  [{idx}]  Saved → {out_path.name}")
            saved += 1

    print(f"\n[draw_bboxes] Saved {saved}/{n_samples} images to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--split", default="train",
                    choices=["train", "silver", "test"])
    ap.add_argument("--from-jsonl", type=str, default=None)
    args = ap.parse_args()

    if args.from_jsonl:
        from src.utils.jsonl import read_jsonl
        records = read_jsonl(args.from_jsonl)
    else:
        from src.data.load_dataset import load_split
        ds = load_split(args.split)
        records = list(ds)

    run(records, n_samples=args.n_samples)
    print("\n Done. Next: python -m src.data.dataset_stats")
