"""
src/visualization/make_contact_sheet.py  —  Contact sheet of crops.

WHY: A contact sheet shows many small images tiled together in one
     overview image. After generating recognition crops, this lets you
     quickly check that hundreds of crops look correct without opening
     each one individually.

USAGE:
  python -m src.visualization.make_contact_sheet
  python -m src.visualization.make_contact_sheet --csv data/crops/train/labels.csv --n 80
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.paths import ensure_dir, get_path


def make_contact_sheet(
    image_paths: list[Path],
    labels: list[str],
    thumb_w: int = 300,
    thumb_h: int = 80,
    cols: int = 4,
    out_path: Path = None,
) -> Path:
    """
    Tile a list of crop images into a single contact-sheet PNG.

    Args:
        image_paths: list of paths to crop images
        labels:      corresponding ground-truth text labels
        thumb_w:     width of each thumbnail
        thumb_h:     height of each thumbnail
        cols:        number of columns in the grid
        out_path:    where to save the result

    Returns:
        path to saved PNG
    """
    n = len(image_paths)
    rows = (n + cols - 1) // cols

    # Canvas dimensions + padding
    pad = 4
    cell_w = thumb_w + pad * 2
    cell_h = thumb_h + 30 + pad  # 30px for text label
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)

    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        row = i // cols
        col = i % cols
        x0  = col * cell_w + pad
        y0  = row * cell_h + pad

        if Path(img_path).exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                thumb = cv2.resize(img, (thumb_w, thumb_h))
                canvas[y0:y0+thumb_h, x0:x0+thumb_w] = thumb
        else:
            # Gray placeholder
            cv2.rectangle(canvas, (x0, y0),
                          (x0+thumb_w, y0+thumb_h), (70, 70, 70), -1)

        # Draw text label below thumbnail
        disp = (label[:40] + "…") if len(label) > 40 else label
        cv2.putText(
            canvas, disp,
            (x0, y0 + thumb_h + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (200, 200, 200), 1, cv2.LINE_AA,
        )

    if out_path is None:
        out_path = ensure_dir(get_path("debug_images_dir")) / "contact_sheet.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"[contact_sheet] ✅ Saved {n} thumbnails → {out_path}")
    return Path(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default=str(get_path("crops_train_csv")))
    ap.add_argument("--n",    type=int, default=60)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.head(args.n)
    paths  = [Path(p) for p in df["crop_path"].tolist()]
    labels = df["text"].fillna("").tolist()
    make_contact_sheet(paths, labels, cols=args.cols)
