"""
src/preprocessing/image_cleaning.py  —  Stage 13: Image preprocessing tools.

WHY PREPROCESSING EXISTS:
  Raw document scans vary wildly:
    - Phone photos → low contrast, shadows, perspective distortion
    - Old archives  → yellowed paper, faint ink, noise
    - School docs   → crumpled, skewed, pencil marks
  Preprocessing normalises these differences before feeding the model.

  IMPORTANT: Do NOT apply all steps blindly to every image.
  Some steps help certain document types and hurt others.
  Always visualise before and after to confirm improvement.

WHEN TO USE EACH STEP:
  grayscale          → always safe; removes irrelevant color variation
  contrast enhance   → helps faint ink on aged paper
  denoise            → helps phone photos with sensor noise
  adaptive threshold → helps old scanned docs; destroys photos
  deskew             → helps rotated/tilted page scans
  resize + pad       → always needed before feeding a fixed-size model

USAGE:
  from src.preprocessing.image_cleaning import (
      to_grayscale, enhance_contrast, denoise, adaptive_threshold,
      deskew, resize_and_pad, run_pipeline
  )

  cleaned = run_pipeline(pil_image, steps=["grayscale","contrast","denoise"])
"""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageEnhance


# ─────────────────────────────────────────────────────────────────
# Individual preprocessing functions
# Each takes a numpy BGR array and returns a numpy BGR array.
# (PIL images are converted at the entry / exit points.)
# ─────────────────────────────────────────────────────────────────

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert PIL Image (any mode) to OpenCV BGR array."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array back to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale then back to 3-channel BGR.

    WHY: Color carries no information for text recognition.
    Removing it reduces noise and speeds up the model.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def enhance_contrast(img: np.ndarray, clip_limit: float = 3.0,
                     tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    WHY: Standard histogram equalisation brightens everything uniformly,
    which can wash out already-clear text. CLAHE applies equalisation
    locally in small tiles, boosting faint areas without overexposing
    bright areas. This is ideal for aged documents with uneven lighting.

    Args:
        clip_limit: contrast amplification limit (higher = more contrast)
        tile_size:  size of grid tiles (8x8 is a good default)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    l_ch = clahe.apply(l_ch)
    lab  = cv2.merge((l_ch, a_ch, b_ch))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, h: float = 10.0) -> np.ndarray:
    """
    Apply Non-Local Means Denoising.

    WHY: Phone camera photos have sensor noise — random pixel variation
    that the model might try to memorise instead of learning text shapes.
    This filter averages similar patches across the image to smooth noise
    while preserving edges (the strokes of letters).

    Args:
        h: filter strength (higher removes more noise but blurs edges)
           10 is a safe default; try 5–20 depending on noise level.
    """
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def adaptive_threshold(img: np.ndarray,
                        block_size: int = 31,
                        c: int = 10) -> np.ndarray:
    """
    Apply adaptive (local) thresholding to binarize the image.

    WHY: Converts the image to pure black-and-white text on white
    background. Adaptive means each region uses its own local threshold
    rather than one global value — essential when lighting varies across
    the page (e.g., shadow in one corner).

    WHEN TO USE: Old scanned documents. NOT for full-color photos.

    Args:
        block_size: neighbourhood size for local threshold (must be odd)
        c:          constant subtracted from the local mean (tune per doc)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def deskew(img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Detect and correct page skew (rotation).

    WHY: Documents placed on a scanner or photographed at an angle
    appear rotated. Feeding a tilted image to the model forces it to
    learn rotation-invariant features, which is harder and requires
    more data. Straightening the image makes training easier.

    HOW: We find all text pixels, fit a minimum-area bounding box, and
    rotate by the detected angle. We clamp to max_angle to prevent
    wild rotations from noise.

    Args:
        max_angle: maximum correction angle in degrees (15° is safe)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert so text is white on black (easier for findNonZero)
    inv  = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 100:
        return img  # not enough pixels to detect angle

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angles in (-90, 0]. Adjust to (-45, 45].
    if angle < -45:
        angle += 90

    # Clamp to avoid extreme rotations
    angle = max(-max_angle, min(max_angle, angle))

    if abs(angle) < 0.5:
        return img  # negligible — skip

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def resize_and_pad(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    pad_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Resize image to fit within target dimensions, then pad to exact size.

    WHY: Models require fixed-size inputs. Simply squishing the image
    distorts letter shapes. Instead we resize keeping aspect ratio, then
    pad the remaining space with white (background colour).

    Args:
        target_w:  target width
        target_h:  target height
        pad_color: BGR color to pad with (default: white)
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create padded canvas
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    y_off  = (target_h - new_h) // 2
    x_off  = (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas


# ─────────────────────────────────────────────────────────────────
# Configurable pipeline
# ─────────────────────────────────────────────────────────────────

STEP_MAP = {
    "grayscale":  to_grayscale,
    "contrast":   enhance_contrast,
    "denoise":    denoise,
    "threshold":  adaptive_threshold,
    "deskew":     deskew,
}


def run_pipeline(
    image: Image.Image,
    steps: Sequence[str] = ("grayscale", "contrast"),
    target_size: tuple[int, int] | None = None,
    save_debug: Path | None = None,
) -> Image.Image:
    """
    Apply a sequence of preprocessing steps to a PIL image.

    Args:
        image:       input PIL Image
        steps:       ordered list of step names from STEP_MAP
        target_size: (w, h) — if given, resize+pad at the end
        save_debug:  if given, save before/after side-by-side here

    Returns:
        processed PIL Image
    """
    arr = pil_to_bgr(image)
    original = arr.copy()

    for step in steps:
        if step not in STEP_MAP:
            raise ValueError(
                f"Unknown preprocessing step: '{step}'. "
                f"Valid: {list(STEP_MAP.keys())}"
            )
        arr = STEP_MAP[step](arr)

    if target_size is not None:
        arr = resize_and_pad(arr, target_size[0], target_size[1])

    if save_debug is not None:
        save_debug = Path(save_debug)
        save_debug.parent.mkdir(parents=True, exist_ok=True)
        # Resize both to same height for side-by-side
        h = max(original.shape[0], arr.shape[0])
        def pad_h(a, h):
            dh = h - a.shape[0]
            return np.pad(a, ((0,dh),(0,0),(0,0)), constant_values=200)
        side_by_side = np.hstack([pad_h(original, h), pad_h(arr, h)])
        cv2.imwrite(str(save_debug), side_by_side)

    return bgr_to_pil(arr)


# ── CLI demo ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocessing.image_cleaning <image_path>")
        print("Example: python -m src.preprocessing.image_cleaning data/raw/train/images/doc.jpg")
        sys.exit(0)

    from src.utils.paths import ensure_dir
    img_path = Path(sys.argv[1])
    pil_img  = Image.open(img_path)

    debug_dir = ensure_dir(get_path("preprocessing_debug_dir"))

    steps_to_test = [
        ["grayscale"],
        ["grayscale", "contrast"],
        ["grayscale", "contrast", "denoise"],
        ["grayscale", "contrast", "denoise", "deskew"],
        ["threshold"],
    ]

    for steps in steps_to_test:
        name = "_".join(steps)
        out  = debug_dir / f"{img_path.stem}_{name}.png"
        run_pipeline(pil_img, steps=steps, save_debug=out)
        print(f"  ✅ {name} → {out}")

    print("\n✅ Done. Check outputs/debug_images/preprocessing/")
