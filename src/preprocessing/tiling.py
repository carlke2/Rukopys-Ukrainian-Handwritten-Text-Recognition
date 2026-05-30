"""
src/preprocessing/tiling.py  —  Split large documents into overlapping tiles.

WHY TILING:
  High-resolution scans (e.g., 4000x5000) contain tiny handwritten text.
  If you resize this whole page to YOLO's standard 640x640, the letters
  become single pixels and the model can't recognize them.

  Tiling splits the image into smaller chunks (e.g., 1024x1024) that
  preserve detail. Overlap ensures that regions on the boundary are
  captured completely in at least one tile.
"""

import numpy as np
from PIL import Image

def get_tiles(
    img: Image.Image,
    tile_size: int = 1024,
    overlap: int = 200,
) -> list[dict]:
    """
    Split a PIL image into overlapping tiles.

    Args:
        img:        input PIL image
        tile_size:  width and height of each tile
        overlap:    number of pixels to overlap between adjacent tiles

    Returns:
        list of {
            "image": PIL image (tile),
            "offset": (x_offset, y_offset) relative to original
        }
    """
    w, h = img.size
    stride = tile_size - overlap
    
    tiles = []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Ensure we don't go out of bounds (last tile in row/col)
            x1 = min(x, max(0, w - tile_size))
            y1 = min(y, max(0, h - tile_size))
            x2 = min(x1 + tile_size, w)
            y2 = min(y1 + tile_size, h)
            
            tile = img.crop((x1, y1, x2, y2))
            tiles.append({
                "image": tile,
                "offset": (x1, y1),
                "bbox_orig": (x1, y1, x2, y2)
            })
            
            if x2 >= w: break
        if y2 >= h: break
            
    return tiles

def map_bbox_to_orig(bbox_tile: list[float], offset: tuple[int, int]) -> list[float]:
    """Convert a bbox from tile coordinates to original image coordinates."""
    ox, oy = offset
    x1, y1, x2, y2 = bbox_tile
    return [x1 + ox, y1 + oy, x2 + ox, y2 + oy]

def map_bbox_to_tile(bbox_orig: list[float], offset: tuple[int, int], tile_size: int) -> list[float] | None:
    """
    Convert a bbox from original coordinates to tile coordinates.
    Returns None if the bbox is not fully or significantly contained in the tile.
    """
    ox, oy = offset
    x1, y1, x2, y2 = bbox_orig
    
    # Coordinates relative to tile
    tx1, ty1 = x1 - ox, y1 - oy
    tx2, ty2 = x2 - ox, y2 - oy
    
    # Check if bbox is within tile boundaries
    # We allow some leeway (intersection)
    if tx2 < 0 or ty2 < 0 or tx1 > tile_size or ty1 > tile_size:
        return None
        
    # Clip to tile boundaries
    cx1 = max(0, tx1)
    cy1 = max(0, ty1)
    cx2 = min(tile_size, tx2)
    cy2 = min(tile_size, ty2)
    
    # If the clipped box is too small or drastically different from original, discard
    # This prevents "sliver" boxes on the edges
    orig_w, orig_h = x2 - x1, y2 - y1
    clip_w, clip_h = cx2 - cx1, cy2 - cy1
    
    if clip_w < orig_w * 0.5 or clip_h < orig_h * 0.5:
        return None
        
    return [cx1, cy1, cx2, cy2]
