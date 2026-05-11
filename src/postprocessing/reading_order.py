"""
src/postprocessing/reading_order.py  —  Sort regions in reading order.

WHY READING ORDER MATTERS:
  After the detector finds all regions on a page, they come back in
  arbitrary order (usually the order the model detected them).
  The final submission JSON must list regions in the order a human
  would read the document: top to bottom, left to right.

  A correct reading order is also important for downstream tasks like
  document understanding or table extraction, where the sequence of
  paragraphs carries meaning.

STRATEGY:
  Simple but effective two-pass approach:
    1. Assign each region to a "row band" based on its vertical position.
       Two regions are in the same row if their y-centers are within
       ROW_BAND_TOLERANCE pixels of each other.
    2. Within each band, sort left to right by x_center.
    3. Sort bands top to bottom.

  This handles most document layouts correctly. Multi-column layouts
  may need a more sophisticated algorithm (future work).

USAGE:
  from src.postprocessing.reading_order import sort_regions

  ordered = sort_regions(regions, row_band_tolerance=30)
"""


def sort_regions(
    regions: list[dict],
    row_band_tolerance: int = 30,
) -> list[dict]:
    """
    Sort regions in reading order: top-to-bottom, left-to-right.

    Args:
        regions:             list of region dicts with 'bbox' key
        row_band_tolerance:  max vertical distance (px) for two regions
                             to be considered in the same row band.
                             Increase for documents with large line spacing.

    Returns:
        new list of region dicts in reading order
        (original dicts are not mutated)
    """
    if not regions:
        return []

    # Filter out regions without valid bboxes
    valid = []
    for r in regions:
        bbox = r.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            valid.append(r)

    if not valid:
        return list(regions)

    # Compute y_center for each region
    def y_center(r):
        x1, y1, x2, y2 = r["bbox"]
        return (y1 + y2) / 2

    def x_center(r):
        x1, y1, x2, y2 = r["bbox"]
        return (x1 + x2) / 2

    # Sort by y_center first
    sorted_by_y = sorted(valid, key=y_center)

    # Group into row bands
    bands: list[list[dict]] = []
    current_band: list[dict] = [sorted_by_y[0]]
    current_y = y_center(sorted_by_y[0])

    for region in sorted_by_y[1:]:
        yc = y_center(region)
        if abs(yc - current_y) <= row_band_tolerance:
            current_band.append(region)
        else:
            bands.append(current_band)
            current_band = [region]
            current_y    = yc

    bands.append(current_band)

    # Sort each band left-to-right, then flatten
    ordered = []
    for band in bands:
        band_sorted = sorted(band, key=x_center)
        ordered.extend(band_sorted)

    return ordered


if __name__ == "__main__":
    # Simple test: 6 regions in scrambled order
    test_regions = [
        {"bbox": [400, 100, 800, 140], "type": "printed",     "text": "B"},
        {"bbox": [10,  100, 390, 140], "type": "handwritten", "text": "A"},
        {"bbox": [10,  200, 390, 240], "type": "handwritten", "text": "C"},
        {"bbox": [400, 300, 800, 340], "type": "printed",     "text": "F"},
        {"bbox": [400, 200, 800, 240], "type": "printed",     "text": "D"},
        {"bbox": [10,  300, 390, 340], "type": "handwritten", "text": "E"},
    ]

    ordered = sort_regions(test_regions, row_band_tolerance=30)
    result  = "".join(r["text"] for r in ordered)
    expected = "ABCDEF"
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✅ Reading order: {result}  (correct!)")
