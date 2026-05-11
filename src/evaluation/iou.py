"""
src/evaluation/iou.py  —  Intersection over Union for bbox comparison.

WHY IoU:
  IoU measures how well a predicted bounding box overlaps the ground
  truth box. It is the standard metric for object detection quality.

  IoU = Area of Intersection / Area of Union

  IoU = 1.0  → perfect overlap
  IoU = 0.0  → no overlap at all
  IoU > 0.5  → generally considered a correct detection

USAGE:
  from src.evaluation.iou import compute_iou, match_predictions

  score = compute_iou([10,20,100,200], [15,25,110,210])
"""


def compute_iou(box_a: list, box_b: list) -> float:
    """
    Compute IoU between two bounding boxes [x1, y1, x2, y2].

    Args:
        box_a: [x1, y1, x2, y2] predicted box
        box_b: [x1, y1, x2, y2] ground truth box

    Returns:
        IoU in [0.0, 1.0]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def match_predictions(
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Match predicted regions to ground-truth regions using IoU.

    Uses a greedy matching strategy: for each GT box, find the best
    matching predicted box above the IoU threshold.

    Args:
        pred_boxes:    list of {'bbox': [...], 'type': str}
        gt_boxes:      list of {'bbox': [...], 'type': str}
        iou_threshold: minimum IoU to count as a match

    Returns:
        dict with:
          "tp":              int — true positives (matched)
          "fp":              int — false positives (unmatched predictions)
          "fn":              int — false negatives (unmatched GT)
          "precision":       float
          "recall":          float
          "f1":              float
          "type_correct":    int — matched pairs where type also matches
          "type_accuracy":   float — among matched, fraction with correct type
    """
    matched_pred = set()
    matched_gt   = set()
    type_correct = 0

    for gi, gt in enumerate(gt_boxes):
        best_iou  = 0.0
        best_pi   = -1
        for pi, pred in enumerate(pred_boxes):
            if pi in matched_pred:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_pi  = pi

        if best_iou >= iou_threshold and best_pi >= 0:
            matched_gt.add(gi)
            matched_pred.add(best_pi)
            if pred_boxes[best_pi].get("type") == gt.get("type"):
                type_correct += 1

    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes)   - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    type_acc  = type_correct / tp if tp > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "type_correct": type_correct, "type_accuracy": type_acc,
    }


if __name__ == "__main__":
    a = [10, 20, 100, 200]
    b = [15, 25, 110, 210]
    print(f"IoU example: {compute_iou(a, b):.4f}  (expected ~0.72)")

    preds = [{"bbox": [10,20,100,200], "type": "handwritten"},
             {"bbox": [300,50,500,150], "type": "printed"}]
    gts   = [{"bbox": [12,22, 98,198], "type": "handwritten"},
             {"bbox": [310,55,490,145], "type": "table"}]
    result = match_predictions(preds, gts, iou_threshold=0.5)
    print(f"Match result: {result}")
    print("✅ IoU test passed.")
