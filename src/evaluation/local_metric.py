"""
src/evaluation/local_metric.py  —  Composite local evaluation metric.

WHY A COMPOSITE METRIC:
  The competition judges submissions on multiple dimensions:
    1. Did we find the right regions?  (detection F1 at IoU ≥ 0.5)
    2. Did we classify them correctly? (type accuracy)
    3. Did we read the text correctly? (CER on scorable regions)

  This module combines those into a single evaluation you can run
  locally against your validation split to track progress without
  submitting to Kaggle every time.

  NOTE: This is an approximation of the official metric. The exact
  Kaggle scoring formula may differ. Use this for relative comparison
  between experiments, not as an absolute score.

USAGE:
  python -m src.evaluation.local_metric \
      --pred data/submissions/my_pred.csv \
      --gt   data/processed/val_split.jsonl
"""

import argparse
import json
import csv
from pathlib import Path

from src.evaluation.cer import compute_batch_cer
from src.evaluation.iou import match_predictions
from src.utils.paths import DATA_CFG


NON_TEXT = set(DATA_CFG.get("non_text_types", ["image", "graph"]))


def load_submission_csv(path: str | Path) -> dict[str, list[dict]]:
    """
    Load a submission CSV and return a dict: filename → list of regions.

    Args:
        path: path to submission CSV (image, regions columns)

    Returns:
        {file_name: [{"bbox":…, "type":…, "text":…}, …]}
    """
    result = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname   = row["image"]
            regions = json.loads(row["regions"])
            result[fname] = regions
    return result


def load_gt_from_jsonl(path: str | Path) -> dict[str, list[dict]]:
    """
    Load ground-truth regions from a JSONL split file.

    Returns:
        {file_name: [region_dict, …]}
    """
    from src.utils.jsonl import read_jsonl
    records = read_jsonl(path)
    return {rec["file_name"]: rec.get("regions", []) for rec in records}


def evaluate(
    pred_map: dict[str, list[dict]],
    gt_map:   dict[str, list[dict]],
    iou_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Run full local evaluation including Page CER and Official Score.
    """
    from src.postprocessing.reading_order import sort_regions
    from src.postprocessing.normalize_text import normalize

    all_tp = all_fp = all_fn = 0
    all_type_correct = all_type_total = 0
    
    # For Region-level CER
    all_pred_texts, all_gt_texts = [], []
    
    # For Page-level CER
    page_cers = []

    for fname, gt_regions in gt_map.items():
        pred_regions = pred_map.get(fname, [])

        # 1. Detection + type accuracy
        match = match_predictions(pred_regions, gt_regions, iou_threshold)
        all_tp           += match["tp"]
        all_fp           += match["fp"]
        all_fn           += match["fn"]
        all_type_correct += match["type_correct"]
        all_type_total   += match["tp"]

        # 2. Region CER (on matched scorable regions)
        for gt_r in gt_regions:
            if (gt_r.get("language") != "uk"
                    or gt_r.get("legibility") != "legible"
                    or gt_r.get("type") in NON_TEXT
                    or not gt_r.get("text", "").strip()):
                continue

            gt_bbox = gt_r["bbox"]
            gt_text = normalize(gt_r["text"])

            best_iou  = 0.0
            best_text = ""
            for pr in pred_regions:
                iou = 0.0
                try:
                    from src.evaluation.iou import compute_iou
                    iou = compute_iou(pr["bbox"], gt_bbox)
                except Exception:
                    pass
                if iou > best_iou:
                    best_iou  = iou
                    best_text = normalize(pr.get("text", ""))

            all_gt_texts.append(gt_text)
            all_pred_texts.append(best_text if best_iou >= iou_threshold else "")

        # 3. Page CER
        # Sort both by reading order and join
        sorted_gt = sort_regions(gt_regions)
        sorted_pr = sort_regions(pred_regions)
        
        gt_page_text = " ".join(normalize(r.get("text", "")) for r in sorted_gt if r.get("type") not in NON_TEXT)
        pr_page_text = " ".join(normalize(r.get("text", "")) for r in sorted_pr if r.get("type") not in NON_TEXT)
        
        if gt_page_text.strip():
            page_cer = compute_cer(pr_page_text, gt_page_text)
            page_cers.append(page_cer)

    # Aggregate metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    det_f1    = (2*precision*recall / (precision+recall)
                 if (precision+recall) > 0 else 0.0)
    type_acc  = all_type_correct / all_type_total if all_type_total > 0 else 0.0

    cer_result = compute_batch_cer(all_pred_texts, all_gt_texts)
    region_cer = cer_result["corpus_cer"]
    
    mean_page_cer = sum(page_cers) / len(page_cers) if page_cers else 0.0

    # Official Score Formula:
    # Score = 0.15 * Detection_F1 + 0.05 * ClassAcc + 0.30 * (1 - CER) + 0.50 * (1 - PageCER)
    official_score = (
        0.15 * det_f1 +
        0.05 * type_acc +
        0.30 * max(0, 1 - region_cer) +
        0.50 * max(0, 1 - mean_page_cer)
    )

    results = {
        "detection_f1":        det_f1,
        "type_accuracy":       type_acc,
        "region_cer":          region_cer,
        "page_cer":            mean_page_cer,
        "official_score":      official_score,
        "num_images":          len(gt_map),
        "num_scorable":        len(all_gt_texts),
    }

    if verbose:
        print("\n" + "="*55)
        print("  LOCAL EVALUATION RESULTS (COMPOSITE)")
        print("="*55)
        print(f"  Images evaluated  : {results['num_images']:,}")
        print(f"  Scorable regions  : {results['num_scorable']:,}")
        print("-" * 55)
        print(f"  1. Detection F1   : {det_f1:.4f}  (weight 0.15)")
        print(f"  2. Type Accuracy  : {type_acc:.4f}  (weight 0.05)")
        print(f"  3. Region CER     : {region_cer:.4f}  (weight 0.30)")
        print(f"  4. Page CER       : {mean_page_cer:.4f}  (weight 0.50)")
        print("-" * 55)
        print(f"  >>> OFFICIAL SCORE: {official_score:.4f} <<<")
        print("="*55)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True,
                    help="Path to submission CSV")
    ap.add_argument("--gt", required=True,
                    help="Path to GT JSONL (val_split.jsonl)")
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    args = ap.parse_args()

    pred_map = load_submission_csv(args.pred)
    gt_map   = load_gt_from_jsonl(args.gt)
    evaluate(pred_map, gt_map, iou_threshold=args.iou_threshold)
