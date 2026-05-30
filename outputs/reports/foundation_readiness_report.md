# Foundation Readiness Report

**Overall Status: FAIL 16 CHECKS FAILING**

## Checklist

- [PASS] **01.** configs/data.yaml exists
- [PASS] **02.** configs/detector.yaml exists
- [PASS] **03.** configs/recognizer.yaml exists
- [PASS] **04.** src/data/load_dataset.py exists
- [PASS] **05.** src/data/inspect_dataset.py exists
- [FAIL] **06.** data/processed/dataset_stats.json exists
      FIX: python -m src.data.dataset_stats --from-jsonl data/processed/train_raw.jsonl
- [FAIL] **07.** outputs/reports/dataset_stats_train.md exists
      FIX: python -m src.data.dataset_stats
- [FAIL] **08.** data/processed/train_split.jsonl exists & non-empty
      FIX: python -m src.data.make_splits
- [FAIL] **09.** data/processed/val_split.jsonl exists & non-empty
      FIX: python -m src.data.make_splits
- [PASS] **10.** src/data/bbox_validation.py exists
- [FAIL] **11.** outputs/reports/bbox_validation_report.md exists
      FIX: python -m src.data.bbox_validation --from-jsonl data/processed/train_split.jsonl
- [PASS] **12.** src/visualization/draw_bboxes.py exists
- [FAIL] **13.** outputs/debug_images/ has annotation images
      FIX: python -m src.visualization.draw_bboxes --n-samples 10
- [FAIL] **14.** data/crops/train/images/ has crop images
      FIX: python -m src.recognition.prepare_crops --split train
- [FAIL] **15.** data/crops/train/labels.csv exists & non-empty
      FIX: python -m src.recognition.prepare_crops --split train
- [FAIL] **16.** data/crops/val/labels.csv exists & non-empty
      FIX: python -m src.recognition.prepare_crops --split val
- [FAIL] **17.** outputs/reports/crop_validation_report_train.md exists
      FIX: python -m src.recognition.validate_crops --split train
- [FAIL] **18.** data/yolo/images/train/ has images
      FIX: python -m src.detection.convert_to_yolo
- [FAIL] **19.** data/yolo/labels/train/ has .txt files
      FIX: python -m src.detection.convert_to_yolo
- [FAIL] **20.** data/yolo/data.yaml exists
      FIX: python -m src.detection.convert_to_yolo
- [FAIL] **21.** outputs/reports/yolo_validation_report.md exists
      FIX: python -m src.detection.validate_yolo
- [FAIL] **22.** data/submissions/ has an empty baseline CSV
      FIX: python -m src.submission.build_empty_submission
- [FAIL] **23.** outputs/reports/submission_validation_report.md exists
      FIX: python -m src.submission.validate_submission --csv <your_csv>
- [PASS] **24.** src/evaluation/cer.py exists
- [PASS] **25.** src/evaluation/iou.py exists
- [PASS] **26.** src/postprocessing/reading_order.py exists

---

## Action Required

Fix the failing checks above before starting training.
Each ❌ item includes a FIX command.
