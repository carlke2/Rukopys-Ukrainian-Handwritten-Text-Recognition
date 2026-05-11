# Handwritten-to-Data · Kaggle AI Challenge

> **Task:** Given a Ukrainian document image (handwritten, printed, or mixed),
> detect all content regions, classify each region type, and transcribe the
> Ukrainian text inside every readable region.

---

## Project Purpose

This project builds a complete AI pipeline for the
**[RUKOPYS](https://huggingface.co/datasets/UkrainianCatholicUniversity/rukopys)**
dataset — a Kaggle competition focused on Ukrainian handwritten document AI.

The goal is to produce a structured JSON prediction for every test document page:

```json
[
  {"bbox": [50, 100, 850, 130], "type": "handwritten", "text": "Доброго ранку"},
  {"bbox": [50, 150, 600, 300], "type": "table",       "text": "..."},
  {"bbox": [20,  80, 400, 120], "type": "image",       "text": ""}
]
```

---

## What Is a "Region"?

A region is an important rectangular area on a document page.

| Type | Meaning | Has Text? |
|------|---------|-----------|
| `handwritten` | Handwritten text | ✅ |
| `printed` | Printed/typed text | ✅ |
| `formula` | Mathematical formula | ✅ |
| `table` | Data table | ✅ |
| `annotation` | Margin note or stamp | ✅ |
| `image` | Embedded photo or drawing | ❌ |
| `graph` | Chart or diagram | ❌ |

**Class IDs** (for detector): handwritten=0, printed=1, formula=2, table=3, annotation=4, image=5, graph=6

---

## Dataset Source

**Hugging Face:** `UkrainianCatholicUniversity/rukopys`

| Split | Content | Use |
|-------|---------|-----|
| `train` | Human-verified annotations | ✅ Train & validate |
| `silver` | Auto-generated annotations (noisy) | 🔜 Later self-training only |
| `test` | Images only — no ground truth | 🔒 Never train on this |
| `sample_submission.csv` | Format guide only | ⚠️ NOT training data |

> **Rule:** `sample_submission.csv` shows the expected CSV format. It is not a
> source of labels. Do not train on it.

---

## Final Pipeline Architecture

```
Document Image
  │
  ▼
Image Preprocessing          ← grayscale, contrast, denoise, deskew
  │
  ▼
Region Detector (YOLO)       ← finds bbox + type for every region
  │
  ▼
Region Classifier            ← confirms or refines type (future)
  │
  ▼
Crop Each Region             ← extract sub-image per bbox
  │
  ▼
Text Recognizer (TrOCR)      ← reads Ukrainian text from each crop
  │
  ▼
Text Normalization           ← collapse whitespace, fix Unicode
  │
  ▼
Reading Order Sort           ← top-to-bottom, left-to-right
  │
  ▼
Build Prediction JSON        ← [{bbox, type, text}, ...]
  │
  ▼
Generate submission.csv      ← Kaggle upload format
```

---

## Project Folder Structure

```
handwritten-data/
├── README.md
├── requirements.txt           ← foundation packages (install first)
├── requirements-training.txt  ← ML packages (install only at training)
├── .gitignore
│
├── configs/
│   ├── data.yaml              ← all paths, class map, dataset settings
│   ├── detector.yaml          ← YOLO training settings
│   └── recognizer.yaml        ← TrOCR training settings
│
├── data/
│   ├── raw/                   ← original downloaded data (not in git)
│   │   ├── train/images/
│   │   ├── test/images/
│   │   └── sample_submission.csv
│   ├── processed/             ← train_split.jsonl, val_split.jsonl, stats
│   ├── crops/                 ← cropped region images for recognizer
│   ├── yolo/                  ← YOLO-format detection dataset
│   └── submissions/           ← generated CSV submissions
│
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_visualize_annotations.ipynb
│   ├── 03_prepare_crops.ipynb
│   ├── 04_prepare_yolo.ipynb
│   └── 05_local_validation.ipynb
│
├── src/
│   ├── data/                  ← loading, inspection, stats, splits, validation
│   ├── visualization/         ← bbox drawing, contact sheets
│   ├── preprocessing/         ← image cleaning (grayscale, contrast, deskew)
│   ├── recognition/           ← crop preparation and validation
│   ├── detection/             ← YOLO conversion and validation
│   ├── evaluation/            ← CER, IoU, local metric
│   ├── postprocessing/        ← text normalization, reading order
│   ├── submission/            ← build and validate CSVs
│   └── utils/                 ← paths, jsonl, config helpers
│
└── outputs/
    ├── debug_images/          ← annotation overlays, crops, preprocessing
    ├── reports/               ← validation and readiness reports
    └── logs/                  ← training logs (future)
```

---

## Installation

### Step 1 — Create virtual environment

```bash
cd handwritten-data
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 2 — Install foundation packages

```bash
pip install -r requirements.txt
```

> ⚠️ Do **not** install `requirements-training.txt` until all foundation
> readiness gates pass.

---

## Execution Order (Step by Step)

Run these commands **in order** from the project root:

```bash
# ── 1. Inspect dataset structure (no download needed) ────────────
python -m src.data.inspect_dataset

# ── 2. Load & cache train split from Hugging Face ────────────────
python -m src.data.load_dataset --splits train test --save-jsonl

# ── 3. Visualize annotations (saves debug PNGs) ──────────────────
python -m src.visualization.draw_bboxes --from-jsonl data/processed/train_raw.jsonl

# ── 4. Generate dataset statistics ───────────────────────────────
python -m src.data.dataset_stats --from-jsonl data/processed/train_raw.jsonl

# ── 5. Create train / validation split ───────────────────────────
python -m src.data.make_splits --from-jsonl data/processed/train_raw.jsonl

# ── 6. Validate all bounding boxes ───────────────────────────────
python -m src.data.bbox_validation

# ── 7. Prepare recognition crops ─────────────────────────────────
python -m src.recognition.prepare_crops --split train
python -m src.recognition.prepare_crops --split val

# ── 8. Validate recognition crops ────────────────────────────────
python -m src.recognition.validate_crops --split train
python -m src.recognition.validate_crops --split val

# ── 9. Convert to YOLO detection format ──────────────────────────
python -m src.detection.convert_to_yolo

# ── 10. Validate YOLO dataset ────────────────────────────────────
python -m src.detection.validate_yolo

# ── 11. Build empty baseline submission ──────────────────────────
python -m src.submission.build_empty_submission

# ── 12. Validate the submission CSV ──────────────────────────────
python -m src.submission.validate_submission --csv data/submissions/empty_*.csv

# ── 13. Run foundation readiness gate ────────────────────────────
python -m src.data.foundation_readiness

# ── 14. Smoke-test utilities ──────────────────────────────────────
python -m src.evaluation.cer
python -m src.evaluation.iou
python -m src.postprocessing.reading_order
python -m src.postprocessing.normalize_text
python -m src.preprocessing.image_cleaning <any_image_path>

# ── 15. Only after ALL gates pass ────────────────────────────────
pip install -r requirements-training.txt
# → Then follow configs/detector.yaml and configs/recognizer.yaml
```

---

## How to Inspect the Dataset

```python
from src.data.load_dataset import load_split
ds = load_split("train")
print(ds[0]["file_name"])
print(ds[0]["regions"][:2])
```

Or stream without downloading:
```python
from src.data.load_dataset import stream_split
for sample in stream_split("train"):
    print(sample["file_name"])
    break
```

---

## How to Visualize Annotations

```bash
python -m src.visualization.draw_bboxes --n-samples 15
```

Output: `outputs/debug_images/` — one PNG per document with colored boxes per region type.

---

## How to Generate Statistics

```bash
python -m src.data.dataset_stats --from-jsonl data/processed/train_raw.jsonl
```

Output:
- `data/processed/dataset_stats.json`
- `outputs/reports/dataset_stats_train.md`

---

## How to Create Train/Val Split

```bash
python -m src.data.make_splits --from-jsonl data/processed/train_raw.jsonl
```

- **85%** → `data/processed/train_split.jsonl`
- **15%** → `data/processed/val_split.jsonl`
- Stratified by document `source` to preserve diversity.

> **Why validation?** The Kaggle leaderboard is slow feedback. Local validation
> lets you compare experiments instantly and detect overfitting.

---

## How to Prepare Recognition Crops

```bash
python -m src.recognition.prepare_crops --split train
python -m src.recognition.prepare_crops --split val
```

Only keeps regions where: language=`uk`, legibility=`legible`,
type NOT IN `image`/`graph`, text non-empty, bbox valid.

Output: `data/crops/train/images/` + `data/crops/train/labels.csv`

---

## How to Prepare YOLO Data

```bash
python -m src.detection.convert_to_yolo
```

Converts all 7 region types to YOLO normalized format. Both `image` and `graph`
regions are kept (they need to be *detected* even if they have no text).

Output: `data/yolo/images/train/`, `data/yolo/labels/train/`, `data/yolo/data.yaml`

---

## How to Validate Submission

```bash
python -m src.submission.validate_submission --csv data/submissions/my_sub.csv
```

Checks: correct columns, valid JSON, all required fields, bbox sanity, type validity.

---

## How to Create Empty Baseline Submission

```bash
python -m src.submission.build_empty_submission
```

Produces a valid CSV with empty region lists (`[]`) for every test image.
Scores 0 on the leaderboard but proves the submission loop works end-to-end.

---

## Training Readiness Gates

### Detector Gate — can begin when:

- [x] train/val split exists
- [x] Bbox validation passed (0 errors)
- [x] `data/yolo/images/train/` has images
- [x] `data/yolo/labels/train/` has .txt files
- [x] `data/yolo/data.yaml` exists
- [x] YOLO validation report generated
- [x] Annotation debug images visually inspected
- [x] YOLO debug images visually inspected

### Recognizer Gate — can begin when:

- [x] train/val split exists
- [x] `data/crops/train/images/` has crops
- [x] `data/crops/train/labels.csv` non-empty
- [x] `data/crops/val/labels.csv` non-empty
- [x] Crop validation passed (0 errors)
- [x] Random crop contact sheet visually inspected
- [x] CER utility exists and tested
- [x] Recognizer readiness report generated

---

## Local Laptop Notes

The foundation runs comfortably on:
- 8 GB RAM, Core i5, no GPU

**Tips:**
- Use `stream_split("train")` instead of `load_split("train")` if RAM is tight.
- Process crops in batches — `prepare_crops.py` already does this.
- Visualization scripts save to disk — no display needed.
- YOLO conversion copies images by default; use `--no-copy` on Linux to symlink.

---

## Kaggle / Colab Training Notes

- Upload this entire `handwritten-data/` folder to your Kaggle dataset.
- In the notebook: `import sys; sys.path.insert(0, "/kaggle/input/handwritten-data")`
- Set `TRANSFORMERS_CACHE` and `HF_HOME` to `/kaggle/working/` to avoid permission issues.
- YOLO training: `!yolo detect train data=data/yolo/data.yaml model=yolov8m.pt`
- TrOCR fine-tuning: see `configs/recognizer.yaml` for hyperparameters.

---

## Final Inference Constraint

The submission pipeline must use **open-weight models only**.

❌ Do NOT use:
- OpenAI API
- Anthropic API
- Google Gemini API
- Any proprietary closed OCR API

✅ Use:
- YOLOv8/v11 (Ultralytics) — for detection
- TrOCR / PaddleOCR / Surya — for recognition
- Any open-weight model on Hugging Face

The final inference pipeline must fit on a single **NVIDIA H100 80 GB** GPU.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `configs/data.yaml` | All paths, class map, dataset settings |
| `src/utils/paths.py` | Path resolver — import `get_path()` everywhere |
| `src/data/load_dataset.py` | HF dataset loading + streaming |
| `src/data/bbox_validation.py` | Bbox quality gate |
| `src/data/foundation_readiness.py` | Final readiness gate |
| `src/recognition/prepare_crops.py` | Extract crop images + CSV |
| `src/recognition/validate_crops.py` | Validate crop dataset |
| `src/detection/convert_to_yolo.py` | Convert to YOLO format |
| `src/detection/validate_yolo.py` | YOLO dataset quality gate |
| `src/evaluation/cer.py` | Character Error Rate metric |
| `src/evaluation/iou.py` | Intersection over Union |
| `src/postprocessing/reading_order.py` | Sort regions for submission |
| `src/submission/build_empty_submission.py` | Baseline submission |
| `src/submission/validate_submission.py` | Pre-upload CSV validator |
