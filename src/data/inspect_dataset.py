"""
src/data/inspect_dataset.py  —  Stage 4: Inspect dataset structure.

WHY: Before training, you must understand every field in the data.
     Blind training on poorly-understood data is a common failure point.

USAGE:
  python -m src.data.inspect_dataset
  python -m src.data.inspect_dataset --split silver --sample-idx 5
  python -m src.data.inspect_dataset --from-jsonl data/processed/train_raw.jsonl
"""

import json
import argparse


def inspect_sample(sample: dict, sample_idx: int = 0) -> None:
    """Pretty-print all fields from one dataset sample."""
    sep = "=" * 68
    print(f"\n{sep}\n  SAMPLE #{sample_idx}\n{sep}")
    print(f"  file_name        : {sample.get('file_name', 'N/A')}")
    print(f"  image size       : {sample.get('image_width','?')} x "
          f"{sample.get('image_height','?')} px")
    print(f"  source           : {sample.get('source', 'N/A')}")
    print(f"  annotation_source: {sample.get('annotation_source', 'N/A')}")

    if "image" in sample and sample["image"] is not None:
        img = sample["image"]
        print(f"  image object     : PIL Image mode={img.mode} size={img.size}")
    else:
        print("  image object     : (not loaded — JSONL or streaming mode)")

    regions = sample.get("regions", [])
    if isinstance(regions, str):
        regions = json.loads(regions)

    print(f"\n  REGIONS ({len(regions)} total on this page)")

    type_counts: dict = {}
    for r in regions:
        t = r.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  by type: {type_counts}")

    print(f"\n  {'─'*60}\n  FIRST 3 REGIONS (detail)\n  {'─'*60}")
    for i, r in enumerate(regions[:3]):
        bbox  = r.get("bbox", "N/A")
        rtype = r.get("type", "N/A")
        lang  = r.get("language", "N/A")
        leg   = r.get("legibility", "N/A")
        text  = r.get("text", "")

        print(f"\n  [{i+1}] type={rtype}  lang={lang}  legibility={leg}")
        print(f"       bbox={bbox}")

        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            print(f"       size={x2-x1} x {y2-y1} px")

        if text:
            disp = text[:100] + ("…" if len(text) > 100 else "")
            print(f"       text='{disp}'  ({len(text)} chars)")
        else:
            print("       text=(empty)")

        scorable = (
            lang == "uk"
            and leg == "legible"
            and rtype not in ("image", "graph")
            and bool(text.strip())
        )
        print(f"       scorable={'✅ YES' if scorable else '❌ NO'}")

    if len(regions) > 3:
        print(f"\n  … {len(regions)-3} more regions not shown.")


def inspect_from_hf(split: str = "train", sample_idx: int = 0) -> None:
    from src.data.load_dataset import load_split
    ds = load_split(split)
    print(f"\n  Split '{split}': {len(ds):,} samples | columns: {ds.column_names}")
    inspect_sample(ds[sample_idx], sample_idx)


def inspect_from_jsonl(path: str, sample_idx: int = 0) -> None:
    from src.utils.jsonl import read_jsonl
    records = read_jsonl(path)
    print(f"  Loaded {len(records):,} records from {path}")
    inspect_sample(records[min(sample_idx, len(records)-1)], sample_idx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train","silver","test"])
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--from-jsonl", type=str, default=None)
    args = ap.parse_args()

    if args.from_jsonl:
        inspect_from_jsonl(args.from_jsonl, args.sample_idx)
    else:
        inspect_from_hf(args.split, args.sample_idx)

    print("\n✅ Done. Next: python -m src.visualization.draw_bboxes")
