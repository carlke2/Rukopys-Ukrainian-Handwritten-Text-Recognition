"""
src/data/dataset_stats.py  —  Stage 6: Generate dataset statistics.

WHY: You need to know the shape of your data before training.
  - Class imbalance → need weighted loss or sampling.
  - Most documents are from one source → val split must be stratified.
  - Short texts dominate → max_seq_length can be tuned.
  - Few scorable regions → know your effective training set size.

A "scorable" region is one that counts toward the competition metric:
  - language = "uk"   (Ukrainian)
  - legibility = "legible"
  - type NOT IN ("image", "graph")  (visual-only regions have no text)
  - text is non-empty

USAGE:
  python -m src.data.dataset_stats
  python -m src.data.dataset_stats --from-jsonl data/processed/train_raw.jsonl
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path

from src.utils.paths import DATA_CFG, ensure_dir, get_path


def compute_stats(records: list[dict]) -> dict:
    """
    Compute statistics over a list of metadata records.

    Args:
        records: list of dicts from JSONL or HF dataset

    Returns:
        stats dict with counts, distributions, and examples
    """
    stats: dict = {
        "num_images": 0,
        "num_regions": 0,
        "regions_by_type": defaultdict(int),
        "images_by_source": defaultdict(int),
        "language_distribution": defaultdict(int),
        "legibility_distribution": defaultdict(int),
        "text_length_distribution": {"0": 0, "1-10": 0, "11-50": 0,
                                      "51-200": 0, "201+": 0},
        "scorable_regions": 0,
        "non_text_types": DATA_CFG.get("non_text_types", ["image", "graph"]),
    }

    non_text = set(stats["non_text_types"])

    for rec in records:
        stats["num_images"] += 1
        source = rec.get("source", "unknown")
        stats["images_by_source"][source] += 1

        regions = rec.get("regions", [])
        if isinstance(regions, str):
            regions = json.loads(regions)

        for r in regions:
            stats["num_regions"] += 1
            rtype = r.get("type", "unknown")
            lang  = r.get("language", "unknown")
            leg   = r.get("legibility", "unknown")
            text  = r.get("text", "")

            stats["regions_by_type"][rtype] += 1
            stats["language_distribution"][lang] += 1
            stats["legibility_distribution"][leg] += 1

            # Text length bucket
            tlen = len(text)
            if tlen == 0:
                stats["text_length_distribution"]["0"] += 1
            elif tlen <= 10:
                stats["text_length_distribution"]["1-10"] += 1
            elif tlen <= 50:
                stats["text_length_distribution"]["11-50"] += 1
            elif tlen <= 200:
                stats["text_length_distribution"]["51-200"] += 1
            else:
                stats["text_length_distribution"]["201+"] += 1

            # Scorable check
            if (
                lang == "uk"
                and leg == "legible"
                and rtype not in non_text
                and text.strip()
            ):
                stats["scorable_regions"] += 1

    # Convert defaultdicts to plain dicts for JSON serialisation
    stats["regions_by_type"]        = dict(stats["regions_by_type"])
    stats["images_by_source"]       = dict(stats["images_by_source"])
    stats["language_distribution"]  = dict(stats["language_distribution"])
    stats["legibility_distribution"]= dict(stats["legibility_distribution"])

    return stats


def print_stats(stats: dict) -> None:
    """Print statistics in a human-readable format."""
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total images          : {stats['num_images']:,}")
    print(f"  Total regions         : {stats['num_regions']:,}")
    print(f"  Scorable regions      : {stats['scorable_regions']:,}")
    pct = (stats['scorable_regions'] / max(stats['num_regions'], 1)) * 100
    print(f"  Scorable %            : {pct:.1f}%")

    print(f"\n  Regions by type:")
    for k, v in sorted(stats["regions_by_type"].items(), key=lambda x: -x[1]):
        print(f"    {k:<14}: {v:,}")

    print(f"\n  Images by source:")
    for k, v in sorted(stats["images_by_source"].items(), key=lambda x: -x[1]):
        print(f"    {k:<14}: {v:,}")

    print(f"\n  Language distribution:")
    for k, v in sorted(stats["language_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {k:<14}: {v:,}")

    print(f"\n  Legibility distribution:")
    for k, v in stats["legibility_distribution"].items():
        print(f"    {k:<14}: {v:,}")

    print(f"\n  Text length distribution (chars):")
    for k, v in stats["text_length_distribution"].items():
        print(f"    {k:<14}: {v:,}")


def save_stats(stats: dict, name: str = "train") -> None:
    """Save stats to JSON and a Markdown report."""
    json_path = get_path("dataset_stats")
    ensure_dir(json_path.parent)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Stats saved → {json_path}")

    # Markdown report
    md_path = ensure_dir(get_path("reports_dir")) / f"dataset_stats_{name}.md"
    lines = [f"# Dataset Statistics — {name}\n"]
    lines.append(f"| Metric | Value |\n|--------|-------|\n")
    lines.append(f"| Total images | {stats['num_images']:,} |\n")
    lines.append(f"| Total regions | {stats['num_regions']:,} |\n")
    lines.append(f"| Scorable regions | {stats['scorable_regions']:,} |\n\n")
    lines.append("## Regions by type\n| Type | Count |\n|------|-------|\n")
    for k, v in sorted(stats["regions_by_type"].items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v:,} |\n")
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"  ✅ Report saved → {md_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-jsonl", type=str, default=None,
                    help="Path to local JSONL file (else loads from HF)")
    ap.add_argument("--split", default="train",
                    choices=["train", "silver", "test"])
    args = ap.parse_args()

    if args.from_jsonl:
        from src.utils.jsonl import read_jsonl
        records = read_jsonl(args.from_jsonl)
    else:
        from src.data.load_dataset import load_split
        ds = load_split(args.split)
        records = list(ds)

    stats = compute_stats(records)
    print_stats(stats)
    save_stats(stats, name=args.split)
    print("\n✅ Done. Next: python -m src.data.make_splits")
