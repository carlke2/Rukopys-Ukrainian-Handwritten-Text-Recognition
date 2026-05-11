"""
src/utils/paths.py
──────────────────
Central path resolver for the entire project.

WHY THIS FILE EXISTS:
  Every script needs to know where files live. If we hardcode paths
  everywhere and then move the project to Kaggle or Colab, we must
  edit dozens of files. Instead, all scripts call get_path() from
  here, and only data.yaml ever changes.

USAGE:
  from src.utils.paths import get_path, PROJECT_ROOT

  yolo_dir   = get_path("yolo_root")
  train_imgs = get_path("yolo_train_images")
"""

import os
import yaml
from pathlib import Path

# ── Project root = the directory that CONTAINS src/
# This works whether you run from the project root or from inside src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_name: str = "data") -> dict:
    """
    Load a YAML config file from configs/.

    Args:
        config_name: filename without extension, e.g. "data", "detector"

    Returns:
        dict with all config keys
    """
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected location: configs/{config_name}.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load the data config once at import time so it is always available.
DATA_CFG = load_config("data")


def get_path(key: str, relative: bool = False) -> Path:
    """
    Resolve a path from data.yaml relative to PROJECT_ROOT.

    Args:
        key:      A key from configs/data.yaml, e.g. "yolo_train_images"
        relative: If True, return relative path (useful for YAML files
                  that need paths relative to themselves)

    Returns:
        Resolved Path object

    Raises:
        KeyError if the key is not in data.yaml
    """
    if key not in DATA_CFG:
        raise KeyError(
            f"Path key '{key}' not found in configs/data.yaml.\n"
            f"Available keys: {list(DATA_CFG.keys())}"
        )
    raw = DATA_CFG[key]
    if relative:
        return Path(raw)
    return PROJECT_ROOT / raw


def ensure_dir(path: Path) -> Path:
    """
    Create a directory (and all parents) if it does not already exist.
    Returns the same path so it can be used inline.

    Example:
        out = ensure_dir(get_path("debug_images_dir")) / "sample.jpg"
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_region_types() -> dict:
    """
    Return the full region_types dict from data.yaml.

    Example returned structure:
        {
          "handwritten": {"id": 0, "color": [0,255,0], "text_bearing": True},
          "printed":     {"id": 1, "color": [0,0,255], "text_bearing": True},
          ...
        }
    """
    return DATA_CFG.get("region_types", {})


def get_class_id(region_type: str) -> int:
    """Return the integer class id for a region type string."""
    rt = get_region_types()
    if region_type not in rt:
        raise ValueError(
            f"Unknown region type: '{region_type}'. "
            f"Valid types: {list(rt.keys())}"
        )
    return rt[region_type]["id"]


def get_id_to_class() -> dict:
    """Return a mapping from integer class id → region type string."""
    return {v["id"]: k for k, v in get_region_types().items()}


def get_class_to_id() -> dict:
    """Return a mapping from region type string → integer class id."""
    return {k: v["id"] for k, v in get_region_types().items()}


# ── Quick sanity check when run directly ─────────────────────────
if __name__ == "__main__":
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"yolo_root    : {get_path('yolo_root')}")
    print(f"train_split  : {get_path('train_split')}")
    print(f"class map    : {get_class_to_id()}")
