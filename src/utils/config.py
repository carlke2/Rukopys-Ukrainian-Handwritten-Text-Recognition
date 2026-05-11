"""
src/utils/config.py
───────────────────
Convenience wrappers for loading project configs.

WHY THIS FILE EXISTS:
  paths.py handles path resolution. This file is the partner module
  that handles reading the *other* config files (detector.yaml,
  recognizer.yaml) and exposing their values cleanly.

  Having this separate from paths.py keeps paths.py focused on path
  resolution and this module focused on config content.

USAGE:
  from src.utils.config import get_detector_cfg, get_recognizer_cfg

  det_cfg = get_detector_cfg()
  print(det_cfg["model"])        # e.g. "yolov8m.pt"
  print(det_cfg["imgsz"])        # e.g. 1024

  rec_cfg = get_recognizer_cfg()
  print(rec_cfg["base_model"])   # e.g. "microsoft/trocr-base-handwritten"
"""

from src.utils.paths import load_config, DATA_CFG


def get_data_cfg() -> dict:
    """
    Return the full data config dict (already loaded from data.yaml).

    This is a convenience alias — DATA_CFG from paths.py is the same object.
    Useful when you want to import config without importing paths directly.

    Returns:
        dict with all keys from configs/data.yaml
    """
    return DATA_CFG


def get_detector_cfg() -> dict:
    """
    Load and return configs/detector.yaml.

    Key fields:
      model       — base YOLO model filename (e.g. "yolov8m.pt")
      imgsz       — input image size for training
      epochs      — number of training epochs
      batch       — batch size per GPU
      yolo_data   — path to data/yolo/data.yaml

    Returns:
        dict from configs/detector.yaml
    """
    return load_config("detector")


def get_recognizer_cfg() -> dict:
    """
    Load and return configs/recognizer.yaml.

    Key fields:
      base_model    — HuggingFace model ID for the recognizer
      img_h         — crop height fed to the model
      img_w         — crop width fed to the model
      batch_size    — training batch size
      epochs        — number of training epochs
      train_csv     — path to crops/train/labels.csv
      val_csv       — path to crops/val/labels.csv

    Returns:
        dict from configs/recognizer.yaml
    """
    return load_config("recognizer")


def describe_configs() -> None:
    """
    Print a human-readable summary of all loaded configs.
    Useful for debugging configuration at the start of a training script.
    """
    print("\n" + "=" * 55)
    print("  PROJECT CONFIGURATION SUMMARY")
    print("=" * 55)

    data_cfg = get_data_cfg()
    print(f"\n  [data.yaml]")
    print(f"    Dataset        : {data_cfg.get('hf_dataset_id')}")
    print(f"    Val fraction   : {data_cfg.get('val_fraction')}")
    print(f"    Random seed    : {data_cfg.get('random_seed')}")
    print(f"    Num classes    : {data_cfg.get('num_classes')}")
    print(f"    Region types   : {list(data_cfg.get('region_types', {}).keys())}")

    try:
        det_cfg = get_detector_cfg()
        print(f"\n  [detector.yaml]")
        print(f"    Model          : {det_cfg.get('model', 'not set')}")
        print(f"    Image size     : {det_cfg.get('imgsz', 'not set')}")
        print(f"    Epochs         : {det_cfg.get('epochs', 'not set')}")
        print(f"    Batch size     : {det_cfg.get('batch', 'not set')}")
    except FileNotFoundError:
        print(f"\n  [detector.yaml] — NOT FOUND (create configs/detector.yaml)")

    try:
        rec_cfg = get_recognizer_cfg()
        print(f"\n  [recognizer.yaml]")
        print(f"    Base model     : {rec_cfg.get('base_model', 'not set')}")
        print(f"    Input size     : {rec_cfg.get('img_w')}x{rec_cfg.get('img_h')}")
        print(f"    Epochs         : {rec_cfg.get('epochs', 'not set')}")
        print(f"    Batch size     : {rec_cfg.get('batch_size', 'not set')}")
    except FileNotFoundError:
        print(f"\n  [recognizer.yaml] — NOT FOUND (create configs/recognizer.yaml)")

    print("=" * 55)


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    describe_configs()
    print("\n✅ Config loader OK.")
