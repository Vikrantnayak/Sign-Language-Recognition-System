"""
Centralized path resolution for Sign Language Recognition System.
Ensures all assets (models, labels, datasets, images) resolve relative to
the project files, regardless of the current working directory.
"""

import os
from pathlib import Path
from typing import List, Optional

# Base directories
SLR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SLR_DIR.parent

MODEL_DIR = SLR_DIR / "model"
UTILS_DIR = SLR_DIR / "utils"
RESOURCES_DIR = PROJECT_ROOT / "resources"
ILLUSTRATIONS_DIR = RESOURCES_DIR / "asl_illustrations"

# Asset file paths
DEFAULT_MODEL_PATH = MODEL_DIR / "slr_model.tflite"
DEFAULT_HDF5_PATH = MODEL_DIR / "slr_model.hdf5"
DEFAULT_LABEL_PATH = MODEL_DIR / "label.csv"
DEFAULT_KEYPOINT_PATH = MODEL_DIR / "keypoint.csv"
DEFAULT_COUNTER_PATH = MODEL_DIR / "counter.json"
DEFAULT_BACKGROUND_PATH = RESOURCES_DIR / "background.png"


def resolve_path(path_str_or_path: Optional[str | Path]) -> Path:
    """
    Resolve a given path. If relative, checks relative to current working directory first,
    then relative to PROJECT_ROOT, then relative to MODEL_DIR, then RESOURCES_DIR, then ILLUSTRATIONS_DIR.
    """
    if path_str_or_path is None:
        return DEFAULT_MODEL_PATH

    p = Path(path_str_or_path)
    if p.is_absolute() and p.exists():
        return p

    # Check relative to cwd
    if p.exists():
        return p.resolve()

    # Check relative to PROJECT_ROOT
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate.resolve()

    # Check relative to MODEL_DIR (if just a filename like 'label.csv')
    candidate_model = MODEL_DIR / p
    if candidate_model.exists():
        return candidate_model.resolve()

    # Check relative to RESOURCES_DIR
    candidate_res = RESOURCES_DIR / p
    if candidate_res.exists():
        return candidate_res.resolve()

    # Check relative to ILLUSTRATIONS_DIR
    candidate_ill = ILLUSTRATIONS_DIR / p
    if candidate_ill.exists():
        return candidate_ill.resolve()

    # Default to PROJECT_ROOT / p even if it doesn't exist yet (for write targets)
    return (PROJECT_ROOT / p).resolve()


def load_labels(label_path: Optional[str | Path] = None) -> List[str]:
    """
    Load gesture labels from label.csv safely.
    Returns list of labels (e.g. ['A', 'B', ...]).
    """
    target = resolve_path(label_path) if label_path else DEFAULT_LABEL_PATH
    if not target.exists():
        # Fallback to default A-Y without J and Z
        return [
            "A", "B", "C", "D", "E", "F", "G", "H", "I",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S",
            "T", "U", "V", "W", "X", "Y"
        ]

    labels = []
    with open(target, mode="r", encoding="utf-8-sig") as f:
        for line in f:
            clean = line.strip()
            if clean:
                labels.append(clean)
    return labels
