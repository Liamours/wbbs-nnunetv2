"""
Generate splits_final.json for nnUNetv2 5-fold CV.
Pool = train_present ∪ val_present from original fixed split.
Test cases excluded even if present in imagesTr.
Seed fixed for reproducibility.
"""

import json
import re
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

# Standard nnUNetv2 environment variables
RAW        = Path(os.environ["nnUNet_raw"])
PRE        = Path(os.environ["nnUNet_preprocessed"])
# Split txt files live in dataset/splits/ relative to repo root
SPLITS_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "splits"
SEED       = 42
N_FOLDS    = 5

DATASETS = {
    "anterior": {
        "images":    RAW / "Dataset001_BoneScanAnterior" / "imagesTr",
        "prefix":    "Anterior",
        "split_dir": SPLITS_DIR / "anterior",
        "out_dir":   PRE / "Dataset001_BoneScanAnterior",
    },
    "posterior": {
        "images":    RAW / "Dataset002_BoneScanPosterior" / "imagesTr",
        "prefix":    "Posterior",
        "split_dir": SPLITS_DIR / "posterior",
        "out_dir":   PRE / "Dataset002_BoneScanPosterior",
    },
}


def extract_id(filename: str) -> int:
    m = re.search(r"_(\d+)_\d+\.", filename)
    return int(m.group(1)) if m else None


def load_split(path: Path) -> set[int]:
    return {int(l.strip()) for l in path.read_text().splitlines() if l.strip()}


def id_to_case(prefix: str, numeric_id: int) -> str:
    return f"{prefix}_{numeric_id:04d}"


for view, cfg in DATASETS.items():
    print(f"\n{'='*50}")
    print(f"  {view.upper()}")
    print(f"{'='*50}")

    files     = list(cfg["images"].glob("*.png"))
    available = {extract_id(f.name) for f in files} - {None}

    train_ids = load_split(cfg["split_dir"] / "train.txt")
    val_ids   = load_split(cfg["split_dir"] / "val.txt")
    test_ids  = load_split(cfg["split_dir"] / "test.txt")

    pool = sorted((train_ids | val_ids) & available)
    test_present = sorted(test_ids & available)

    print(f"  Train+val pool : {len(pool)}")
    print(f"  Test excluded  : {len(test_present)}")

    case_ids = np.array([id_to_case(cfg["prefix"], i) for i in pool])

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(case_ids)):
        fold_train = sorted(case_ids[train_idx].tolist())
        fold_val   = sorted(case_ids[val_idx].tolist())
        splits.append({"train": fold_train, "val": fold_val})
        print(f"  Fold {fold_idx}: train={len(fold_train)}, val={len(fold_val)}")

    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    out_path = cfg["out_dir"] / "splits_final.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  Saved → {out_path}")

print("\nDone. Run preprocessing next.")
