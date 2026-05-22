"""
Verify dataset IDs vs fixed train/val/test split files.
Checks which split IDs exist in imagesTr, and flags missing/extra.
"""

import os
import re
from pathlib import Path

# Standard nnUNetv2 environment variable
RAW        = Path(os.environ["nnUNet_raw"])
# Split txt files live in dataset/splits/ relative to repo root
SPLITS_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "splits"

DATASETS = {
    "anterior": {
        "images":    RAW / "Dataset001_BoneScanAnterior" / "imagesTr",
        "prefix":    "Anterior",
        "split_dir": SPLITS_DIR / "anterior",
    },
    "posterior": {
        "images":    RAW / "Dataset002_BoneScanPosterior" / "imagesTr",
        "prefix":    "Posterior",
        "split_dir": SPLITS_DIR / "posterior",
    },
}


def extract_id(filename: str) -> int:
    """Anterior_0001_0000.png → 1"""
    m = re.search(r"_(\d+)_\d+\.", filename)
    return int(m.group(1)) if m else None


def load_split(path: Path) -> set[int]:
    return {int(line.strip()) for line in path.read_text().splitlines() if line.strip()}


def report(view: str, cfg: dict):
    print(f"\n{'='*50}")
    print(f"  VIEW: {view.upper()}")
    print(f"{'='*50}")

    # IDs present in imagesTr
    files = list(cfg["images"].glob("*.png"))
    available = {extract_id(f.name) for f in files} - {None}

    # Load splits
    train = load_split(cfg["split_dir"] / "train.txt")
    val   = load_split(cfg["split_dir"] / "val.txt")
    test  = load_split(cfg["split_dir"] / "test.txt")
    all_split = train | val | test

    # Overlap check
    tv_overlap  = train & val
    tt_overlap  = train & test
    vt_overlap  = val   & test

    # Match stats
    train_present = train & available
    val_present   = val   & available
    test_present  = test  & available

    train_missing = train - available
    val_missing   = val   - available
    test_missing  = test  - available

    extra = available - all_split  # in imagesTr but NOT in any split

    print(f"\n{'--- COUNTS':}")
    print(f"  {'imagesTr files':<30} {len(available)}")
    print(f"  {'split train IDs':<30} {len(train)}")
    print(f"  {'split val IDs':<30} {len(val)}")
    print(f"  {'split test IDs':<30} {len(test)}")
    print(f"  {'split total IDs':<30} {len(all_split)}")

    print(f"\n{'--- SPLIT OVERLAP (should all be 0)':}")
    print(f"  {'train ∩ val':<30} {len(tv_overlap)}", "✅" if not tv_overlap else f"❌ {sorted(tv_overlap)[:5]}")
    print(f"  {'train ∩ test':<30} {len(tt_overlap)}", "✅" if not tt_overlap else f"❌ {sorted(tt_overlap)[:5]}")
    print(f"  {'val ∩ test':<30} {len(vt_overlap)}", "✅" if not vt_overlap else f"❌ {sorted(vt_overlap)[:5]}")

    print(f"\n{'--- AVAILABILITY (split IDs present in imagesTr)':}")
    print(f"  {'train present':<30} {len(train_present)}/{len(train)}", "✅" if not train_missing else f"⚠️  {len(train_missing)} missing")
    print(f"  {'val present':<30} {len(val_present)}/{len(val)}", "✅" if not val_missing else f"⚠️  {len(val_missing)} missing")
    print(f"  {'test present':<30} {len(test_present)}/{len(test)}", "✅" if not test_missing else f"⚠️  {len(test_missing)} missing")

    print(f"\n{'--- EXTRA (in imagesTr but not in any split)':}")
    print(f"  {'extra files':<30} {len(extra)}", "✅" if not extra else f"⚠️  {sorted(extra)[:10]}")

    print(f"\n{'--- FOR 5-FOLD CV (train+val pool)':}")
    trainval = train_present | val_present
    print(f"  {'usable train+val cases':<30} {len(trainval)}")
    print(f"  {'usable test cases':<30} {len(test_present)}")
    print(f"  {'~cases per fold (val)':<30} {len(trainval) // 5}")


for view, cfg in DATASETS.items():
    report(view, cfg)

print("\n" + "="*50)
print("  DONE")
print("="*50)
