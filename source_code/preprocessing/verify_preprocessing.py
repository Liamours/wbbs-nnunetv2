"""
Verify nnUNetv2 preprocessing output for both datasets.
Checks: required files, splits_final.json (5 folds), plans config, case counts.
"""

import json
import os
from pathlib import Path

# Standard nnUNetv2 environment variables
PRE = Path(os.environ["nnUNet_preprocessed"])
RAW = Path(os.environ["nnUNet_raw"])

DATASETS = {
    "Dataset001_BoneScanAnterior":  {"expected_cases": 1748, "view": "Anterior"},
    "Dataset002_BoneScanPosterior": {"expected_cases": 1664, "view": "Posterior"},
}

# Expected from paper
EXPECTED_PLANS = {
    "batch_size":          48,
    "patch_size":          [512, 128],
    "n_stages":            7,
    "features_per_stage":  [32, 64, 128, 256, 512, 512, 512],
    "norm_op":             "InstanceNorm2d",
    "nonlin":              "LeakyReLU",
}

all_ok = True

for ds_name, cfg in DATASETS.items():
    ds_dir = PRE / ds_name
    print(f"\n{'='*55}")
    print(f"  {ds_name}")
    print(f"{'='*55}")

    # --- Required files ---
    required = ["dataset.json", "dataset_fingerprint.json", "nnUNetPlans.json"]
    for f in required:
        exists = (ds_dir / f).exists()
        print(f"  {'[OK]' if exists else '[MISSING]'} {f}")
        if not exists:
            all_ok = False

    # --- splits_final.json ---
    splits_path = ds_dir / "splits_final.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        n_folds = len(splits)
        fold_ok = n_folds == 5
        print(f"  {'[OK]' if fold_ok else '[FAIL]'} splits_final.json — {n_folds} folds (need 5)")
        if not fold_ok:
            all_ok = False
        total_cases = set()
        for i, fold in enumerate(splits):
            t, v = len(fold["train"]), len(fold["val"])
            overlap = set(fold["train"]) & set(fold["val"])
            overlap_ok = len(overlap) == 0
            print(f"         Fold {i}: train={t}, val={v}, overlap={'0 [OK]' if overlap_ok else str(len(overlap))+' [FAIL]'}")
            if not overlap_ok:
                all_ok = False
            total_cases.update(fold["train"])
            total_cases.update(fold["val"])
        count_ok = len(total_cases) == cfg["expected_cases"]
        print(f"         Total unique cases: {len(total_cases)} (expected {cfg['expected_cases']}) {'[OK]' if count_ok else '[FAIL]'}")
        if not count_ok:
            all_ok = False
    else:
        print(f"  [INFO]  splits_final.json not yet generated (created on first train run)")

    # --- Plans config check ---
    plans_path = ds_dir / "nnUNetPlans.json"
    if plans_path.exists():
        plans = json.loads(plans_path.read_text())
        cfg_2d = plans.get("configurations", {}).get("2d", {})
        arch   = cfg_2d.get("architecture", {}).get("arch_kwargs", {})

        checks = {
            "batch_size":         (cfg_2d.get("batch_size"), EXPECTED_PLANS["batch_size"]),
            "patch_size":         (cfg_2d.get("patch_size"), EXPECTED_PLANS["patch_size"]),
            "n_stages":           (arch.get("n_stages"),     EXPECTED_PLANS["n_stages"]),
            "features_per_stage": (list(arch.get("features_per_stage", [])), EXPECTED_PLANS["features_per_stage"]),
        }
        print(f"\n  Plans config vs paper:")
        for key, (got, want) in checks.items():
            ok = got == want
            print(f"    {'[OK]' if ok else '[FAIL]'} {key}: {got} (expected {want})")
            if not ok:
                all_ok = False

    # --- Preprocessed case count ---
    # nnUNetv2 uses .b2nd format (blosc2), not .npz
    data_dir = ds_dir / "nnUNetPlans_2d"
    if data_dir.exists():
        pkl_files = list(data_dir.glob("*.pkl"))  # one .pkl per case
        count_ok  = len(pkl_files) == cfg["expected_cases"]
        print(f"\n  Preprocessed cases (.pkl): {len(pkl_files)} (expected {cfg['expected_cases']}) {'[OK]' if count_ok else '[FAIL]'}")
        if not count_ok:
            all_ok = False
    else:
        print(f"\n  [MISSING] nnUNetPlans_2d/ folder — preprocessing may have failed")
        all_ok = False

    # --- splits_final.json note ---
    if not splits_path.exists():
        print(f"\n  [NOTE] splits_final.json absent — generated on first training run. Normal.")

print(f"\n{'='*55}")
print(f"  OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED — review above'}")
print(f"{'='*55}\n")
