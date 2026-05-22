"""
Run inference for each fold (0-4) and ensemble on test sets.

Outputs per dataset:
    labelsTs_pred_0 ... labelsTs_pred_4   (per-fold predictions)
    labelsTs_pred_e                        (ensemble of all 5 folds)

Requires standard nnUNetv2 environment variables:
    nnUNet_raw      path to nnUNet_raw/
    nnUNet_results  path to nnUNet_results/

Usage:
    python inference_all_folds.py --dataset 1
    python inference_all_folds.py --dataset 2
    python inference_all_folds.py --dataset 1 2
"""

import argparse
import os
import torch
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join

RAW         = Path(os.environ["nnUNet_raw"])
RESULTS     = Path(os.environ["nnUNet_results"])
TRAINER     = "nnUNetTrainer_65epochs__nnUNetPlans__2d"
CHECKPOINT  = "checkpoint_final.pth"
FOLDS       = (0, 1, 2, 3, 4)

DATASET_META = {
    1: "Dataset001_BoneScanAnterior",
    2: "Dataset002_BoneScanPosterior",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def get_predictor():
    return nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=DEVICE,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )


def run_predict(predictor, model_folder, folds, indir, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name=CHECKPOINT,
    )
    predictor.predict_from_files(
        str(indir),
        str(outdir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


def process_dataset(dataset_id):
    ds_name     = DATASET_META[dataset_id]
    model_folder = str(RESULTS / ds_name / TRAINER)
    indir        = RAW / ds_name / "imagesTs"

    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"Model:   {model_folder}")
    print(f"Input:   {indir}")
    print(f"{'='*60}")

    # --- Per-fold predictions ---
    for fold in FOLDS:
        outdir = RAW / ds_name / f"labelsTs_pred_{fold}"
        print(f"\n[Fold {fold}] -> {outdir.name}")
        predictor = get_predictor()
        run_predict(predictor, model_folder, (fold,), indir, outdir)
        print(f"[Fold {fold}] Done.")

    # --- Ensemble (all folds) ---
    outdir = RAW / ds_name / "labelsTs_pred_e"
    print(f"\n[Ensemble] -> {outdir.name}")
    predictor = get_predictor()
    run_predict(predictor, model_folder, FOLDS, indir, outdir)
    print(f"[Ensemble] Done.")

    print(f"\nAll predictions complete for {ds_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, nargs="+", choices=[1, 2],
                        default=[1, 2], help="Dataset ID(s) to process")
    args = parser.parse_args()

    for ds_id in args.dataset:
        process_dataset(ds_id)


if __name__ == "__main__":
    main()
