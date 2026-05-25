# Whole-Body Bone Scan Segmentation using nnU-Netv2

Code and data for the paper:  
**"Deep Learning-Based Segmentation of Whole-Body Bone Scan Images Using nnU-Netv2"**  
*International Journal of Intelligent Engineering and Systems (IJIES-INASS)*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20342310.svg)](https://doi.org/10.5281/zenodo.20342310)

> **Annotated masks** are released separately on Zenodo: [10.5281/zenodo.20342310](https://doi.org/10.5281/zenodo.20342310)

---

## Abstract

Bone scintigraphy (BS) is widely used to detect skeletal metastases, but automated anatomical segmentation remains challenging because of image variability, noise, artifacts, and heterogeneous lesions. Existing methods, including FCN, DeepLabv3+, DANet, and SegFormer, have limitations in global context modeling, in handling irregular bone structures and morphological variability, or in posterior-view segmentation. We propose an nnU-Netv2-based approach that self-configures the network topology, preprocessing, and training hyperparameters without manual tuning. Using acquisition-level splitting, 3,593 whole-body BS images from the BS-80K dataset were divided into a 95% development set and a 5% independent hold-out test set. On the development set, nnU-Netv2 achieved 5-fold macro-averaged Dice coefficients of 0.8168 ± 0.0029 for anterior views and 0.8345 ± 0.0018 for posterior views. On the same hold-out test set, nnU-Netv2 achieved the highest mean DSC in the posterior view, whereas SegFormer achieved the highest mean DSC in the anterior view. These results support automated segmentation for Bone Scan Index computation and bone scan analysis.

---

## Dataset

Images are from [BS-80K](https://doi.org/10.1016/j.compbiomed.2022.106221) (Huang et al., 2022).  
3,593 whole-body bone scintigraphy images were manually annotated using IbisPaintX.

| View | Total | Dev | Test |
|---|---|---|---|
| Anterior | 1,841 | 1,748 | 93 |
| Posterior | 1,752 | 1,664 | 88 |

**12 annotated regions** (anterior): skull, cervical vertebrae, thoracic vertebrae, ribs, sternum, clavicle, scapula, humerus, lumbar vertebrae, sacrum, pelvis, femur.  
Posterior excludes sternum (11 regions).  
See `dataset/class_labels.json` for label-to-index mapping.

---

## Splits

- **Dev/test**: 95% development set, 5% independent hold-out test set (fixed, never used during training or cross-validation).
- **5-fold CV**: Applied to the development set only. Each fold uses ~80% for training and ~20% for validation.
- Splitting was done at the acquisition level (anterior/posterior pairs assigned to the same partition).

Split files are in `dataset/splits/anterior/` and `dataset/splits/posterior/`:
- `train.txt`, `val.txt`, `test.txt` — development/test assignment (one image ID per line)
- `splits_5fold.json` — 5-fold cross-validation assignments for nnU-Netv2

---

## Repository Structure

```
dataset/
├── class_labels.json
├── masks/
│   ├── anterior/           # 1,841 ground-truth masks (label index 0-12)
│   ├── anterior_colored/   # same masks, RGB colored for visualization
│   ├── posterior/          # 1,752 ground-truth masks (label index 0-12)
│   └── posterior_colored/  # same masks, RGB colored for visualization
└── splits/
    ├── anterior/
    │   ├── train.txt
    │   ├── val.txt
    │   ├── test.txt
    │   └── splits_5fold.json
    └── posterior/
        ├── train.txt
        ├── val.txt
        ├── test.txt
        └── splits_5fold.json

source_code/
├── preprocessing/
│   ├── generate_splits.py      # generates splits_5fold.json from split txt files
│   ├── verify_splits.py        # verifies no leakage between splits
│   └── verify_preprocessing.py
├── evaluation/
│   ├── inference_all_folds.py       # runs nnU-Netv2 inference across all 5 folds
│   ├── evaluate_and_visualize.py    # computes DSC/IoU per class, saves results
│   └── build_crossval_report.py     # aggregates fold results into Excel report
└── configs/
    ├── nnUNetPlans_anterior.json    # nnU-Netv2 self-configured plan, anterior
    ├── nnUNetPlans_posterior.json   # nnU-Netv2 self-configured plan, posterior
    └── baselines/
        ├── fcn_config.json
        ├── deeplabv3plus_config.json
        ├── danet_config.json
        └── segformer_config.json
```

---

## nnU-Netv2 Setup

**Version:** nnUNetv2 v2.6.0  
**Environment:** Python 3.10, PyTorch 2.5.1, CUDA 12.4, cuDNN 9.1.0

**Preprocessing applied before nnU-Netv2 fingerprinting:**
1. Contrast enhancement ×1.13
2. Color inversion (pixel = max\_intensity − pixel)
3. Z-score normalization (non-zero pixels only)
4. Resize to 512×128

Data augmentation was disabled for all methods to keep training conditions comparable.

**Training:**
- Optimizer: SGD + Nesterov momentum (µ = 0.99)
- LR: 0.01, decayed 0.0001/epoch (polynomial)
- Epochs: 59 (anterior), 64 (posterior)
- Batch size: 48
- Loss: Dice + Cross-Entropy
- Best checkpoint: highest EMA pseudo-Dice on validation set

**Inference:**  
Run `source_code/evaluation/inference_all_folds.py` after training all 5 folds.  
Evaluate with `source_code/evaluation/evaluate_and_visualize.py`.

---

## Baseline Configuration

FCN, DeepLabv3+, DANet, and SegFormer were implemented using mmsegmentation v1.2.2.  
All trained on the full development set, evaluated on the same hold-out test set.  
Parameter details are in `source_code/configs/baselines/`.

---

## Results

**nnU-Netv2 5-fold cross-validation (development set):**

| View | Macro-avg DSC |
|---|---|
| Anterior | 0.8168 ± 0.0029 |
| Posterior | 0.8345 ± 0.0018 |

**Hold-out test set comparison (Fold-0 model vs baselines):**

| Method | Anterior DSC | Posterior DSC |
|---|---|---|
| FCN | 0.691 | 0.756 |
| DeepLabv3+ | 0.814 | 0.711 |
| DANet | 0.814 | 0.831 |
| SegFormer | **0.824** | 0.806 |
| nnU-Netv2 | 0.814 | **0.835** |

---

## Data Responsibility

Dataset created and maintained by:  
**Ema Rachmawati** — Telkom University, Indonesia  
Contact: emarachmawati@telkomuniversity.ac.id

Funding:
- Ministry of Research, Technology, and Higher Education of Indonesia — Regular Fundamental Research, grant no. **125/C3/DT.05.00/PL/2025**
- Telkom University — research grant no. **063/LIT07/PPM-LIT/2025**

Annotation supervised by nuclear medicine physicians, Department of Nuclear Medicine and Molecular Theranostics, Dr. Hasan Sadikin General Hospital, Faculty of Medicine, Universitas Padjadjaran, Indonesia.

---

## Annotated Masks

Ground-truth masks are hosted on Zenodo (too large for GitHub):  
**[https://doi.org/10.5281/zenodo.20342310](https://doi.org/10.5281/zenodo.20342310)**

Download and place under `dataset/masks/` to match the repository structure.

---

## Citation

*To be added upon acceptance.*

---

## License

Ground-truth masks: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)  
Code: MIT
