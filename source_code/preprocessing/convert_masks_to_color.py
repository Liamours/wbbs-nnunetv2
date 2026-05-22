"""
Convert label masks (pixel values 0-12) to RGB colored masks.

Input:  dataset/masks/{anterior,posterior}/         - grayscale label PNGs (uint8, values 0-12)
Output: dataset/masks/{anterior,posterior}_colored/ - RGB colored PNGs

Color palette matches the original annotation colors used during labeling.
See dataset/palette_to_label.json for hex-to-label-index mapping.

Usage:
    python convert_masks_to_color.py
"""

from pathlib import Path
from PIL import Image
import numpy as np

# Label index -> RGB color (matches original annotation palette)
PALETTE = {
    0:  [0,   0,   0  ],  # background
    1:  [176, 230, 13 ],  # skull
    2:  [0,   151, 219],  # cervical vertebrae
    3:  [126, 230, 225],  # thoracic vertebrae
    4:  [166, 55,  167],  # rib
    5:  [230, 157, 180],  # sternum (anterior only)
    6:  [167, 110, 77 ],  # collarbone
    7:  [121, 0,   24 ],  # scapula
    8:  [56,  65,  184],  # humerus
    9:  [230, 218, 0  ],  # lumbar vertebrae
    10: [230, 114, 35 ],  # sacrum
    11: [12,  187, 62 ],  # pelvis
    12: [230, 182, 22 ],  # femur
}

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "masks"


def convert(src_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    if not files:
        print(f"  No PNG files found in {src_dir}")
        return
    for f in files:
        label = np.array(Image.open(f))
        assert label.ndim == 2, f"Expected grayscale label, got shape {label.shape}"
        rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
        for idx, color in PALETTE.items():
            rgb[label == idx] = color
        Image.fromarray(rgb).save(out_dir / f.name)
    print(f"  {len(files)} files converted: {src_dir.name} -> {out_dir.name}")


if __name__ == "__main__":
    for view in ["anterior", "posterior"]:
        convert(DATASET_DIR / view, DATASET_DIR / (view + "_colored"))
    print("Done.")
