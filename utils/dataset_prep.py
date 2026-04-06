import os
import cv2
import numpy as np
import shutil
import random
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────
BASE         = Path("D:/Urban_Green_Space_Optimizer")
LOVEDA_SPLITS = {
    "train": BASE / "data/loveda/Train/Urban",
    "val":   BASE / "data/loveda/Val/Urban",
}
DEEPGLOBE    = BASE / "data/deepglobe/train"
OUT_IMG      = BASE / "data/prepared/images"
OUT_MASK     = BASE / "data/prepared/masks"
OUT_SPLITS   = BASE / "data/prepared/splits"
IMG_SIZE     = (256, 256)

# ── LoveDA class IDs → plantable ──────────────────────────────────
# 0=bg, 1=building, 2=road, 3=water, 4=barren, 5=forest, 6=agriculture
LOVEDA_PLANTABLE = {4, 5, 6}

# ── DeepGlobe RGB colours → plantable ─────────────────────────────
DEEPGLOBE_PLANTABLE = [
    (255, 255,   0),   # Agriculture
    (255,   0, 255),   # Rangeland
    (  0, 255,   0),   # Forest
    (255, 255, 255),   # Barren
]

# ─────────────────────────────────────────────────────────────────
def loveda_to_binary(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    binary = np.zeros_like(mask, dtype=np.uint8)
    for cls in LOVEDA_PLANTABLE:
        binary[mask == cls] = 255
    return binary

def deepglobe_to_binary(mask_path):
    mask = cv2.imread(str(mask_path))
    if mask is None:
        return None
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    binary = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for color in DEEPGLOBE_PLANTABLE:
        match = np.all(mask_rgb == np.array(color), axis=-1)
        binary[match] = 255
    return binary

def save_pair(img, mask, name):
    img_r  = cv2.resize(img,  IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT_IMG  / f"{name}.png"), img_r)
    cv2.imwrite(str(OUT_MASK / f"{name}.png"), mask_r)

# ─────────────────────────────────────────────────────────────────
def process_loveda():
    count = skip = 0
    for split, folder in LOVEDA_SPLITS.items():
        img_dir  = folder / "images_png"
        mask_dir = folder / "masks_png"
        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping")
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                skip += 1; continue
            img  = cv2.imread(str(img_path))
            mask = loveda_to_binary(mask_path)
            if img is None or mask is None:
                skip += 1; continue
            # skip if less than 3% plantable
            if np.sum(mask == 255) / mask.size < 0.03:
                skip += 1; continue
            save_pair(img, mask, f"ld_{split}_{img_path.stem}")
            count += 1
    print(f"  LoveDA: {count} saved, {skip} skipped")
    return count

def process_deepglobe():
    count = skip = 0
    for sat_path in sorted(DEEPGLOBE.glob("*_sat.jpg")):
        stem      = sat_path.stem.replace("_sat", "")
        mask_path = DEEPGLOBE / f"{stem}_mask.png"
        if not mask_path.exists():
            skip += 1; continue
        img  = cv2.imread(str(sat_path))
        mask = deepglobe_to_binary(mask_path)
        if img is None or mask is None:
            skip += 1; continue
        if np.sum(mask == 255) / mask.size < 0.03:
            skip += 1; continue
        save_pair(img, mask, f"dg_{stem}")
        count += 1
    print(f"  DeepGlobe: {count} saved, {skip} skipped")
    return count

def create_splits(names):
    random.seed(42)
    random.shuffle(names)
    n       = len(names)
    n_train = int(n * 0.75)
    n_val   = int(n * 0.15)
    train   = names[:n_train]
    val     = names[n_train:n_train+n_val]
    test    = names[n_train+n_val:]
    for sname, slist in [("train",train),("val",val),("test",test)]:
        with open(OUT_SPLITS / f"{sname}.txt","w") as f:
            f.write("\n".join(slist))
        print(f"  {sname}: {len(slist)} samples")

# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  Dataset Preparation — Urban Green Space Optimizer")
    print("="*55)

    # clear old prepared data
    for folder in [OUT_IMG, OUT_MASK]:
        for f in folder.glob("*"):
            f.unlink()
    print("\nCleared old prepared data.")

    print("\n[1] Processing LoveDA Urban (Train + Val)...")
    c1 = process_loveda()

    print("\n[2] Processing DeepGlobe...")
    c2 = process_deepglobe()

    total = c1 + c2
    print(f"\n[3] Total samples: {total}")

    print("\n[4] Creating splits (75% train / 15% val / 10% test)...")
    all_names = [p.stem for p in OUT_IMG.glob("*.png")]
    create_splits(all_names)

    print("\nDone! data/prepared/ is ready for training.")
    print("="*55)