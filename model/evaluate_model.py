"""
evaluate_model.py  —  Urban Green Space Optimizer
Evaluates the trained PyTorch U-Net on the test split (data/prepared/splits/test.txt).
Run from project root:  python model/evaluate_model.py

Outputs:
  - Per-image metrics printed every 25 images
  - Average IoU, Dice, Precision, Recall printed at end
  - Results saved to model/evaluation_results.csv
"""

import os
import sys
import csv
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path

# ── Model definition (must match train_pytorch.py exactly) ───────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class PlantableUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 32);   self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128); self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1  = nn.ConvTranspose2d(64,  32,  2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out  = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x);           e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))

# ── Metric functions ─────────────────────────────────────────────────────────

def compute_metrics(pred_bin, gt_bin):
    pred = pred_bin.astype(bool)
    gt   = gt_bin.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    iou       = tp / (tp + fp + fn + 1e-8)
    dice      = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    return {
        "IoU":       round(float(iou),       4),
        "Dice":      round(float(dice),      4),
        "Precision": round(float(precision), 4),
        "Recall":    round(float(recall),    4),
    }

# ── Paths ────────────────────────────────────────────────────────────────────

IMAGE_DIR   = Path("data/prepared/images")
MASK_DIR    = Path("data/prepared/masks")
SPLIT_FILE  = Path("data/prepared/splits/test.txt")
MODEL_PATH  = Path("model/checkpoints/best_model.pth")
RESULTS_CSV = Path("model/evaluation_results.csv")
IMG_SIZE    = 256

# ── Load model ───────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*55}")
print(f"  Urban Green Space Optimizer — Model Evaluation")
print(f"{'='*55}")
print(f"  Device     : {device}")
print(f"  Model      : {MODEL_PATH}")

if not MODEL_PATH.exists():
    MODEL_PATH = Path("model/plantable_trained_model.pth")
    if not MODEL_PATH.exists():
        print("\n[ERROR] No trained model found. Exiting.")
        sys.exit(1)

model = PlantableUNet().to(device)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device, weights_only=False))
model.eval()
print(f"  Model loaded successfully.")

# ── Read test split ───────────────────────────────────────────────────────────

if not SPLIT_FILE.exists():
    print(f"\n[ERROR] Split file not found: {SPLIT_FILE}")
    sys.exit(1)

with open(SPLIT_FILE, "r") as f:
    test_files = [
        (line.strip() if line.strip().endswith(".png") else line.strip() + ".png")
        for line in f if line.strip()
    ]

print(f"  Test images : {len(test_files)}")
print(f"{'='*55}\n")

# ── Run evaluation ────────────────────────────────────────────────────────────

all_metrics  = []
skipped      = 0
results_rows = []

for idx, fname in enumerate(test_files):
    img_path  = IMAGE_DIR / fname
    mask_path = MASK_DIR  / fname

    image = cv2.imread(str(img_path))
    if image is None:
        skipped += 1
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

    if not mask_path.exists():
        skipped += 1
        continue
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        skipped += 1
        continue
    gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE))
    gt_bin  = (gt_mask > 127).astype(np.uint8)

    tensor = torch.from_numpy(
        image_resized.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()

    pred_bin = (pred > 0.5).astype(np.uint8)

    m = compute_metrics(pred_bin, gt_bin)
    all_metrics.append(m)
    results_rows.append({"filename": fname, **m})

    if (idx + 1) % 25 == 0 or (idx + 1) == len(test_files):
        print(f"  [{idx+1:3d}/{len(test_files)}] "
              f"IoU={m['IoU']:.4f}  Dice={m['Dice']:.4f}  "
              f"Prec={m['Precision']:.4f}  Rec={m['Recall']:.4f}")

# ── Aggregate ─────────────────────────────────────────────────────────────────

if len(all_metrics) == 0:
    print("\n[ERROR] No images evaluated.")
    sys.exit(1)

avg_iou       = float(np.mean([m["IoU"]       for m in all_metrics]))
avg_dice      = float(np.mean([m["Dice"]      for m in all_metrics]))
avg_precision = float(np.mean([m["Precision"] for m in all_metrics]))
avg_recall    = float(np.mean([m["Recall"]    for m in all_metrics]))
std_iou       = float(np.std( [m["IoU"]       for m in all_metrics]))
std_dice      = float(np.std( [m["Dice"]      for m in all_metrics]))

print(f"\n{'='*55}")
print(f"  EVALUATION RESULTS  ({len(all_metrics)} test images)")
print(f"{'='*55}")
print(f"  IoU          : {avg_iou:.4f}  (+-{std_iou:.4f})")
print(f"  Dice Score   : {avg_dice:.4f}  (+-{std_dice:.4f})")
print(f"  Precision    : {avg_precision:.4f}")
print(f"  Recall       : {avg_recall:.4f}")
if skipped:
    print(f"  Skipped      : {skipped}")
print(f"{'='*55}\n")

# ── Save CSV ──────────────────────────────────────────────────────────────────

RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename","IoU","Dice","Precision","Recall"])
    writer.writeheader()
    writer.writerows(results_rows)
    writer.writerow({
        "filename":  "AVERAGE",
        "IoU":       round(avg_iou,       4),
        "Dice":      round(avg_dice,      4),
        "Precision": round(avg_precision, 4),
        "Recall":    round(avg_recall,    4),
    })

print(f"  Results saved to: {RESULTS_CSV}")
print(f"  Done.\n")