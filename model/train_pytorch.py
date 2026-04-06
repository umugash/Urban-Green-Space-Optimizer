import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import csv

sys.path.append(str(Path(__file__).resolve().parent.parent))

# ── CONFIG ────────────────────────────────────────────────────────
IMG_DIR    = Path("data/prepared/images")
MASK_DIR   = Path("data/prepared/masks")
SPLITS_DIR = Path("data/prepared/splits")
CKPT_DIR   = Path("model/checkpoints")
MODEL_OUT  = Path("model/plantable_trained_model.pth")
LOG_FILE   = Path("model/training_log.csv")

IMG_SIZE   = 256
BATCH_SIZE = 16
EPOCHS     = 50
LR         = 1e-4
PATIENCE   = 12

# ── GPU SETUP ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── DATASET ───────────────────────────────────────────────────────
class PlantableDataset(Dataset):
    def __init__(self, split, augment=False):
        split_file = SPLITS_DIR / f"{split}.txt"
        with open(split_file) as f:
            self.names = [l.strip() for l in f if l.strip()]
        # filter to only existing pairs
        self.names = [n for n in self.names
                      if (IMG_DIR/f"{n}.png").exists()
                      and (MASK_DIR/f"{n}.png").exists()]
        self.augment = augment
        print(f"  {split}: {len(self.names)} samples")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img  = cv2.imread(str(IMG_DIR  / f"{name}.png"))
        mask = cv2.imread(str(MASK_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)

        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST)

        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img  = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                img  = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            # brightness jitter
            factor = np.random.uniform(0.85, 1.15)
            img    = np.clip(img * factor, 0, 1)

        # HWC → CHW
        img  = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

# ── MODEL (U-Net) ─────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class PlantableUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(3,   32)
        self.enc2 = ConvBlock(32,  64)
        self.enc3 = ConvBlock(64,  128)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)
        # Decoder
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        # Output
        self.out  = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))

# ── LOSS ──────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce  = self.bce(pred, target)
        smooth = 1e-6
        pred_f   = pred.view(-1)
        target_f = target.view(-1)
        inter    = (pred_f * target_f).sum()
        dice = 1 - (2*inter + smooth) / (pred_f.sum() + target_f.sum() + smooth)
        return bce + dice

# ── IoU ───────────────────────────────────────────────────────────
def iou_score(pred, target, threshold=0.5):
    pred_b = (pred > threshold).float()
    inter  = (pred_b * target).sum()
    union  = pred_b.sum() + target.sum() - inter
    return ((inter + 1e-6) / (union + 1e-6)).item()

# ── TRAIN ONE EPOCH ───────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = total_iou = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou  += iou_score(preds.detach(), masks)
    n = len(loader)
    return total_loss/n, total_iou/n

# ── VALIDATE ─────────────────────────────────────────────────────
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = total_iou = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            total_loss += criterion(preds, masks).item()
            total_iou  += iou_score(preds, masks)
    n = len(loader)
    return total_loss/n, total_iou/n

# ── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nLoading datasets...")
    train_ds = PlantableDataset("train", augment=True)
    val_ds   = PlantableDataset("val",   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print("\nBuilding model...")
    model     = PlantableUNet().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True)

    # CSV log
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch","train_loss","train_iou","val_loss","val_iou","lr"])

    best_iou    = 0.0
    patience_ct = 0

    print(f"\nStarting training — {EPOCHS} epochs, batch {BATCH_SIZE}")
    print(f"Early stopping patience: {PATIENCE}\n")
    print(f"{'Epoch':>6} {'T-Loss':>8} {'T-IoU':>7} {'V-Loss':>8} {'V-IoU':>7} {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, EPOCHS+1):
        t_loss, t_iou = train_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_iou = val_epoch(model, val_loader, criterion)
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6} {t_loss:>8.4f} {t_iou:>7.4f} "
              f"{v_loss:>8.4f} {v_iou:>7.4f} {lr:>10.2e}")

        # log
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, t_loss, t_iou, v_loss, v_iou, lr])

        scheduler.step(v_iou)

        # save best
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(),
                       str(CKPT_DIR / "best_model.pth"))
            print(f"  ✅ Best model saved — Val IoU: {best_iou:.4f}")
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # save final
    torch.save(model.state_dict(), str(MODEL_OUT))
    print(f"\nFinal model saved → {MODEL_OUT}")
    print(f"Best Val IoU: {best_iou:.4f}")
    print("\nTraining complete! 🎉")