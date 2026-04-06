import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.plantable_area_cnn import build_plantable_model

# ── CONFIG ────────────────────────────────────────────────────────
IMG_DIR    = Path("data/prepared/images")
MASK_DIR   = Path("data/prepared/masks")
SPLITS_DIR = Path("data/prepared/splits")
CKPT_DIR   = Path("model/checkpoints")
MODEL_OUT  = Path("model/plantable_trained_model.h5")

IMG_SIZE   = 256
BATCH_SIZE = 16     # RTX 3050 4GB — safe at 16
EPOCHS     = 50
LR         = 1e-4

# ── GPU SETUP ─────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU found: {gpus[0].name}")
else:
    print("No GPU found — running on CPU")

# ── DATA LOADER ───────────────────────────────────────────────────
def load_split(split_name):
    split_file = SPLITS_DIR / f"{split_name}.txt"
    if not split_file.exists():
        return np.array([]), np.array([])

    with open(split_file) as f:
        names = [l.strip() for l in f if l.strip()]

    images, masks = [], []
    skip = 0
    for name in names:
        img_path  = IMG_DIR  / f"{name}.png"
        mask_path = MASK_DIR / f"{name}.png"
        if not img_path.exists() or not mask_path.exists():
            skip += 1; continue

        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            skip += 1; continue

        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        images.append(img.astype(np.float32))
        masks.append(mask)

    print(f"  {split_name}: {len(images)} loaded, {skip} skipped")
    return np.array(images), np.array(masks)

# ── AUGMENTATION ─────────────────────────────────────────────────
def augment(image, mask):
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)
    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.image.flip_up_down(mask)
    # Random brightness
    image = tf.image.random_brightness(image, 0.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask

# ── BUILD DATASET ─────────────────────────────────────────────────
print("\nLoading data...")
X_train, y_train = load_split("train")
X_val,   y_val   = load_split("val")

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = (train_ds
            .shuffle(500)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ── LOSS — Dice + BCE combined ────────────────────────────────────
def dice_bce_loss(y_true, y_pred):
    # BCE
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Dice
    smooth  = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = 1 - (2*intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return bce + dice

# ── IoU METRIC ───────────────────────────────────────────────────
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter  = tf.reduce_sum(y_true * y_pred)
    union  = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + 1e-6) / (union + 1e-6)

# ── BUILD & COMPILE ───────────────────────────────────────────────
print("\nBuilding model...")
model = build_plantable_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=dice_bce_loss,
    metrics=["accuracy", iou_metric]
)
model.summary()

# ── CALLBACKS ────────────────────────────────────────────────────
callbacks = [
    # Save best model
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT_DIR / "best_model.h5"),
        monitor="val_iou_metric",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    # Reduce LR on plateau
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    # Early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor="val_iou_metric",
        mode="max",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    # CSV log
    tf.keras.callbacks.CSVLogger("model/training_log.csv"),
]

# ── TRAIN ─────────────────────────────────────────────────────────
print(f"\nStarting training — {EPOCHS} epochs max, batch {BATCH_SIZE}")
print("Early stopping enabled (patience=12 on val_iou_metric)\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ── SAVE FINAL ────────────────────────────────────────────────────
model.save(str(MODEL_OUT))
print(f"\nModel saved → {MODEL_OUT}")

# ── QUICK SUMMARY ─────────────────────────────────────────────────
best_iou = max(history.history.get("val_iou_metric", [0]))
best_acc = max(history.history.get("val_accuracy",   [0]))
print(f"\nBest Val IoU:      {best_iou:.4f}")
print(f"Best Val Accuracy: {best_acc:.4f}")
print("\nTraining complete!") 