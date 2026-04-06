print("=== EVALUATION SCRIPT STARTED ===")

import sys
import os
import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.evaluation_metrics import calculate_metrics

IMAGE_DIR = "data/training_data/images"
MASK_DIR = "data/training_data/masks"

IMG_SIZE = 256

print("Loading trained model...")
model = tf.keras.models.load_model(
    "model/plantable_trained_model.h5",
    compile=False
)

image_files = os.listdir(IMAGE_DIR)

valid_files = []

# -----------------------------
# Filter images with non-empty masks
# -----------------------------
for file in image_files:
    mask_path = os.path.join(MASK_DIR, file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        continue

    if np.max(mask) > 0:   # mask contains white pixels
        valid_files.append(file)

# Use first 5 valid images
image_files = valid_files[:5]

if len(image_files) == 0:
    print("No valid masks with plantable regions found.")
    exit()

all_metrics = []

for file in image_files:

    img_path = os.path.join(IMAGE_DIR, file)
    mask_path = os.path.join(MASK_DIR, file)

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    # Normalize image
    image_input = np.expand_dims(image / 255.0, axis=0)

    # Convert mask from 0–255 to 0–1
    ground_truth = (mask / 255).astype(np.uint8)

    # Predict
    prediction = model.predict(image_input)[0]
    prediction = prediction.squeeze()
    prediction_binary = (prediction > 0.5).astype(np.uint8)

    metrics = calculate_metrics(ground_truth, prediction_binary)
    all_metrics.append(metrics)

# -----------------------------
# Compute Average Metrics
# -----------------------------
avg_iou = np.mean([m["IoU"] for m in all_metrics])
avg_dice = np.mean([m["Dice"] for m in all_metrics])
avg_precision = np.mean([m["Precision"] for m in all_metrics])
avg_recall = np.mean([m["Recall"] for m in all_metrics])

print("\nAverage Evaluation Metrics (5 Non-Empty Masks):")
print(f"IoU: {avg_iou:.4f}")
print(f"Dice Score: {avg_dice:.4f}")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")

print("\nEvaluation Complete.")