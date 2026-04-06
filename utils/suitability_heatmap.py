import cv2
import numpy as np

def generate_suitability_heatmap(mask):

    # Ensure binary mask
    binary_mask = (mask > 0).astype(np.uint8)

    # Blur to create smooth suitability gradient
    heat = cv2.GaussianBlur(binary_mask.astype(np.float32), (51, 51), 0)

    # Normalize heat values
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply color map
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    return heatmap


def overlay_heatmap(image, heatmap):

    blended = cv2.addWeighted(image, 0.65, heatmap, 0.35, 0)

    return blended