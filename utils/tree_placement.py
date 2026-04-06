import numpy as np
import cv2


def generate_tree_positions(mask, spacing=40):
    positions = []

    if mask.max() == 255:
        binary_mask = mask
    else:
        binary_mask = (mask > 0).astype(np.uint8) * 255

    # Light erosion - removes edge pixels only, keeps interior regions
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=2)

    height, width = eroded.shape
    for y in range(0, height, spacing):
        for x in range(0, width, spacing):
            if eroded[y, x] == 255:
                positions.append((x, y))

    return positions


def draw_tree_positions(image, positions):
    output = image.copy()
    for (x, y) in positions:
        cv2.circle(output, (x, y), 7, (0, 100, 0),   -1)
        cv2.circle(output, (x, y), 7, (0, 255, 0),    2)
        cv2.circle(output, (x, y), 2, (255, 255, 255), -1)
    return output