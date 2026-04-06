import cv2
import numpy as np

# ---------------------------------
# Realistic spatial assumptions
# ---------------------------------

# Pixel to square meter conversion
PIXEL_TO_SQM = 0.05

# Space required per tree (4m × 4m)
TREE_SPACE_SQM = 16


# ---------------------------------
# Calculate plantable area
# ---------------------------------

def calculate_plantable_area(mask_image, image_type=None):
    """
    Calculates plantable area from mask image.
    image_type parameter is kept for compatibility with app.py
    """

    # Convert mask to grayscale if needed
    if len(mask_image.shape) == 3:
        mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask = mask_image

    # Convert to binary mask
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Count plantable pixels
    plantable_pixels = np.sum(binary == 255)

    # Convert pixels to square meters
    plantable_area = plantable_pixels * PIXEL_TO_SQM

    return plantable_pixels, plantable_area


# ---------------------------------
# Estimate tree count
# ---------------------------------

def calculate_tree_count(area):

    if area <= 0:
        return 0

    trees = int(area / TREE_SPACE_SQM)

    return trees