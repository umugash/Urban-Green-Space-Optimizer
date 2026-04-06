import matplotlib
matplotlib.use("TkAgg")
import sys
import os

# ADD PROJECT ROOT TO PYTHON PATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import matplotlib.pyplot as plt

from model.plantable_area_cnn import build_plantable_model
# Save output image
output_path = "data/processed_images/plantable_overlay.png"
plt.savefig(output_path)
print(f"✅ Plantable area overlay saved at: {output_path}")
