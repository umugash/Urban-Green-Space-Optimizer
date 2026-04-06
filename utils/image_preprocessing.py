import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(256, 256)):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image_resized = cv2.resize(image, target_size)

    # Normalize pixel values
    image_normalized = image_resized / 255.0

    return image, image_normalized


if __name__ == "__main__":
    original, processed = preprocess_image("data/raw_images/ground1.jpg")

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Preprocessed Image")
    plt.imshow(processed)
    plt.axis("off")

    plt.show()
