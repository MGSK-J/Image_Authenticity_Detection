import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image_heatmap(image_path, color_map='hot'):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Show image as heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap=color_map)
    plt.colorbar(label='Pixel Intensity')
    plt.title(f"Heatmap of Pixel Intensities ({color_map} colormap)")
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = 'test2.jpg'  # Change this to your image filename
    show_image_heatmap(image_path, color_map='hot')  # Other options: 'jet', 'viridis', 'plasma', etc.
