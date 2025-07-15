import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image_heatmap_and_pixels_with_histogram(image_path, color_map='hot', max_display_size=(10, 10)):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Resize image if too large for table display
    if img.shape[0] > max_display_size[0] or img.shape[1] > max_display_size[1]:
        img_display = cv2.resize(img, max_display_size, interpolation=cv2.INTER_AREA)
    else:
        img_display = img

    # Prepare plot with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Heatmap ---
    axes[0].imshow(img_display, cmap=color_map)
    axes[0].set_title("Heatmap of Pixel Intensities")
    axes[0].axis('off')
    im = axes[0].imshow(img_display, cmap=color_map)
    fig.colorbar(im, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)

    # --- Pixel Value Table ---
    axes[1].axis('off')
    axes[1].set_title("Pixel Values Table")
    pixel_values = img_display.astype(str)
    table_data = pixel_values.tolist()
    table = axes[1].table(cellText=table_data,
                          loc='center',
                          cellLoc='center',
                          colWidths=[0.05] * img_display.shape[1])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.2, 1.2)

    # --- Histogram ---
    axes[2].hist(img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.8)
    axes[2].set_title("Pixel Intensity Histogram")
    axes[2].set_xlabel("Intensity Value")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = 'test2.jpg'  # Change this to your image filename
    show_image_heatmap_and_pixels_with_histogram(image_path, color_map='hot', max_display_size=(10, 10))
