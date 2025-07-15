import cv2
import numpy as np

def detect_manipulation(image_path, output_path="output.jpg", quality=90, threshold=12):
    # Read original image
    original = cv2.imread(image_path)
    if original is None:
        print("Error: Could not read image")
        return

    # Create temporary compressed image
    cv2.imwrite("temp_compressed.jpg", original, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed = cv2.imread("temp_compressed.jpg")

    # Calculate ELA (Error Level Analysis)
    ela_image = np.abs(original.astype(np.int16) - compressed.astype(np.int16))
    ela_image = ela_image.clip(0, 255).astype(np.uint8)

    # Convert ELA to grayscale and threshold to find potential manipulated areas
    gray_ela = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_ela, threshold, 255, cv2.THRESH_BINARY)

    # Dilate mask to make regions more visible
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Create red overlay for manipulated areas
    red_overlay = np.zeros_like(original)
    red_overlay[:,:] = (0, 0, 255)  # Red color in BGR format
    red_areas = cv2.bitwise_and(red_overlay, red_overlay, mask=mask)

    # Combine original with red overlay
    result = cv2.addWeighted(original, 0.7, red_areas, 0.3, 0)

    # Save and show results
    cv2.imwrite(output_path, result)
    cv2.imwrite("ela_result.jpg", ela_image)

    # Display results
    cv2.imshow("Original Image", original)
    cv2.imshow("ELA Result", ela_image)
    cv2.imshow("Potential Manipulations", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
detect_manipulation("test.jpg", output_path="result.jpg", quality=90, threshold=12)