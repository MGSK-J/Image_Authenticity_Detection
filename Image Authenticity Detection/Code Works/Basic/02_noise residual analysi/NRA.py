import cv2
import numpy as np

def detect_manipulation_noise(image_path, output_path="noise_result.jpg", 
                             window_size=32, threshold=2.5, kernel_size=3):
    """
    Detect manipulated areas using noise inconsistency analysis.
    
    Parameters:
        image_path: Input image path
        output_path: Output image path
        window_size: Size of local noise analysis window (odd number recommended)
        threshold: Sensitivity threshold (higher = stricter detection)
        kernel_size: Morphology kernel size for cleaning detections
    """
    # Read image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Calculate noise residual using Gaussian filter
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = gray - blurred
    
    # Calculate local noise variance using sliding window
    mean_square = cv2.filter2D(noise_residual**2, -1, np.ones((window_size, window_size))/(window_size**2))
    square_mean = cv2.filter2D(noise_residual, -1, np.ones((window_size, window_size))/(window_size**2))**2
    local_variance = np.maximum(mean_square - square_mean, 0)
    
    # Normalize and threshold variance map
    variance_norm = cv2.normalize(local_variance, None, 0, 255, cv2.NORM_MINMAX)
    _, binary_mask = cv2.threshold(variance_norm.astype(np.uint8), 
                                  threshold*np.median(variance_norm), 
                                  255, cv2.THRESH_BINARY)
    
    # Clean up mask with morphology operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours and draw red rectangles around suspicious regions
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Save and display results
    cv2.imwrite(output_path, result)
    cv2.imshow("Noise Residual", noise_residual.astype(np.uint8))
    cv2.imshow("Variance Map", variance_norm.astype(np.uint8))
    cv2.imshow("Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
detect_manipulation_noise("test2.jpg", 
                         output_path="noise_detection.jpg",
                         window_size=32, 
                         threshold=2.5, 
                         kernel_size=3)