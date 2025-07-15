import cv2
import numpy as np

def edge_consistency_analysis(image_path, output_path="edge_result.jpg", 
                             window_size=15, threshold=0.35):
    # Read image and convert to grayscale
    original = cv2.imread(image_path)
    if original is None:
        print("Error: Could not read image")
        return
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Edge detection using Canny
    edges_low = cv2.Canny(gray, 30, 100)
    edges_high = cv2.Canny(gray, 50, 150)
    
    # Calculate edge consistency map
    edge_diff = cv2.absdiff(edges_high, edges_low)
    edge_diff = cv2.GaussianBlur(edge_diff, (5,5), 0)
    
    # Calculate local edge consistency
    kernel = np.ones((window_size, window_size), np.float32)/(window_size**2)
    consistency_map = cv2.filter2D(edge_diff.astype(np.float32), -1, kernel)
    
    # Normalize and threshold
    consistency_norm = cv2.normalize(consistency_map, None, 0, 255, cv2.NORM_MINMAX)
    _, anomaly_mask = cv2.threshold(consistency_norm, threshold*255, 255, cv2.THRESH_BINARY)
    
    # Post-processing
    kernel = np.ones((5,5), np.uint8)
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
    
    # Create visualization
    result = original.copy()
    result[anomaly_mask == 255] = (0, 0, 255)  # Mark anomalies in red
    
    # Save and display results
    cv2.imwrite(output_path, result)
    cv2.imshow("Edge Consistency Map", consistency_norm.astype(np.uint8))
    cv2.imshow("Potential Manipulations", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
edge_consistency_analysis("test.jpg", 
                         output_path="edge_analysis_result.jpg",
                         window_size=15,
                         threshold=0.35)