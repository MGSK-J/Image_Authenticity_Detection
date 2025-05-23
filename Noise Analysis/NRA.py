import cv2
import numpy as np
import time

def detect_manipulation_noise_optimized(image_path, output_path="noise_result.jpg",
                                        window_size=32, threshold=2.5, kernel_size=3):
    """
    Detect manipulated areas using noise inconsistency analysis (optimized).

    Parameters:
        image_path: Input image path
        output_path: Output image path
        window_size: Size of local noise analysis window (odd number recommended)
        threshold: Sensitivity threshold (higher = stricter detection)
        kernel_size: Morphology kernel size for cleaning detections
    """
    start_time = time.time()
    try:
        # Read image and convert to grayscale
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return False, None, None, None, f"Error: Could not read image at {image_path}"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Calculate noise residual using Gaussian filter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_residual = gray - blurred

        # Calculate local noise variance using integral images for efficiency
        s = noise_residual.shape
        padded_nr_sq = np.pad(noise_residual**2, [(window_size // 2, window_size - 1 - window_size // 2),
                                                 (window_size // 2, window_size - 1 - window_size // 2)], mode='reflect')
        padded_nr = np.pad(noise_residual, [(window_size // 2, window_size - 1 - window_size // 2),
                                            (window_size // 2, window_size - 1 - window_size // 2)], mode='reflect')

        integral_sq = cv2.integral(padded_nr_sq)
        integral = cv2.integral(padded_nr)

        def get_sum(integral_img, r1, c1, r2, c2):
            return integral_img[r2 + 1, c2 + 1] - integral_img[r1, c2 + 1] - integral_img[r2 + 1, c1] + integral_img[r1, c1]

        local_variance = np.zeros_like(gray)
        window_area = window_size * window_size
        for y in range(s[0]):
            for x in range(s[1]):
                y1 = y
                x1 = x
                y2 = y + window_size - 1
                x2 = x + window_size - 1

                sum_sq = get_sum(integral_sq, y1, x1, y2, x2)
                sum_val = get_sum(integral, y1, x1, y2, x2)

                mean_sq = sum_sq / window_area
                square_mean = (sum_val / window_area) ** 2
                local_variance[y, x] = np.maximum(mean_sq - square_mean, 0)

        # Normalize and threshold variance map
        variance_norm = cv2.normalize(local_variance, None, 0, 255, cv2.NORM_MINMAX)
        _, binary_mask = cv2.threshold(variance_norm.astype(np.uint8),
                                        threshold * np.median(variance_norm),
                                        255, cv2.THRESH_BINARY)

        # Clean up mask with morphology operations
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        # Find contours and draw red rectangles around suspicious regions
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = img.copy()
        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                detections.append(((x, y), (x + w, y + h)))

        # Save the result
        cv2.imwrite(output_path, result)

        end_time = time.time()
        processing_time = end_time - start_time

        return True, noise_residual, variance_norm, result, f"Manipulation detection completed in {processing_time:.4f} seconds."

    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        return False, None, None, None, f"An error occurred: {e} (Processing time: {processing_time:.4f} seconds)"

def generate_report(image_path, detection_successful, message, output_path="noise_detection.jpg"):
    """Generates a report on the manipulation detection process."""
    report = f"""
    Manipulation Detection Report

    Image Path: {image_path}
    Detection Status: {'Successful' if detection_successful else 'Failed'}
    Message: {message}
    Output Image Saved To: {output_path}

    --- Detailed Analysis ---
    The algorithm analyzes inconsistencies in the image noise to detect potential manipulations.
    It calculates the local noise variance and thresholds it to identify regions with unusual noise patterns.
    These regions are then highlighted with red rectangles in the output image.

    Further analysis might involve adjusting the `window_size`, `threshold`, and `kernel_size` parameters
    to fine-tune the detection sensitivity and reduce false positives or negatives.
    """
    print(report)
    with open("manipulation_report.txt", "w") as f:
        f.write(report)
    print("Report saved to manipulation_report.txt")

if __name__ == "__main__":
    image_file = "test2.jpg"  # Replace with the path to your image
    output_file = "noise_detection_optimized.jpg"

    success, noise, variance, detected_image, message = detect_manipulation_noise_optimized(
        image_file,
        output_path=output_file,
        window_size=32,
        threshold=2.5,
        kernel_size=3
    )

    print(message)
    generate_report(image_file, success, message, output_file)

    if success:
        if noise is not None:
            cv2.imshow("Optimized Noise Residual", noise.astype(np.uint8))
        if variance is not None:
            cv2.imshow("Optimized Variance Map", variance.astype(np.uint8))
        if detected_image is not None:
            cv2.imshow("Optimized Detection Result", detected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()