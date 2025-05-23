import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.ndimage import gaussian_filter

def z_score_anomaly_map(gray_image, window_size=15):
    local_mean = cv2.blur(gray_image.astype(np.float32), (window_size, window_size))
    squared = gray_image.astype(np.float32)**2
    local_var = cv2.blur(squared, (window_size, window_size)) - local_mean**2
    local_std = np.sqrt(local_var)
    z_score = (gray_image - local_mean) / (local_std + 1e-5)
    z_score_map = cv2.normalize(np.abs(z_score), None, 0, 255, cv2.NORM_MINMAX)
    return z_score_map.astype(np.uint8)

def high_pass_noise_residual(image):
    blurred = gaussian_filter(image, sigma=2)
    residual = cv2.absdiff(image, blurred.astype(np.uint8))
    return residual

def compute_lbp_texture(gray_image, radius=1, n_points=8):
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_normalized = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)
    return lbp_normalized.astype(np.uint8)

def generate_pixel_analysis_report(image_path, stats_dict, output_image_path):
    report = f"""
    Advanced Manipulation Detection Report
    --------------------------------------
    Image analyzed: {image_path}
    Output saved: {output_image_path}

    Description:
    The image was analyzed using multiple local descriptors:
    - High-pass noise residual map: highlights abrupt noise inconsistencies.
    - Local Binary Pattern (LBP) texture map: identifies unnatural textural patterns.
    - Z-score anomaly map: detects pixel intensity outliers using local statistical deviation.
    - Edge map: outlines structural inconsistencies using Sobel gradient.

    Combined anomaly map statistics:
    - Min pixel value: {stats_dict['min']}
    - Max pixel value: {stats_dict['max']}
    - Mean pixel value: {stats_dict['mean']:.2f}
    - Threshold used (Otsu): {stats_dict['otsu_threshold']}

    Suspicious regions were overlaid in red on the original image.
    """

    with open("advanced_manipulation_report.txt", "w") as f:
        f.write(report.strip())
    print(report.strip())

def detect_advanced_manipulation(image_path, output_path="advanced_output.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Noise Residual
    noise_map = high_pass_noise_residual(gray)
    cv2.imwrite("step1_noise_residual.jpg", noise_map)

    # Step 2: LBP Texture
    lbp_map = compute_lbp_texture(gray)
    cv2.imwrite("step2_lbp_texture.jpg", lbp_map)

    # Step 3: Z-Score Map
    z_map = z_score_anomaly_map(gray)
    cv2.imwrite("step3_z_score_map.jpg", z_map)

    # Step 4: Edge Map
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = cv2.magnitude(grad_x, grad_y)
    edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("step4_edge_map.jpg", edge_map)

    # Step 5: Combine all maps
    combined_map = cv2.addWeighted(noise_map, 0.25, lbp_map, 0.25, 0)
    combined_map = cv2.addWeighted(combined_map, 0.6, z_map, 0.25, 0)
    combined_map = cv2.addWeighted(combined_map, 0.8, edge_map, 0.2, 0)
    cv2.imwrite("step5_combined_map.jpg", combined_map)

    # Step 6: Threshold
    otsu_threshold, binary_mask = cv2.threshold(combined_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 7: Overlay red mask
    red_overlay = np.zeros_like(image)
    red_overlay[:, :] = (0, 0, 255)
    red_areas = cv2.bitwise_and(red_overlay, red_overlay, mask=binary_mask)
    final_output = cv2.addWeighted(image, 0.7, red_areas, 0.3, 0)
    cv2.imwrite(output_path, final_output)

    # Step 8: Create report
    stats = {
        "min": int(np.min(combined_map)),
        "max": int(np.max(combined_map)),
        "mean": float(np.mean(combined_map)),
        "otsu_threshold": int(otsu_threshold)
    }
    generate_pixel_analysis_report(image_path, stats, output_path)

    # Optional visualization
    cv2.imshow("Original", image)
    cv2.imshow("Final Detection", final_output)
    cv2.imshow("Combined Anomaly Map", combined_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_advanced_manipulation("test.jpg", output_path="advanced_result.jpg")
