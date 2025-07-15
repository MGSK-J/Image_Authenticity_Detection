import cv2
import numpy as np
from PIL import Image, ExifTags
import os



def generate_manipulation_report(image_path, ela_mask, report_output_path="manipulation_report.txt"):
    """
    Generates a textual report summarizing the manipulation detection findings,
    with a focus on how Error Level Analysis (ELA) works.

    Args:
        image_path (str): The path to the analyzed image.
        ela_mask (numpy.ndarray): The binary mask from ELA indicating manipulated areas.
        metadata_results (dict): The dictionary of metadata analysis findings.
        report_output_path (str): The path to save the generated report.
    """
    report_content = []
    report_content.append(f"--- Image Manipulation Analysis Report for: {os.path.basename(image_path)} ---")
    report_content.append(f"Analysis Date: {np.datetime_as_string(np.datetime64('now'))}")

    report_content.append("\n--- Understanding Error Level Analysis (ELA) ---")
    report_content.append("ELA is a technique used in digital image forensics to detect alterations or manipulations within an image.")
    report_content.append("It works on the principle that re-saving a JPEG image, especially after modifications, will introduce different levels of compression artifacts across the image.")
    report_content.append("Original areas of the image will have a consistent level of compression error, while manipulated or newly introduced areas (copied, pasted, or heavily edited) will show a different error level because they have been re-compressed, often multiple times or with different settings.")
    report_content.append("\nHow ELA is Performed in this Analysis:")
    report_content.append("1.  **Re-compression:** The original image is intentionally re-saved at a known JPEG quality (e.g., 90%).")
    report_content.append("2.  **Difference Calculation:** The absolute difference between the original image and this newly re-compressed image is calculated. Areas that have been manipulated will typically show higher differences (brighter areas in the ELA output) because their compression history is inconsistent with the rest of the image.")
    report_content.append("3.  **Thresholding:** The difference image is converted to grayscale, and a thresholding technique (Otsu's method) is applied to create a binary mask. This mask highlights pixels with significant error levels.")
    report_content.append("4.  **Dilation:** The mask is then dilated to expand the detected regions, making potential manipulations more visible and easier to interpret.")
    report_content.append("5.  **Overlay:** A red overlay is created using this mask and combined with the original image to visually pinpoint the areas of suspected manipulation.")

    report_content.append("\n--- ELA Findings for This Image ---")

    total_pixels = ela_mask.shape[0] * ela_mask.shape[1]
    manipulated_pixels = np.sum(ela_mask > 0)
    manipulation_percentage = (manipulated_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    report_content.append(f"Total Pixels in Image: {total_pixels}")
    report_content.append(f"Pixels Identified as Potentially Manipulated by ELA: {manipulated_pixels}")
    report_content.append(f"Percentage of Image Showing ELA Anomalies: {manipulation_percentage:.2f}%")

    if manipulation_percentage > 5: # Arbitrary threshold for significant manipulation
        report_content.append("\nInterpretation of ELA Anomalies:")
        report_content.append("The ELA analysis indicates significant areas ({manipulation_percentage:.2f}%) with inconsistent compression characteristics. This strongly suggests that parts of the image have been altered. Common manipulations that cause such anomalies include:")
        report_content.append("  - **Content Insertion/Splicing:** Adding objects or regions from another image.")
        report_content.append("  - **Content Removal/Inpainting:** Erasing an object and filling the void, often with synthesized content.")
        report_content.append("  - **Copy-Move Forgery:** Duplicating parts of the image and pasting them elsewhere within the same image.")
        report_content.append("  - **Localized Adjustments:** Applying filters, brightness/contrast changes, or other edits to specific, non-uniform areas.")
        report_content.append("  - **Multiple Re-compressions:** Parts of the image may have been saved multiple times or at different quality settings, leading to varied error levels.")
    elif manipulation_percentage > 0:
        report_content.append("\nInterpretation of ELA Anomalies:")
        report_content.append("The ELA analysis indicates minor areas ({manipulation_percentage:.2f}%) with compression inconsistencies. This could be due to:")
        report_content.append("  - Minor retouching or subtle localized edits.")
        report_content.append("  - Natural variations in image content or slight artifacts from normal image processing (e.g., resizing, minor color correction) that are not necessarily malicious.")
    else:
        report_content.append("\nInterpretation of ELA Anomalies:")
        report_content.append("ELA does not indicate significant areas of manipulation based on compression inconsistencies. The error levels appear relatively uniform across the image.")

   
    # Save the report
    with open(report_output_path, "w") as f:
        for line in report_content:
            f.write(line + "\n")
    print(f"\nComprehensive manipulation report saved to: {report_output_path}")

def detect_manipulation(image_path, output_ela_path="ela_result.jpg", output_overlay_path="manipulation_overlay.jpg", report_output_path="manipulation_report.txt", quality=90, threshold=12, kernel_size=(3, 3)):
    """
    Detects image manipulation using Error Level Analysis (ELA) and metadata analysis,
    and generates a detailed report.

    Args:
        image_path (str): The path to the input image.
        output_ela_path (str): Path to save the ELA output image.
        output_overlay_path (str): Path to save the image with manipulation overlay.
        report_output_path (str): Path to save the textual report.
        quality (int): JPEG quality factor for ELA compression (0-100).
        threshold (int): Threshold for ELA grayscale image to create binary mask.
        kernel_size (tuple): Kernel size for dilation operation on the ELA mask.
    """
    # --- ELA (Error Level Analysis) ---
    print("\n--- Performing Error Level Analysis (ELA) ---")
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not read image at {image_path}. Please ensure the path is correct and the image exists.")
        return

    # Normalize the original image to 0-255 range
    original_normalized = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX)

    # Create temporary compressed image
    temp_compressed_path = "temp_compressed_ela.jpg"
    try:
        cv2.imwrite(temp_compressed_path, original_normalized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imread(temp_compressed_path)

        if compressed is None:
            print(f"Error: Could not create or read temporary compressed image at {temp_compressed_path}. Check write permissions or image format.")
            return

        # Calculate ELA (absolute difference)
        ela_image = np.abs(original_normalized.astype(np.int16) - compressed.astype(np.int16))
        ela_image = ela_image.clip(0, 255).astype(np.uint8) # Clip values to 0-255 and convert to uint8

        # Convert ELA to grayscale for thresholding
        gray_ela = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)

        # Use Otsu's thresholding for adaptive thresholding
        _, mask = cv2.threshold(gray_ela, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilate mask to make manipulated regions more prominent
        kernel = np.ones(kernel_size, np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Create a red overlay for detected manipulated areas
        red_overlay = np.zeros_like(original_normalized)
        red_overlay[:, :] = (0, 0, 255)  # Red color in BGR format
        red_areas = cv2.bitwise_and(red_overlay, red_overlay, mask=mask)

        # Combine original image with the red overlay
        # Adjust alpha values for better visibility of manipulation
        result_overlay = cv2.addWeighted(original_normalized, 0.7, red_areas, 0.3, 0)

        # Save ELA results
        cv2.imwrite(output_ela_path, ela_image)
        cv2.imwrite(output_overlay_path, result_overlay)

        print(f"ELA result saved to: {output_ela_path}")
        print(f"Manipulation overlay saved to: {output_overlay_path}")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_compressed_path):
            os.remove(temp_compressed_path)


    # --- Generate Comprehensive Report ---
    generate_manipulation_report(image_path, mask, report_output_path)


    # Display results (optional, for local testing)
    cv2.imshow("Original Image", original_normalized)
    cv2.imshow("ELA Result", ela_image)
    cv2.imshow("Potential Manipulations Overlay", result_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Usage Example ---
if __name__ == "__main__":
    # Specify the path to your image here.
    # Make sure the image file exists in the same directory as your script,
    # or provide the full path to the image.
    image_to_analyze = "test.jpg" # <--- IMPORTANT: Change this to your actual image file name!

    # Example usage:
    detect_manipulation(
        image_to_analyze,
        output_ela_path="ela_output.jpg",
        output_overlay_path="manipulation_detected.jpg",
        report_output_path="manipulation_report.txt", # New parameter for report
        quality=90,
        threshold=12,
        kernel_size=(5, 5)
    )

    # To test with a manipulated image:
    # 1. Take an original photo.
    # 2. Open it in an image editor (e.g., Photoshop, GIMP).
    # 3. Make a small, subtle change (e.g., clone stamp a small area, adjust brightness/contrast on a specific region, or add a small object).
    # 4. Save it as a NEW JPEG file (do not overwrite the original).
    # 5. Replace "your_image_name.jpg" above with the path to your newly manipulated image.
    # 6. Run the script and observe the ELA output, metadata report, and the generated 'manipulation_report.txt' file.
