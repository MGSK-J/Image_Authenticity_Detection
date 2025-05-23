import cv2
import numpy as np
from PIL import Image, ExifTags
import os

def analyze_metadata(image_path):
    """
    Analyzes EXIF metadata of an image to detect potential manipulation.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing metadata analysis findings.
    """
    metadata_findings = {
        "Software_Used": "N/A",
        "DateTime_Original": "N/A",
        "DateTime_Digitized": "N/A",
        "DateTime_Modified": "N/A",
        "Camera_Make": "N/A",
        "Camera_Model": "N/A",
        "Lens_Model": "N/A",
        "Flash_Used": "N/A",
        "Orientation": "N/A",
        "User_Comment": "N/A",
        "GPS_Info": "N/A",
        "Potential_Manipulation_Indicators": []
    }

    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()

            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)

                    if tag_name == 'Software':
                        metadata_findings["Software_Used"] = value
                        # Common image manipulation software signatures
                        manipulation_software_keywords = [
                            "photoshop", "gimp", "lightroom", "affinity photo", "corel paintshop",
                            "adobe illustrator", "adobe premiere", "davinci resolve", "figma",
                            "stable diffusion", "midjourney", "dall-e", "runwayml", "artbreeder"
                        ]
                        if any(keyword in value.lower() for keyword in manipulation_software_keywords):
                            indicator_message = f"Software '{value}' detected. This indicates post-processing, potentially including generative AI tools."
                            if "stable diffusion" in value.lower() or "midjourney" in value.lower() or "dall-e" in value.lower():
                                indicator_message += " (Likely AI-generated or heavily AI-processed)."
                            metadata_findings["Potential_Manipulation_Indicators"].append(indicator_message)
                    elif tag_name == 'DateTimeOriginal':
                        metadata_findings["DateTime_Original"] = value
                    elif tag_name == 'DateTimeDigitized':
                        metadata_findings["DateTime_Digitized"] = value
                    elif tag_name == 'DateTime': # DateTime is usually modification date
                        metadata_findings["DateTime_Modified"] = value
                    elif tag_name == 'Make':
                        metadata_findings["Camera_Make"] = value
                    elif tag_name == 'Model':
                        metadata_findings["Camera_Model"] = value
                    elif tag_name == 'LensModel':
                        metadata_findings["Lens_Model"] = value
                    elif tag_name == 'Flash':
                        metadata_findings["Flash_Used"] = value # Value typically indicates if flash was fired
                    elif tag_name == 'Orientation':
                        metadata_findings["Orientation"] = value
                    elif tag_name == 'UserComment' or tag_name == 'XPComment':
                        try:
                            # UserComment can be bytes, try decoding
                            metadata_findings["User_Comment"] = value.decode('utf-8')
                        except (UnicodeDecodeError, AttributeError):
                            metadata_findings["User_Comment"] = str(value) # Fallback to string representation
                        if metadata_findings["User_Comment"] and "generated" in metadata_findings["User_Comment"].lower():
                            metadata_findings["Potential_Manipulation_Indicators"].append(
                                f"User comment contains '{metadata_findings['User_Comment']}'. This might indicate synthetic generation."
                            )
                    elif tag_name == 'GPSInfo':
                        # Decode GPSInfo if available
                        gps_info = {}
                        for gps_tag, gps_value in value.items():
                            gps_tag_name = ExifTags.GPSTAGS.get(gps_tag, gps_tag)
                            gps_info[gps_tag_name] = gps_value
                        metadata_findings["GPS_Info"] = gps_info
                        if gps_info:
                            metadata_findings["Potential_Manipulation_Indicators"].append(
                                f"GPS information found: {gps_info}. Verify if this location is consistent with the image content or if it seems unusual for the image subject."
                            )

                # Check for inconsistencies and common manipulation patterns
                if metadata_findings["Software_Used"] != "N/A" and \
                   metadata_findings["Camera_Make"] == "N/A" and \
                   metadata_findings["Camera_Model"] == "N/A":
                    metadata_findings["Potential_Manipulation_Indicators"].append(
                        "Image processed by software but no camera make/model found. This could suggest the image was not directly from a camera or had its camera metadata stripped/overwritten."
                    )
                
                if metadata_findings["DateTime_Original"] != "N/A" and \
                   metadata_findings["DateTime_Modified"] != "N/A" and \
                   metadata_findings["DateTime_Original"] != metadata_findings["DateTime_Modified"]:
                    metadata_findings["Potential_Manipulation_Indicators"].append(
                        f"Original creation date ({metadata_findings['DateTime_Original']}) differs from last modification date ({metadata_findings['DateTime_Modified']}). This indicates the image has been modified after creation."
                    )
                
                if metadata_findings["Camera_Make"] == "N/A" and \
                   metadata_findings["Camera_Model"] == "N/A" and \
                   "photoshop" not in metadata_findings["Software_Used"].lower() and \
                   "gimp" not in metadata_findings["Software_Used"].lower() and \
                   "lightroom" not in metadata_findings["Software_Used"].lower() and \
                   not metadata_findings["Potential_Manipulation_Indicators"]: # Only if no other indicators are present
                    metadata_findings["Potential_Manipulation_Indicators"].append(
                        "No camera metadata and no common editing software detected. This could indicate a stripped metadata or an image generated by less common/custom tools, including some AI models."
                    )

            else:
                metadata_findings["Potential_Manipulation_Indicators"].append("No EXIF metadata found. This can sometimes indicate manipulation (e.g., metadata stripping) or that the image was generated by a system that doesn't embed standard EXIF data (common with some AI-generated images).")

    except Exception as e:
        print(f"Error reading EXIF data: {e}")
        metadata_findings["Potential_Manipulation_Indicators"].append(f"Error during metadata analysis: {e}. This might be due to a corrupted file or unusual metadata structure.")

    return metadata_findings

def detect_manipulation(image_path, output_ela_path="ela_result.jpg", output_overlay_path="manipulation_overlay.jpg", quality=90, threshold=12, kernel_size=(3, 3)):
    """
    Detects image manipulation using Error Level Analysis (ELA) and metadata analysis.

    Args:
        image_path (str): The path to the input image.
        output_ela_path (str): Path to save the ELA output image.
        output_overlay_path (str): Path to save the image with manipulation overlay.
        quality (int): JPEG quality factor for ELA compression (0-100).
        threshold (int): Threshold for ELA grayscale image to create binary mask.
        kernel_size (tuple): Kernel size for dilation operation on the ELA mask.
    """
    # --- ELA (Error Level Analysis) ---
    print("\n--- Performing Error Level Analysis (ELA) ---")
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Normalize the original image to 0-255 range
    original_normalized = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX)

    # Create temporary compressed image
    temp_compressed_path = "temp_compressed_ela.jpg"
    cv2.imwrite(temp_compressed_path, original_normalized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed = cv2.imread(temp_compressed_path)

    if compressed is None:
        print(f"Error: Could not create or read temporary compressed image at {temp_compressed_path}")
        os.remove(temp_compressed_path)
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

    # Clean up temporary file
    if os.path.exists(temp_compressed_path):
        os.remove(temp_compressed_path)

    print(f"ELA result saved to: {output_ela_path}")
    print(f"Manipulation overlay saved to: {output_overlay_path}")


    # --- Metadata Analysis ---
    print("\n--- Performing Metadata Analysis ---")
    metadata_results = analyze_metadata(image_path)
    print("\nMetadata Analysis Report:")
    for key, value in metadata_results.items():
        if key == "Potential_Manipulation_Indicators":
            print(f"  {key}:")
            if value:
                for indicator in value:
                    print(f"    - {indicator}")
            else:
                print("    No strong metadata manipulation indicators found.")
        else:
            print(f"  {key}: {value}")

    # Display results (optional, for local testing)
    cv2.imshow("Original Image", original_normalized)
    cv2.imshow("Result", ela_image)
    cv2.imshow("Potential Manipulations Overlay", result_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Usage Example ---
if __name__ == "__main__":
    # Specify the path to your image here.
    # Make sure the image file exists in the same directory as your script,
    # or provide the full path to the image.
    image_to_analyze = "Sonali Jayarathne.png" # <--- IMPORTANT: Change this to your actual image file name!

    # Example usage:
    detect_manipulation(
        image_to_analyze,
        output_ela_path="output.jpg",
        output_overlay_path="manipulation_detected.jpg",
        quality=90,
        threshold=12,
        kernel_size=(5, 5)
    )

