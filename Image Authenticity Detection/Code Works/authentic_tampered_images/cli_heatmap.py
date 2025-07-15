import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageChops, ImageEnhance, ImageOps
from scipy.ndimage import laplace
import tensorflow as tf
from keras.models import load_model
from keras.utils import custom_object_scope
import matplotlib.cm as cm

# --- Preprocessing Functions (same as CLI predict) ---
def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp_heatmap.jpg'
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

def prepare_image_ela(image_path, target_size=(64, 64)):
    return np.array(convert_to_ela_image(image_path).resize(target_size)) / 255.0

def convert_to_hist_eq(path):
    image = Image.open(path).convert('RGB')
    r, g, b = image.split()
    r_eq = ImageOps.equalize(r)
    g_eq = ImageOps.equalize(g)
    b_eq = ImageOps.equalize(b)
    eq_img = Image.merge('RGB', (r_eq, g_eq, b_eq))
    return eq_img

def prepare_image_hist_eq(image_path, target_size=(64, 64)):
    return np.array(convert_to_hist_eq(image_path).resize(target_size)) / 255.0

def convert_to_laplacian_edge(path):
    image = Image.open(path).convert('L')
    arr = np.array(image, dtype=np.float32)
    edge = np.abs(laplace(arr))
    edge = np.clip(edge, 0, 255).astype(np.uint8)
    edge_rgb = np.stack([edge]*3, axis=-1)
    return Image.fromarray(edge_rgb)

def prepare_image_laplacian(image_path, target_size=(64, 64)):
    return np.array(convert_to_laplacian_edge(image_path).resize(target_size)) / 255.0

def extract_noise(image_path, method='combined', enhanced=True):
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros((64, 64, 3), dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    image_float = image.astype(np.float32) / 255.0

    denoised_nlm = cv2.fastNlMeansDenoisingColored(image, None, h=8, hColor=8, 
                                                  templateWindowSize=7, searchWindowSize=21)
    denoised_nlm = denoised_nlm.astype(np.float32) / 255.0
    denoised_bilateral = cv2.bilateralFilter(image, 9, 75, 75).astype(np.float32) / 255.0
    denoised_gaussian = cv2.GaussianBlur(image_float, (5, 5), 1.0)
    denoised_avg = (denoised_nlm + denoised_bilateral + denoised_gaussian) / 3.0
    noise = np.abs(image_float - denoised_avg)

    if enhanced:
        noise_enhanced = np.zeros_like(noise)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        for channel in range(3):
            noise_uint8 = (noise[:, :, channel] * 255).astype(np.uint8)
            noise_enhanced[:, :, channel] = clahe.apply(noise_uint8).astype(np.float32) / 255.0
        gamma = 0.7
        noise_enhanced = np.power(noise_enhanced, gamma)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 9.0
        noise_filtered = np.zeros_like(noise_enhanced)
        for channel in range(3):
            filtered = cv2.filter2D(noise_enhanced[:, :, channel], -1, kernel)
            noise_filtered[:, :, channel] = np.clip(filtered, 0, 1)
        noise = noise_filtered

    noise = np.clip(noise, 0, 1)
    return noise.astype(np.float32)

def prepare_image_noise(image_path, target_size=(64, 64)):
    return extract_noise(image_path, method='combined', enhanced=True)

class PassThroughCast(tf.keras.layers.Layer):
    def __init__(self, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self._dtype = dtype
    def call(self, inputs):
        return inputs

# --- ELA-based Heatmap Generation ---
def generate_ela_heatmap(image_path, target_size=(256, 256), quality=90):
    """Generate heatmap for tampering detection"""
    # Create ELA image at higher resolution for better detail
    temp_filename = 'temp_heatmap_ela.jpg'
    image = Image.open(image_path).convert('RGB')
    
    # Resize to target size for analysis
    image_resized = image.resize(target_size)
    
    # Save and reload to introduce JPEG compression artifacts
    image_resized.save(temp_filename, 'JPEG', quality=quality)
    compressed_image = Image.open(temp_filename)
    
    # Calculate ELA difference
    ela_image = ImageChops.difference(image_resized, compressed_image)
    
    # Convert to numpy array
    ela_array = np.array(ela_image, dtype=np.float32)
    
    # Calculate error magnitude (combine RGB channels)
    ela_magnitude = np.sqrt(np.sum(ela_array**2, axis=2))
    
    # Normalize to 0-1 range
    ela_magnitude = ela_magnitude / 255.0
    
    # Apply enhancement to make tampering more visible
    # Use a power function to enhance high error regions
    ela_enhanced = np.power(ela_magnitude, 0.5)  # Enhance contrast
    
    # Apply threshold to focus on significant differences
    threshold = np.percentile(ela_enhanced, 85)  # Top 15% of errors
    ela_thresholded = np.where(ela_enhanced > threshold, ela_enhanced, ela_enhanced * 0.3)
    
    # Clean up temp file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    return ela_thresholded

def overlay_heatmap_on_image(image_path, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay ELA heatmap on original image"""
    # Load original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image size
    original_size = (original.shape[1], original.shape[0])  # (width, height)
    heatmap_resized = cv2.resize(heatmap, original_size, interpolation=cv2.INTER_CUBIC)
    
    # Convert heatmap to color
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(original, 1-alpha, heatmap_colored, alpha, 0)
    
    return original, heatmap_colored, overlayed

def predict_ela_only(model, image_path):
    """Predict using only ELA model for comparison"""
    ela_model_path = "ela_model.h5"
    if os.path.exists(ela_model_path):
        ela_model = load_model(ela_model_path)
        img = prepare_image_ela(image_path)
        img = np.expand_dims(img, axis=0)
        pred = ela_model.predict(img)
        class_idx = np.argmax(pred)
        confidence = np.max(pred)
        return ("Real" if class_idx == 1 else "Tampered"), confidence
    return "Unknown", 0.0

def predict_hybrid(model, image_path):
    """Predict using hybrid model"""
    ela = prepare_image_ela(image_path)
    hist = prepare_image_hist_eq(image_path)
    lap = prepare_image_laplacian(image_path)
    noise = prepare_image_noise(image_path)
    ela = np.expand_dims(ela, axis=0)
    hist = np.expand_dims(hist, axis=0)
    lap = np.expand_dims(lap, axis=0)
    noise = np.expand_dims(noise, axis=0)
    pred = model.predict((ela, hist, lap, noise))
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    return ("Real" if class_idx == 1 else "Tampered"), confidence

def main():
    print("Image Tampering Detection with Heatmap Visualization")
    image_path = input("Enter the path to the image file: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Load hybrid model for comparison
    hybrid_model_path = "hybrid_ensemble_model.h5"
    if os.path.exists(hybrid_model_path):
        print("Loading hybrid model...")
        with custom_object_scope({'Cast': PassThroughCast}):
            hybrid_model = load_model(hybrid_model_path)
        
        # Make hybrid prediction
        hybrid_prediction, hybrid_confidence = predict_hybrid(hybrid_model, image_path)
        print(f"Hybrid Model: {hybrid_prediction} ({hybrid_confidence*100:.2f}% confidence)")
    
    # Make ELA-only prediction
    ela_prediction, ela_confidence = predict_ela_only(None, image_path)
    print(f"ELA Model: {ela_prediction} ({ela_confidence*100:.2f}% confidence)")

    # Generate tampering heatmap
    print("Generating tampering heatmap...")
    ela_heatmap = generate_ela_heatmap(image_path, target_size=(512, 512))
    
    # Create visualization
    original, heatmap_colored, overlayed = overlay_heatmap_on_image(image_path, ela_heatmap)
    
    # Display results
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 4, 1)
    plt.imshow(original)
    plt.title(f'Original Image')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(ela_heatmap, cmap='hot')
    plt.title('ELA Heatmap\n(Bright = Potential Tampering)')
    plt.axis('off')
    plt.colorbar(shrink=0.8)
    
    plt.subplot(1, 4, 3)
    plt.imshow(overlayed)
    plt.title('Overlay (Original + ELA)')
    plt.axis('off')
    
    # Show ELA difference image for reference
    plt.subplot(1, 4, 4)
    ela_ref = prepare_image_ela(image_path, target_size=(256, 256))
    plt.imshow(ela_ref)
    plt.title('ELA Difference Image\n(Model Input)')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_dir = "ela_heatmap_results"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_ela_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    
    plt.show()
    
    # Display interpretation
    print("\n" + "="*60)
    print("ELA HEATMAP INTERPRETATION:")
    print("="*60)
    print("🔥 Bright/Hot areas: High compression error (potential tampering)")
    print("🔵 Dark/Cool areas: Low compression error (likely authentic)")
    print("⚡ Sharp edges: May show high error but could be natural")
    print("\nELA works by detecting compression inconsistencies:")
    print("- Tampered regions often have different compression levels")
    print("- Copy-paste operations create compression artifacts")
    print("- Authentic images show more uniform compression patterns")
    
    # Analyze tampering indicators
    max_error = np.max(ela_heatmap)
    mean_error = np.mean(ela_heatmap)
    high_error_ratio = np.sum(ela_heatmap > np.percentile(ela_heatmap, 90)) / ela_heatmap.size
    
    print(f"\nELA Analysis Statistics:")
    print(f"Maximum Error Level: {max_error:.3f}")
    print(f"Average Error Level: {mean_error:.3f}")
    print(f"High Error Regions: {high_error_ratio*100:.1f}% of image")
    
    if max_error > 0.3 and high_error_ratio > 0.05:
        print(f"\n⚠️  STRONG TAMPERING INDICATORS DETECTED!")
        print("Focus on the brightest regions in the heatmap.")
    elif max_error > 0.2:
        print(f"\n⚠️  MODERATE TAMPERING INDICATORS detected.")
        print("Some regions show compression inconsistencies.")
    else:
        print(f"\n✅ LOW TAMPERING INDICATORS.")
        print("Image shows relatively uniform compression patterns.")

if __name__ == "__main__":
    main()
