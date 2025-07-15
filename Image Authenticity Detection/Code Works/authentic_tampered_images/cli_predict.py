import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps
import cv2
import tensorflow as tf
from keras.models import load_model
from keras.utils import custom_object_scope

# --- ELA Preprocessing ---
def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp_cli.jpg'
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

# --- Histogram Equalization Preprocessing ---
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

# --- Laplacian Edge Preprocessing ---
from scipy.ndimage import laplace
def convert_to_laplacian_edge(path):
    image = Image.open(path).convert('L')
    arr = np.array(image, dtype=np.float32)
    edge = np.abs(laplace(arr))
    edge = np.clip(edge, 0, 255).astype(np.uint8)
    edge_rgb = np.stack([edge]*3, axis=-1)
    return Image.fromarray(edge_rgb)

def prepare_image_laplacian(image_path, target_size=(64, 64)):
    return np.array(convert_to_laplacian_edge(image_path).resize(target_size)) / 255.0

# --- Noise Extraction Preprocessing ---
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

# --- Pixel Analysis Preprocessing ---
def prepare_image_pixel(image_path, target_size=(64, 64)):
    # Load RGB image
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    arr = np.array(image) / 255.0  # shape (64, 64, 3)
    # Repeat or pad to get 10 channels
    if arr.shape[-1] == 3:
        # Repeat channels to reach 10 (3*3=9, then pad 1)
        arr10 = np.concatenate([arr]*3 + [arr[..., :1]], axis=-1)  # shape (64, 64, 10)
    else:
        arr10 = arr
    return arr10

def predict_pixel(model, image_path):
    img = prepare_image_pixel(image_path)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    return ("Real" if class_idx == 1 else "Tampered"), confidence

# --- Model Prediction Wrappers ---
def predict_ela(model, image_path):
    img = prepare_image_ela(image_path)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    return ("Real" if class_idx == 1 else "Tampered"), confidence

def predict_hist(model, image_path):
    img = prepare_image_hist_eq(image_path)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    return ("Real" if class_idx == 1 else "Tampered"), confidence

def predict_lap(model, image_path):
    img = prepare_image_laplacian(image_path)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    return ("Real" if class_idx == 1 else "Tampered"), confidence

def predict_noise(model, image_path):
    img = prepare_image_noise(image_path)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    return ("Real" if class_idx == 1 else "Tampered"), confidence

def predict_hybrid(model, image_path):
    ela = prepare_image_ela(image_path)
    hist = prepare_image_hist_eq(image_path)
    lap = prepare_image_laplacian(image_path)
    noise = prepare_image_noise(image_path)
    ela = np.expand_dims(ela, axis=0)
    hist = np.expand_dims(hist, axis=0)
    lap = np.expand_dims(lap, axis=0)
    noise = np.expand_dims(noise, axis=0)
    # Pass as tuple to avoid retracing warning
    pred = model.predict((ela, hist, lap, noise))
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    return ("Real" if class_idx == 1 else "Tampered"), confidence

class PassThroughCast(tf.keras.layers.Layer):
    def __init__(self, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self._dtype = dtype
    def call(self, inputs):
        return inputs

def main():
    print("Image Tampering Detection CLI")
    image_path = input("Enter the path to the image file: ").strip()
    models_dir = "."  # All models are in the current directory

    # Model filenames (matching your directory listing)
    ela_model_path = os.path.join(models_dir, "ela_model.h5")
    hist_model_path = os.path.join(models_dir, "histogram_equalization_model.h5")
    lap_model_path = os.path.join(models_dir, "laplacian_edge_model.h5")
    noise_model_path = os.path.join(models_dir, "noise_model.h5")
    pixel_model_path = os.path.join(models_dir, "pixel_analysis_model.h5")
    hybrid_model_path = os.path.join(models_dir, "hybrid_ensemble_model.h5")

    # Load models
    print("Loading models...")
    ela_model = load_model(ela_model_path)
    hist_model = load_model(hist_model_path)
    lap_model = load_model(lap_model_path)
    noise_model = load_model(noise_model_path)
    pixel_model = load_model(pixel_model_path)
    # Register 'Cast' as PassThroughCast to handle dtype argument
    with custom_object_scope({'Cast': PassThroughCast}):
        hybrid_model = load_model(hybrid_model_path)
    print("Models loaded.")

    # Predict with each model
    print(f"\nPredicting for image: {os.path.basename(image_path)}\n")

    result_ela, conf_ela = predict_ela(ela_model, image_path)
    print(f"ELA Model:        {result_ela} ({conf_ela*100:.2f}% confidence)")

    result_hist, conf_hist = predict_hist(hist_model, image_path)
    print(f"HistEq Model:     {result_hist} ({conf_hist*100:.2f}% confidence)")

    result_lap, conf_lap = predict_lap(lap_model, image_path)
    print(f"Laplacian Model:  {result_lap} ({conf_lap*100:.2f}% confidence)")

    result_noise, conf_noise = predict_noise(noise_model, image_path)
    print(f"Noise Model:      {result_noise} ({conf_noise*100:.2f}% confidence)")

    result_pixel, conf_pixel = predict_pixel(pixel_model, image_path)
    print(f"Pixel Model:      {result_pixel} ({conf_pixel*100:.2f}% confidence)")

    result_hybrid, conf_hybrid = predict_hybrid(hybrid_model, image_path)
    print(f"\nHybrid Model:     {result_hybrid} ({conf_hybrid*100:.2f}% confidence)")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
