import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps
import cv2
import tensorflow as tf
from keras.models import load_model
from keras.utils import custom_object_scope
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from scipy.ndimage import laplace
import tempfile

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- AI Detection Preprocessing ---
def prepare_image_ai_detection(image_path, target_size=(512, 512)):
    """Prepare image for AI vs Human detection"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return img_array

def predict_ai_detection(model, image_path):
    """Predict if image is AI-generated or human-created"""
    img = prepare_image_ai_detection(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    pred = model.predict(img)
    probability = pred[0][0]
    class_name = "AI" if probability > 0.5 else "Human"
    confidence = probability if probability > 0.5 else (1 - probability)
    return class_name, confidence

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
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    arr = np.array(image) / 255.0
    if arr.shape[-1] == 3:
        arr10 = np.concatenate([arr]*3 + [arr[..., :1]], axis=-1)
    else:
        arr10 = arr
    return arr10

# --- Prediction Functions ---
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

def predict_pixel(model, image_path):
    img = prepare_image_pixel(image_path)
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

# Load models at startup
print("Loading models...")
models = {}
try:
    # Load AI detection model first
    models['ai_detector'] = tf.keras.models.load_model(os.path.join(MODELS_FOLDER, "ai_vs_human_resnet50.keras"))
    
    # Load tampering detection models
    models['ela'] = load_model(os.path.join(MODELS_FOLDER, "ela_model.h5"))
    models['hist'] = load_model(os.path.join(MODELS_FOLDER, "histogram_equalization_model.h5"))
    models['lap'] = load_model(os.path.join(MODELS_FOLDER, "laplacian_edge_model.h5"))
    models['noise'] = load_model(os.path.join(MODELS_FOLDER, "noise_model.h5"))
    models['pixel'] = load_model(os.path.join(MODELS_FOLDER, "pixel_analysis_model.h5"))
    with custom_object_scope({'Cast': PassThroughCast}):
        models['hybrid'] = load_model(os.path.join(MODELS_FOLDER, "hybrid_ensemble_model.h5"))
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Step 1: Check if image is AI-generated
            ai_prediction, ai_confidence = predict_ai_detection(models['ai_detector'], filepath)
            
            results = {
                'ai_detection': (ai_prediction, ai_confidence)
            }
            
            # Step 2: If image is human-generated, check for tampering
            if ai_prediction == "Human":
                results['ela'] = predict_ela(models['ela'], filepath)
                results['hist'] = predict_hist(models['hist'], filepath)
                results['lap'] = predict_lap(models['lap'], filepath)
                results['noise'] = predict_noise(models['noise'], filepath)
                results['pixel'] = predict_pixel(models['pixel'], filepath)
                results['hybrid'] = predict_hybrid(models['hybrid'], filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('results.html', results=results, filename=filename)
            
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload an image file.')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Step 1: AI Detection
        ai_prediction, ai_confidence = predict_ai_detection(models['ai_detector'], filepath)
        
        formatted_results = {
            'ai_detection': {
                'prediction': ai_prediction,
                'confidence': float(ai_confidence)
            }
        }
        
        # Step 2: Tampering detection only for human images
        if ai_prediction == "Human":
            tampering_results = {}
            tampering_results['ela'] = predict_ela(models['ela'], filepath)
            tampering_results['hist'] = predict_hist(models['hist'], filepath)
            tampering_results['lap'] = predict_lap(models['lap'], filepath)
            tampering_results['noise'] = predict_noise(models['noise'], filepath)
            tampering_results['pixel'] = predict_pixel(models['pixel'], filepath)
            tampering_results['hybrid'] = predict_hybrid(models['hybrid'], filepath)
            
            # Format tampering results
            for model_name, (prediction, confidence) in tampering_results.items():
                formatted_results[model_name] = {
                    'prediction': prediction,
                    'confidence': float(confidence)
                }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(formatted_results)
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
