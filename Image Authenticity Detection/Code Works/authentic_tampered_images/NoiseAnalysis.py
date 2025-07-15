import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Noise extraction function
def extract_noise(image_path, method='combined', enhanced=True):
    """
    Enhanced noise residual extraction from an image
    
    Args:
        image_path: Path to the image file
        method: 'nlm', 'gaussian', 'bilateral', 'combined', or 'wavelet'
        enhanced: Whether to apply additional enhancement techniques
    
    Returns:
        Noise residual as float32 array normalized to [0, 1]
    """
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros((64, 64, 3), dtype=np.float32)
    
    # Convert to RGB and resize with better interpolation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to float32 for better precision
    image_float = image.astype(np.float32) / 255.0
    
    if method == 'nlm':
        # Optimized Non-local Means denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, h=8, hColor=8, 
                                                  templateWindowSize=7, searchWindowSize=21)
        denoised = denoised.astype(np.float32) / 255.0
        noise = np.abs(image_float - denoised)
        
    elif method == 'gaussian':
        # Gaussian blur for noise extraction
        denoised = cv2.GaussianBlur(image_float, (5, 5), 1.0)
        noise = np.abs(image_float - denoised)
        
    elif method == 'bilateral':
        # Bilateral filter preserves edges better
        denoised = cv2.bilateralFilter(image, 9, 75, 75).astype(np.float32) / 255.0
        noise = np.abs(image_float - denoised)
        
    elif method == 'wavelet':
        # Wavelet-based noise extraction (requires pywt)
        try:
            import pywt
            noise_channels = []
            for channel in range(3):
                coeffs = pywt.dwt2(image_float[:, :, channel], 'db4')
                # Zero out the approximation coefficients to keep only details
                coeffs_thresh = (np.zeros_like(coeffs[0]), coeffs[1], coeffs[2], coeffs[3])
                noise_channel = pywt.idwt2(coeffs_thresh, 'db4')
                noise_channels.append(np.abs(noise_channel))
            noise = np.stack(noise_channels, axis=2)
        except ImportError:
            # Fallback to Gaussian if pywt not available
            denoised = cv2.GaussianBlur(image_float, (5, 5), 1.0)
            noise = np.abs(image_float - denoised)
            
    elif method == 'combined':
        # Combine multiple denoising methods for robust noise extraction
        # Non-local means
        denoised_nlm = cv2.fastNlMeansDenoisingColored(image, None, h=8, hColor=8, 
                                                      templateWindowSize=7, searchWindowSize=21)
        denoised_nlm = denoised_nlm.astype(np.float32) / 255.0
        
        # Bilateral filter
        denoised_bilateral = cv2.bilateralFilter(image, 9, 75, 75).astype(np.float32) / 255.0
        
        # Gaussian blur
        denoised_gaussian = cv2.GaussianBlur(image_float, (5, 5), 1.0)
        
        # Average the denoised versions
        denoised_avg = (denoised_nlm + denoised_bilateral + denoised_gaussian) / 3.0
        noise = np.abs(image_float - denoised_avg)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if enhanced:
        # Apply enhancement techniques
        # 1. Contrast enhancement using CLAHE
        noise_enhanced = np.zeros_like(noise)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        
        for channel in range(3):
            # Convert to uint8 for CLAHE
            noise_uint8 = (noise[:, :, channel] * 255).astype(np.uint8)
            noise_enhanced[:, :, channel] = clahe.apply(noise_uint8).astype(np.float32) / 255.0
        
        # 2. Gamma correction to enhance weak signals
        gamma = 0.7
        noise_enhanced = np.power(noise_enhanced, gamma)
        
        # 3. High-pass filter to emphasize high-frequency noise
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 9.0
        
        noise_filtered = np.zeros_like(noise_enhanced)
        for channel in range(3):
            filtered = cv2.filter2D(noise_enhanced[:, :, channel], -1, kernel)
            noise_filtered[:, :, channel] = np.clip(filtered, 0, 1)
        
        noise = noise_filtered
    
    # Ensure values are in [0, 1] range
    noise = np.clip(noise, 0, 1)
    
    return noise.astype(np.float32)

def prepare_image_noise(image_path):
    """Prepare noise image for model input"""
    return extract_noise(image_path)

# Define paths
au_path = "/kaggle/input/casia-dataset/CASIA2/Au"  # Authentic images
tp_path = "/kaggle/input/casia-dataset/CASIA2/Tp"  # Tampered images

# Initialize lists
X = []  # Images
Y = []  # Labels (0=Tampered, 1=Authentic)

print("Loading authentic images...")
for img_name in os.listdir(au_path)[:2000]:  # Limit to 2000 for faster processing
    img_path = os.path.join(au_path, img_name)
    if img_path.lower().endswith(('.jpg', '.png', '.tif')):
        X.append(prepare_image_noise(img_path))
        Y.append(1)  # Label 1 = Authentic

print("Loading tampered images...")
for img_name in os.listdir(tp_path)[:2000]:  # Limit to 2000 for faster processing
    img_path = os.path.join(tp_path, img_name)
    if img_path.lower().endswith(('.jpg', '.png', '.tif')):
        X.append(prepare_image_noise(img_path))
        Y.append(0)  # Label 0 = Tampered

# Convert to numpy arrays
X = np.array(X)
Y = to_categorical(Y, 2)  # One-hot encoding

# Split into Train, Validation, Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

print("\nData shapes:")
print("Train:", X_train.shape, Y_train.shape)
print("Val:", X_val.shape, Y_val.shape)
print("Test:", X_test.shape, Y_test.shape)

# Visualization function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Show sample images
def display_samples(image_paths, title):
    plt.figure(figsize=(15, 8))
    plt.suptitle(title, fontsize=16)
    
    for i, img_path in enumerate(image_paths[:5]):
        # Original Image
        plt.subplot(2, 5, i+1)
        orig_img = Image.open(img_path)
        plt.imshow(orig_img)
        plt.title(f"Original {i+1}")
        plt.axis('off')
        
        # Noise Residual Image
        plt.subplot(2, 5, i+6)
        noise_img = extract_noise(img_path)
        # For visualization, we need to scale noise to 0-255 range
        noise_vis = (noise_img * 255).astype(np.uint8)
        plt.imshow(noise_vis)
        plt.title(f"Noise {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Authentic images
au_images = [os.path.join(au_path, f) for f in os.listdir(au_path) 
             if f.lower().endswith(('.jpg', '.png', '.tif'))][:5]
display_samples(au_images, "Authentic Images - Original vs Noise Residual")

# Tampered images
tp_images = [os.path.join(tp_path, f) for f in os.listdir(tp_path) 
             if f.lower().endswith(('.jpg', '.png', '.tif'))][:5]
display_samples(tp_images, "Tampered Images - Original vs Noise Residual")

# Model architecture optimized for noise analysis
def build_noise_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax', dtype='float32')
    ])
    return model

# Build model
model_noise = build_noise_model()
model_noise.summary()

# Optimizer with learning rate schedule
optimizer = Adam(learning_rate=0.001)
model_noise.compile(optimizer=optimizer, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train model
print("\nTraining model...")
history_noise = model_noise.fit(
    X_train, Y_train,
    batch_size=64,
    epochs=20,
    validation_data=(X_val, Y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Save model
model_noise.save('noise_model.h5')
print("Model saved as 'noise_model.h5'")

# Evaluate model
print("\nEvaluating model...")
test_loss, test_acc = model_noise.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
Y_pred = model_noise.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_true, Y_pred_classes, target_names=['Tampered', 'Real']))

# Confusion Matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(confusion_mtx, classes=['Tampered', 'Real'])
plt.title('Confusion Matrix')
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history_noise.history['accuracy'], label='Train Accuracy')
plt.plot(history_noise.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history_noise.history['loss'], label='Train Loss')
plt.plot(history_noise.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# Sample predictions
def predict_image(model, image_path):
    image = prepare_image_noise(image_path)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return "Real" if class_idx == 1 else "Tampered", confidence

# Test on sample images
print("\nSample predictions:")
for img_path in au_images[:2] + tp_images[:2]:
    prediction, confidence = predict_image(model_noise, img_path)
    print(f"Image: {os.path.basename(img_path)}")
    print(f"  Prediction: {prediction} with {confidence*100:.2f}% confidence")