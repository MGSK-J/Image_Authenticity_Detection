import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import cv2
from scipy import ndimage

# Enable mixed precision for faster training (uses FP16 where possible)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Pixel-level analysis functions
def compute_pixel_statistics(image):
    """Compute pixel-level statistical features"""
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Laplacian (second derivative)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    return gradient_magnitude, laplacian

def noise_analysis(image):
    """Analyze noise patterns in the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply median filter to get denoised version
    denoised = cv2.medianBlur(gray, 5)
    
    # Noise is the difference between original and denoised
    noise = gray.astype(np.float32) - denoised.astype(np.float32)
    
    # Normalize noise
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    
    return noise

def local_binary_pattern(image, radius=1, n_points=8):
    """Compute Local Binary Pattern for texture analysis"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Simple LBP implementation
    h, w = gray.shape
    lbp = np.zeros_like(gray)
    
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = gray[i, j]
            code = 0
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                x = int(i + radius * np.cos(angle))
                y = int(j + radius * np.sin(angle))
                if x < h and y < w and gray[x, y] >= center:
                    code |= (1 << k)
            lbp[i, j] = code
    
    return lbp / 255.0  # Normalize

def pixel_level_analysis(image_path, target_size=(64, 64)):
    """Perform comprehensive pixel-level analysis"""
    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image)
    
    # 1. Original RGB channels
    rgb_normalized = image_array / 255.0
    
    # 2. Gradient analysis
    gradient_mag, laplacian = compute_pixel_statistics(image_array)
    gradient_mag = cv2.resize(gradient_mag, target_size) / 255.0
    laplacian = cv2.resize(laplacian, target_size)
    laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min() + 1e-8)
    
    # 3. Noise analysis
    noise = noise_analysis(image_array)
    noise = cv2.resize(noise, target_size)
    
    # 4. Texture analysis (LBP)
    lbp = local_binary_pattern(image_array)
    lbp = cv2.resize(lbp, target_size)
    
    # 5. Color channel statistics
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    hsv = cv2.resize(hsv, target_size) / 255.0
    
    # Combine all features into multi-channel representation
    # Stack: RGB(3) + Gradient(1) + Laplacian(1) + Noise(1) + LBP(1) + HSV(3) = 10 channels
    features = np.dstack([
        rgb_normalized,                    # RGB channels (3)
        gradient_mag[..., np.newaxis],     # Gradient magnitude (1)
        laplacian[..., np.newaxis],        # Laplacian (1)
        noise[..., np.newaxis],            # Noise (1)
        lbp[..., np.newaxis],              # LBP texture (1)
        hsv                                # HSV channels (3)
    ])
    
    return features

def prepare_image_pixel_analysis(image_path, target_size=(64, 64)):
    """Wrapper function for pixel-level analysis"""
    return pixel_level_analysis(image_path, target_size)

# Define paths
au_path = "/kaggle/input/casia-dataset/CASIA2/Au"  # Authentic images
tp_path = "/kaggle/input/casia-dataset/CASIA2/Tp"  # Tampered images

# Initialize lists
X = []  # Images
Y = []  # Labels (0=Tampered, 1=Authentic)

print("Loading authentic images with pixel-level analysis...")
for img_name in os.listdir(au_path)[:2000]:  # Limit to 2000 for faster processing
    img_path = os.path.join(au_path, img_name)
    if img_path.lower().endswith(('.jpg', '.png', '.tif')):
        try:
            features = prepare_image_pixel_analysis(img_path)
            X.append(features)
            Y.append(1)  # Label 1 = Authentic
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

print("Loading tampered images with pixel-level analysis...")
for img_name in os.listdir(tp_path)[:2000]:  # Limit to 2000 for faster processing
    img_path = os.path.join(tp_path, img_name)
    if img_path.lower().endswith(('.jpg', '.png', '.tif')):
        try:
            features = prepare_image_pixel_analysis(img_path)
            X.append(features)
            Y.append(0)  # Label 0 = Tampered
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

# Convert to numpy arrays
X = np.array(X)
Y = to_categorical(Y, 2)  # One-hot encoding

print(f"Final dataset shape: {X.shape}")
print(f"Feature channels: RGB(3) + Gradient(1) + Laplacian(1) + Noise(1) + LBP(1) + HSV(3) = {X.shape[-1]} channels")

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

# Enhanced visualization for pixel-level features
def display_pixel_analysis_samples(image_paths, title):
    plt.figure(figsize=(20, 12))
    plt.suptitle(title, fontsize=16)
    
    for i, img_path in enumerate(image_paths[:3]):
        features = pixel_level_analysis(img_path)
        
        # Original Image (RGB)
        plt.subplot(4, 3, i+1)
        plt.imshow(features[:, :, :3])
        plt.title(f"Original RGB {i+1}")
        plt.axis('off')
        
        # Gradient Magnitude
        plt.subplot(4, 3, i+4)
        plt.imshow(features[:, :, 3], cmap='gray')
        plt.title(f"Gradient {i+1}")
        plt.axis('off')
        
        # Noise Pattern
        plt.subplot(4, 3, i+7)
        plt.imshow(features[:, :, 5], cmap='gray')
        plt.title(f"Noise {i+1}")
        plt.axis('off')
        
        # LBP Texture
        plt.subplot(4, 3, i+10)
        plt.imshow(features[:, :, 6], cmap='gray')
        plt.title(f"LBP Texture {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Show sample analysis
au_images = [os.path.join(au_path, f) for f in os.listdir(au_path) 
             if f.lower().endswith(('.jpg', '.png', '.tif'))][:3]
display_pixel_analysis_samples(au_images, "Authentic Images - Pixel-Level Analysis")

tp_images = [os.path.join(tp_path, f) for f in os.listdir(tp_path) 
             if f.lower().endswith(('.jpg', '.png', '.tif'))][:3]
display_pixel_analysis_samples(tp_images, "Tampered Images - Pixel-Level Analysis")

# Enhanced model architecture for multi-channel input
def build_pixel_analysis_model(input_shape=(64, 64, 10)):
    model = Sequential([
        # First block - handle multi-channel input
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fourth block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        
        # Classifier
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax', dtype='float32')  # Ensure output is float32
    ])
    return model

# Build model
model_pixel = build_pixel_analysis_model(input_shape=X.shape[1:])
model_pixel.summary()

# Optimizer with learning rate schedule
optimizer = Adam(learning_rate=0.0005)  # Slightly lower learning rate for complex features
model_pixel.compile(optimizer=optimizer, 
                   loss='categorical_crossentropy', 
                   metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7)

# Train model
print("\nTraining pixel-level analysis model...")
history_pixel = model_pixel.fit(
    X_train, Y_train,
    batch_size=32,  # Smaller batch size due to larger input
    epochs=30,
    validation_data=(X_val, Y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Save model
model_pixel.save('pixel_analysis_model.h5')
print("Model saved as 'pixel_analysis_model.h5'")

# Evaluate model
print("\nEvaluating model...")
test_loss, test_acc = model_pixel.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
Y_pred = model_pixel.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_true, Y_pred_classes, target_names=['Tampered', 'Real']))

# Confusion Matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(confusion_mtx, classes=['Tampered', 'Real'])
plt.title('Confusion Matrix - Pixel-Level Analysis')
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history_pixel.history['accuracy'], label='Train Accuracy')
plt.plot(history_pixel.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy - Pixel-Level Analysis')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history_pixel.history['loss'], label='Train Loss')
plt.plot(history_pixel.history['val_loss'], label='Val Loss')
plt.title('Model Loss - Pixel-Level Analysis')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# Sample predictions
def predict_image_pixel(model, image_path):
    features = prepare_image_pixel_analysis(image_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return "Real" if class_idx == 1 else "Tampered", confidence

# Test on sample images
print("\nSample predictions with pixel-level analysis:")
for img_path in au_images[:2] + tp_images[:2]:
    prediction, confidence = predict_image_pixel(model_pixel, img_path)
    print(f"Image: {os.path.basename(img_path)}")
    print(f"  Prediction: {prediction} with {confidence*100:.2f}% confidence")

# Feature importance analysis
def analyze_feature_channels():
    """Analyze which feature channels are most important"""
    print("\nFeature Channel Analysis:")
    print("Channel 0-2: RGB color information")
    print("Channel 3: Gradient magnitude (edge detection)")
    print("Channel 4: Laplacian (second derivatives)")
    print("Channel 5: Noise patterns")
    print("Channel 6: Local Binary Pattern (texture)")
    print("Channel 7-9: HSV color space")
    
    # You can extend this to show actual feature importance if needed

analyze_feature_channels()