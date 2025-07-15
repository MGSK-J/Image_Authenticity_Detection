import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance, ImageOps
from scipy.ndimage import laplace
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, Concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import tensorflow as tf
import cv2

# Enable mixed precision for faster training (uses FP16 where possible)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- Preprocessing Functions ---

def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp.jpg'
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

    if method == 'nlm':
        denoised = cv2.fastNlMeansDenoisingColored(image, None, h=8, hColor=8, 
                                                  templateWindowSize=7, searchWindowSize=21)
        denoised = denoised.astype(np.float32) / 255.0
        noise = np.abs(image_float - denoised)
    elif method == 'gaussian':
        denoised = cv2.GaussianBlur(image_float, (5, 5), 1.0)
        noise = np.abs(image_float - denoised)
    elif method == 'bilateral':
        denoised = cv2.bilateralFilter(image, 9, 75, 75).astype(np.float32) / 255.0
        noise = np.abs(image_float - denoised)
    elif method == 'wavelet':
        try:
            import pywt
            noise_channels = []
            for channel in range(3):
                coeffs = pywt.dwt2(image_float[:, :, channel], 'db4')
                coeffs_thresh = (np.zeros_like(coeffs[0]), coeffs[1], coeffs[2], coeffs[3])
                noise_channel = pywt.idwt2(coeffs_thresh, 'db4')
                noise_channels.append(np.abs(noise_channel))
            noise = np.stack(noise_channels, axis=2)
        except ImportError:
            denoised = cv2.GaussianBlur(image_float, (5, 5), 1.0)
            noise = np.abs(image_float - denoised)
    elif method == 'combined':
        denoised_nlm = cv2.fastNlMeansDenoisingColored(image, None, h=8, hColor=8, 
                                                      templateWindowSize=7, searchWindowSize=21)
        denoised_nlm = denoised_nlm.astype(np.float32) / 255.0
        denoised_bilateral = cv2.bilateralFilter(image, 9, 75, 75).astype(np.float32) / 255.0
        denoised_gaussian = cv2.GaussianBlur(image_float, (5, 5), 1.0)
        denoised_avg = (denoised_nlm + denoised_bilateral + denoised_gaussian) / 3.0
        noise = np.abs(image_float - denoised_avg)
    else:
        raise ValueError(f"Unknown method: {method}")

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

# --- Data Preparation ---

au_path = "/kaggle/input/casia-dataset/CASIA2/Au"
tp_path = "/kaggle/input/casia-dataset/CASIA2/Tp"

X_ela, X_hist, X_lap, X_noise, Y = [], [], [], [], []

print("Loading authentic images...")
for img_name in os.listdir(au_path)[:2000]:
    img_path = os.path.join(au_path, img_name)
    if img_path.lower().endswith(('.jpg', '.png', '.tif')):
        X_ela.append(prepare_image_ela(img_path))
        X_hist.append(prepare_image_hist_eq(img_path))
        X_lap.append(prepare_image_laplacian(img_path))
        X_noise.append(prepare_image_noise(img_path))
        Y.append(1)

print("Loading tampered images...")
for img_name in os.listdir(tp_path)[:2000]:
    img_path = os.path.join(tp_path, img_name)
    if img_path.lower().endswith(('.jpg', '.png', '.tif')):
        X_ela.append(prepare_image_ela(img_path))
        X_hist.append(prepare_image_hist_eq(img_path))
        X_lap.append(prepare_image_laplacian(img_path))
        X_noise.append(prepare_image_noise(img_path))
        Y.append(0)

X_ela = np.array(X_ela, dtype=np.float32)
X_hist = np.array(X_hist, dtype=np.float32)
X_lap = np.array(X_lap, dtype=np.float32)
X_noise = np.array(X_noise, dtype=np.float32)
Y = to_categorical(Y, 2)

# Split into Train, Validation, Test
X_ela_train, X_ela_test, X_hist_train, X_hist_test, X_lap_train, X_lap_test, X_noise_train, X_noise_test, Y_train, Y_test = train_test_split(
    X_ela, X_hist, X_lap, X_noise, Y, test_size=0.1, random_state=42)
X_ela_train, X_ela_val, X_hist_train, X_hist_val, X_lap_train, X_lap_val, X_noise_train, X_noise_val, Y_train, Y_val = train_test_split(
    X_ela_train, X_hist_train, X_lap_train, X_noise_train, Y_train, test_size=0.2, random_state=42)

print("\nData shapes:")
print("Train:", X_ela_train.shape, X_hist_train.shape, X_lap_train.shape, X_noise_train.shape, Y_train.shape)
print("Val:", X_ela_val.shape, X_hist_val.shape, X_lap_val.shape, X_noise_val.shape, Y_val.shape)
print("Test:", X_ela_test.shape, X_hist_test.shape, X_lap_test.shape, X_noise_test.shape, Y_test.shape)

# --- Hybrid Model Definition ---

def feature_branch(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.10)(x)  # less dropout
    x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.10)(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.10)(x)
    x = Flatten()(x)
    return inp, x

def build_hybrid_model(input_shape=(64,64,3)):
    inp_ela, feat_ela = feature_branch(input_shape)
    inp_hist, feat_hist = feature_branch(input_shape)
    inp_lap, feat_lap = feature_branch(input_shape)
    inp_noise, feat_noise = feature_branch(input_shape)
    merged = Concatenate()([feat_ela, feat_hist, feat_lap, feat_noise])
    x = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(5e-5))(merged)
    x = Dropout(0.3)(x)  # less dropout
    x = Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(5e-5))(x)
    x = Dropout(0.15)(x)
    out = Dense(2, activation='softmax', dtype='float32')(x)
    model = Model(inputs=[inp_ela, inp_hist, inp_lap, inp_noise], outputs=out)
    return model

model = build_hybrid_model()
model.summary()

optimizer = Adam(learning_rate=0.001)  # restore to 0.001
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss', patience=8, min_delta=0.001, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)

print("\nTraining hybrid model...")
history = model.fit(
    [X_ela_train, X_hist_train, X_lap_train, X_noise_train], Y_train,
    batch_size=64,
    epochs=40,
    validation_data=([X_ela_val, X_hist_val, X_lap_val, X_noise_val], Y_val),
    callbacks=[early_stop, reduce_lr],
    shuffle=True,  # ensure shuffling
    verbose=1
)

model.save('hybrid_ensemble_model.h5')
print("Model saved as 'hybrid_ensemble_model.h5'")

print("\nEvaluating model...")
test_loss, test_acc = model.evaluate([X_ela_test, X_hist_test, X_lap_test, X_noise_test], Y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

Y_pred = model.predict([X_ela_test, X_hist_test, X_lap_test, X_noise_test])
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

print("\nClassification Report:")
print(classification_report(Y_true, Y_pred_classes, target_names=['Tampered', 'Real']))

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(8, 6))
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
plot_confusion_matrix(confusion_mtx, classes=['Tampered', 'Real'])
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

def predict_image(model, image_path):
    ela = prepare_image_ela(image_path)
    hist = prepare_image_hist_eq(image_path)
    lap = prepare_image_laplacian(image_path)
    noise = prepare_image_noise(image_path)
    ela = np.expand_dims(ela, axis=0)
    hist = np.expand_dims(hist, axis=0)
    lap = np.expand_dims(lap, axis=0)
    noise = np.expand_dims(noise, axis=0)
    prediction = model.predict([ela, hist, lap, noise])
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return "Real" if class_idx == 1 else "Tampered", confidence

# Show sample predictions
au_images = [os.path.join(au_path, f) for f in os.listdir(au_path) if f.lower().endswith(('.jpg', '.png', '.tif'))][:2]
tp_images = [os.path.join(tp_path, f) for f in os.listdir(tp_path) if f.lower().endswith(('.jpg', '.png', '.tif'))][:2]
print("\nSample predictions:")
for img_path in au_images + tp_images:
    prediction, confidence = predict_image(model, img_path)
    print(f"Image: {os.path.basename(img_path)}")
    print(f"  Prediction: {prediction} with {confidence*100:.2f}% confidence")
