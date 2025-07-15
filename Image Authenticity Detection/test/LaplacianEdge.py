import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import laplace
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')

def convert_to_laplacian_edge(path):
    image = Image.open(path).convert('L')
    arr = np.array(image, dtype=np.float32)
    edge = np.abs(laplace(arr))
    edge = np.clip(edge, 0, 255).astype(np.uint8)
    edge_rgb = np.stack([edge]*3, axis=-1)
    return Image.fromarray(edge_rgb)

def prepare_image_laplacian(image_path, target_size=(64, 64)):
    return np.array(convert_to_laplacian_edge(image_path).resize(target_size)) / 255.0

au_path = "/kaggle/input/casia-dataset/CASIA2/Au"
tp_path = "/kaggle/input/casia-dataset/CASIA2/Tp"

X, Y = [], []
print("Loading authentic images...")
for img_name in os.listdir(au_path)[:2000]:
    img_path = os.path.join(au_path, img_name)
    if img_path.lower().endswith(('.jpg', '.png', '.tif')):
        X.append(prepare_image_laplacian(img_path))
        Y.append(1)
print("Loading tampered images...")
for img_name in os.listdir(tp_path)[:2000]:
    img_path = os.path.join(tp_path, img_name)
    if img_path.lower().endswith(('.jpg', '.png', '.tif')):
        X.append(prepare_image_laplacian(img_path))
        Y.append(0)

X = np.array(X)
Y = to_categorical(Y, 2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

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

def display_samples(image_paths, title):
    plt.figure(figsize=(15, 8))
    plt.suptitle(title, fontsize=16)
    for i, img_path in enumerate(image_paths[:5]):
        plt.subplot(2, 5, i+1)
        orig_img = Image.open(img_path)
        plt.imshow(orig_img)
        plt.title(f"Original {i+1}")
        plt.axis('off')
        plt.subplot(2, 5, i+6)
        edge_img = convert_to_laplacian_edge(img_path)
        plt.imshow(edge_img)
        plt.title(f"Laplacian Edge {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

au_images = [os.path.join(au_path, f) for f in os.listdir(au_path) if f.lower().endswith(('.jpg', '.png', '.tif'))][:5]
display_samples(au_images, "Authentic Images - Original vs Laplacian Edge")
tp_images = [os.path.join(tp_path, f) for f in os.listdir(tp_path) if f.lower().endswith(('.jpg', '.png', '.tif'))][:5]
display_samples(tp_images, "Tampered Images - Original vs Laplacian Edge")

def build_model(input_shape=(64, 64, 3)):
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
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax', dtype='float32')
    ])
    return model

model = build_model()
model.summary()
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

print("\nTraining model...")
history = model.fit(
    X_train, Y_train,
    batch_size=64,
    epochs=20,
    validation_data=(X_val, Y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model.save('laplacian_edge_model.h5')
print("Model saved as 'laplacian_edge_model.h5'")

print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

print("\nClassification Report:")
print(classification_report(Y_true, Y_pred_classes, target_names=['Tampered', 'Real']))

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(8, 6))
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
    image = prepare_image_laplacian(image_path)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return "Real" if class_idx == 1 else "Tampered", confidence

print("\nSample predictions:")
for img_path in au_images[:2] + tp_images[:2]:
    prediction, confidence = predict_image(model, img_path)
    print(f"Image: {os.path.basename(img_path)}")
    print(f"  Prediction: {prediction} with {confidence*100:.2f}% confidence")