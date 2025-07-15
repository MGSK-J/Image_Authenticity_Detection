from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pickle
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_image(model, image_path, img_size=(512, 512)):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    class_name = "AI" if probability > 0.5 else "Human"
    
    # Display results
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {class_name} (Confidence: {probability:.2f})")
    plt.show()
    
    return class_name, probability

def cli_application():
    # Load the saved model (updated extension)
    try:
        loaded_model = tf.keras.models.load_model('ai_vs_human_resnet50.keras')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train and save the model first.")
        return
    
    # Rest of the CLI code remains the same...
    parser = argparse.ArgumentParser(description='AI vs Human Image Classifier')
    parser.add_argument('image_path', help='Path to the image file to classify')
    args = parser.parse_args()
    
    class_name, confidence = predict_image(loaded_model, args.image_path)
    print(f"\nResult: This image was created by {class_name}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == '__main__':
    cli_application()