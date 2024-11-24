import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load pre-trained MobileNetV2 model + weights
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess the image
def preprocess_image(img_path):
    # Load the image with target size 224x224 (MobileNetV2 input size)
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert image to array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to simulate a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image (apply same preprocessing as MobileNetV2)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_array

# Function to make predictions
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    
    # Get the model prediction
    predictions = model.predict(img_array)
    
    # Decode the predictions to human-readable labels
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    
    # Print predictions
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}. {label}: {score:.2f}")

# Example usage
if __name__ == "__main__":
    # Set the path to your image file
    img_path = 'Image Classifier/Dog.jpg'  # Replace with your image path

    # Check if the image file exists
    if os.path.exists(img_path):
        print(f"Predicting the image: {img_path}")
        predict_image(img_path)
    else:
        print(f"Image file does not exist: {img_path}")
