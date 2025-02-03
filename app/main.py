import os
import json
import gdown  # To download from Google Drive
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# Set paths
MODEL_DIR = "app/trained_model"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_prediction_model.h5")
CLASS_INDICES_PATH = "app/class_indices.json"

# Google Drive File ID
GDRIVE_FILE_ID = "1eJr1kCEng9nliGRWdAK2fanyxw6V9iPP"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if not available
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model file from Google Drive... ⏳")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the trained model
st.info("Loading model... 🚀")
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)
    class_labels = {int(k): v for k, v in class_indices.items()}  # Convert keys to int

# Image processing function
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)  # Resize to model input size
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("🌿 Plant Disease Prediction App")
st.write("Upload a plant leaf image to check if it's healthy or diseased.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    
    # Display result
    st.subheader("Prediction Result 🏷️")
    if "healthy" in predicted_label.lower():
        st.success(f"✅ The plant is **Healthy**! ({predicted_label})")
    else:
        st.error(f"⚠️ The plant has **{predicted_label}**.")
    
    # Show prediction probabilities (optional)
    st.write("### Prediction Probabilities")
    for idx, label in class_labels.items():
        st.write(f"{label}: {prediction[0][idx]:.4f}")

# Footer
st.markdown("---")
st.write("👨‍💻 Built by Afeef Shaikh using TensorFlow & Streamlit")
