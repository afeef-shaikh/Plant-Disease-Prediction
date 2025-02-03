import os
import json
import gdown  # To download from Google Drive
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Define model path
MODEL_PATH = "app/trained_model/plant_disease_prediction_model.h5"

# Google Drive file ID (Extracted from the sharing link)
GDRIVE_MODEL_URL = "https://drive.google.com/uc?id=1eJr1kCEng9nliGRWdAK2fanyxw6V9iPP"

# Download the model if it's not already present
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model from Google Drive...")
    gdown.download(GDRIVE_MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
CLASS_INDICES_PATH = "app/class_indices.json"
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)
    class_labels = {int(k): v for k, v in class_indices.items()}

# Function to preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
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

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    st.subheader("Prediction Result 🏷️")
    if "healthy" in predicted_label.lower():
        st.success(f"✅ The plant is **Healthy**! ({predicted_label})")
    else:
        st.error(f"⚠️ The plant has **{predicted_label}**.")

    st.write("### Prediction Probabilities")
    for idx, label in class_labels.items():
        st.write(f"{label}: {prediction[0][idx]:.4f}")

st.markdown("---")
st.write("👨‍💻 Built by Afeef Shaikh using TensorFlow & Streamlit")
