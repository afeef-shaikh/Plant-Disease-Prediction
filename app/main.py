import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Get the absolute path of the current working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths
MODEL_PATH = os.path.join(working_dir, "trained_model/plant_disease_prediction_model.h5")
CLASS_INDICES_PATH = os.path.join(working_dir, "class_indices.json")

# Verify if the model file exists before loading
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at {MODEL_PATH}. Please check the path.")
    st.stop()

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)
    class_labels = {int(k): v for k, v in class_indices.items()}  # Convert keys to int

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
    predicted_label = class_labels.get(predicted_class, "Unknown")

    st.subheader("Prediction Result 🏷️")
    if "healthy" in predicted_label.lower():
        st.success(f"✅ The plant is **Healthy**! ({predicted_label})")
    else:
        st.error(f"⚠️ The plant is NOT Healthy :( \n It has **{predicted_label}** disease.")

# Footer
st.markdown("---")
st.write("👨‍💻 Built by Afeef Shaikh using TensorFlow & Streamlit")
