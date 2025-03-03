import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained U-Net model
@st.cache_resource
def load_unet_model():
    return load_model("lightweight_unet_model.h5")

model = load_unet_model()

# Define class labels
class_names = {0: "Normal", 1: "Tuberculosis"}

# Streamlit UI
st.title("Tuberculosis Detector using U-Net")
st.write("Upload a chest X-ray image to detect Tuberculosis")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded X-ray Image", use_column_width=True)
    
    # Preprocess image
    img = image_pil.resize((128, 128))  # Resize to model input size
    img_array = image.img_to_array(img)
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)  # Convert grayscale to RGB
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Display result
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    
    # Confidence bar chart
    fig, ax = plt.subplots()
    ax.bar(class_names.values(), predictions[0] * 100, color=['green', 'red'])
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)
