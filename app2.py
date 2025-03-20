import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time

# Set page configuration
st.set_page_config(
    page_title="Tuberculosis Detection from Chest X-rays",
    page_icon="🫁",
    layout="wide",
)

# App title and description
st.title("Tuberculosis Detection from Chest X-rays")
st.markdown("""
This application uses deep learning models (VGG16 and ResNet50) to detect tuberculosis from chest X-ray images.
Upload an X-ray image to get predictions from both models.
""")

# Create sidebar for model selection
st.sidebar.title("Model Options")
selected_model = st.sidebar.radio(
    "Choose model for primary analysis:",
    ("VGG16", "ResNet50", "Both (Compare)")
)

# Function to load models
@st.cache_resource
def load_models():
    try:
        vgg_model = load_model("vgg16_model.h5")
        resnet_model = load_model("resnet50_model.h5")
        return vgg_model, resnet_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
with st.spinner("Loading models..."):
    vgg_model, resnet_model = load_models()

# Class indices
class_indices = {'Normal': 0, 'Tuberculosis': 1}
# Invert the dictionary to map indices to class names
class_names = {v: k for k, v in class_indices.items()}

# Function to preprocess the image
def preprocess_image(img):
    # Resize image
    img = img.resize((224, 224))
    # Convert to array and add batch dimension
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale
    img_array = img_array / 255.0
    return img_array

# Function to make prediction
def predict(img, model):
    # Preprocess the image
    preprocessed_img = preprocess_image(img)
    # Make prediction
    prediction = model.predict(preprocessed_img)
    # Get class index with highest probability
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    # Get confidence score
    confidence = prediction[0][predicted_class_index] * 100
    return predicted_class_index, confidence, prediction[0]

# Function to display prediction results
def display_prediction(prediction_results, model_name):
    predicted_class_index, confidence, raw_predictions = prediction_results
    predicted_class = class_names[predicted_class_index]
    
    # Create a styled container based on the prediction
    if predicted_class == "Normal":
        result_container = st.container(border=True)
        with result_container:
            st.markdown(f"### {model_name} Model Prediction: 🟢 **{predicted_class}**")
            st.markdown(f"**Confidence**: {confidence:.2f}%")
    else:
        result_container = st.container(border=True)
        with result_container:
            st.markdown(f"### {model_name} Model Prediction: 🔴 **{predicted_class}**")
            st.markdown(f"**Confidence**: {confidence:.2f}%")
    
    # Display raw probabilities
    st.write("Class probabilities:")
    probs_df = {class_names[i]: f"{prob*100:.2f}%" for i, prob in enumerate(raw_predictions)}
    st.json(probs_df)
    
    return predicted_class, confidence

# Function to create a comparison bar chart
def plot_comparison(vgg_preds, resnet_preds):
    labels = list(class_names.values())
    vgg_values = [vgg_preds[0], vgg_preds[1]]
    resnet_values = [resnet_preds[0], resnet_preds[1]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    vgg_bars = ax.bar(x - width/2, vgg_values, width, label='VGG16', color='skyblue')
    resnet_bars = ax.bar(x + width/2, resnet_values, width, label='ResNet50', color='lightcoral')
    
    ax.set_ylabel('Probability')
    ax.set_title('Model Predictions Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add probability values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(vgg_bars)
    autolabel(resnet_bars)
    
    plt.ylim(0, 1.0)
    st.pyplot(fig)

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        img = Image.open(uploaded_file)
        st.image(img, width=400)
    
    # Add a button to trigger prediction
    if st.button("Run Diagnosis"):
        with st.spinner("Analyzing image..."):
            time.sleep(1)  # Simulate processing time
            
            if selected_model == "VGG16" or selected_model == "Both (Compare)":
                with col2:
                    if vgg_model is not None:
                        vgg_results = predict(img, vgg_model)
                        vgg_class, vgg_conf = display_prediction(vgg_results, "VGG16")
            
            if selected_model == "ResNet50" or selected_model == "Both (Compare)":
                with col2:
                    if resnet_model is not None:
                        resnet_results = predict(img, resnet_model)
                        resnet_class, resnet_conf = display_prediction(resnet_results, "ResNet50")
            
            # If both models are selected, show comparison
            if selected_model == "Both (Compare)":
                st.subheader("Model Comparison")
                plot_comparison(vgg_results[2], resnet_results[2])
                
                # Show agreement analysis
                st.subheader("Model Agreement Analysis")
                if vgg_class == resnet_class:
                    st.success(f"✅ Both models agree on the diagnosis: **{vgg_class}**")
                    avg_conf = (vgg_conf + resnet_conf) / 2
                    st.markdown(f"Average confidence: **{avg_conf:.2f}%**")
                else:
                    st.warning("⚠️ Models disagree on the diagnosis")
                    st.markdown(f"- VGG16 predicts: **{vgg_class}** with {vgg_conf:.2f}% confidence")
                    st.markdown(f"- ResNet50 predicts: **{resnet_class}** with {resnet_conf:.2f}% confidence")
                    st.markdown("Consider consulting with a medical professional for a definitive diagnosis.")

# Add information about the model
with st.expander("About  the Models"):
    st.markdown("""
    ### Model Information
    This application uses two pre-trained deep learning models:
    
    1. **VGG16**: A convolutional neural network model proposed by the Visual Geometry Group (VGG) at the University of Oxford. The model achieves high accuracy on image classification tasks.
    
    2. **ResNet50**: A residual neural network that is 50 layers deep. It introduces the concept of skip connections, which helps to train deeper networks without the vanishing gradient problem.
    
    Both models have been fine-tuned on a dataset of chest X-ray images to detect tuberculosis.
    
    ### Classes
    - **Normal**: No signs of tuberculosis
    - **Tuberculosis**: Signs of tuberculosis infection
    
    ### Disclaimer
    This tool is for educational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Developed for medical imaging analysis</p>
    </div>
    """, 
    unsafe_allow_html=True
)