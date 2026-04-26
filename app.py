import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
import torch
import torch.nn as nn
from torchvision import transforms, models

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI602: Road Surface Classifier", page_icon="🛣️")

# --- 2. MODEL LOADING FUNCTION ---
# We define this at the top to avoid NameErrors
@st.cache_resource
def load_models(is_deep):
    if is_deep:
        try:
            # For ViT, we load the weights into the architecture
            # Make sure 'rtk_vit_model.pth' is uploaded to your GitHub
            model = torch.load('rtk_vit_model.pth', map_location=torch.device('cpu'))
            if isinstance(model, torch.nn.Module):
                model.eval()
                return model
            else:
                st.error("The loaded file is not a valid PyTorch model.")
                return None
        except Exception as e:
            st.error(f"Waiting for ViT model file... (Error: {e})")
            return None
    else:
        try:
            # Load your existing Classical Random Forest model
            return joblib.load('rtk_classical_model.pkl')
        except Exception as e:
            st.error(f"Classical model not found! (Error: {e})")
            return None

# --- 3. SIDEBAR NAVIGATION & TOGGLE ---
with st.sidebar:
    st.title("Settings")
    # This is the toggle you requested for Deep Learning
    use_deep_learning = st.toggle("Enable Deep Learning (ViT)", value=False)
    
    st.divider()
    if use_deep_learning:
        st.success("Current Mode: Vision Transformer")
        st.info("Using Self-Attention for Global Context")
    else:
        st.info("Current Mode: Random Forest")
        st.info("Using HOG Features for Texture Analysis")

# Load the selected model
model = load_models(use_deep_learning)
categories = ['Asphalt', 'Paved', 'Unpaved']

# --- 4. MAIN USER INTERFACE ---
st.title("🛣️ Autonomous Vehicle Perception")
st.markdown(f"**Researcher:** Muhammad Hamza Khan")
st.write("---")

uploaded_file = st.file_uploader("Upload a road image for classification...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image immediately
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Road Scene', use_container_width=True)
    
    if st.button('Analyze Road Surface'):
        if model is None:
            st.error("Please ensure the model files are uploaded to GitHub.")
        else:
            with st.spinner('Calculating...'):
                try:
                    if use_deep_learning:
                        # --- ViT PREPROCESSING ---
                        preprocess = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                        input_tensor = preprocess(image).unsqueeze(0)
                        
                        with torch.no_grad():
                            output = model(input_tensor)
                            prediction = torch.argmax(output, dim=1).item()
                            probs = torch.nn.functional.softmax(output, dim=1)
                            confidence = torch.max(probs).item()
                    else:
                        # --- RANDOM FOREST PREPROCESSING ---
                        img_array = np.array(image.convert('RGB'))
                        img_resized = cv2.resize(img_array, (128, 128))
                        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
                        # Extract HOG features
                        features = hog(img_gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
                        
                        prediction = model.predict(features.reshape(1, -1))[0]
                        confidence = np.max(model.predict_proba(features.reshape(1, -1)))

                    # --- DISPLAY RESULTS ---
                    st.divider()
                    st.subheader(f"Result: {categories[prediction]}")
                    st.write(f"**Confidence Score:** {confidence*100:.2f}%")
                    st.progress(float(confidence))
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

# --- 5. FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("AI601 Term Project")
