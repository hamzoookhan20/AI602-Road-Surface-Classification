import streamlit as st
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
from PIL import Image
from skimage.feature import hog
from torchvision import transforms

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI601: Road Surface Classification", page_icon="🛣️")
st.title("Autonomous Vehicle Perception")
st.markdown("Developed by: **Muhammad Hamza Khan**")

# --- 2. MODEL LOADING FUNCTIONS ---
# Using @st.cache_resource so the app doesn't reload the models every time you click a button
@st.cache_resource
def load_classical_model():
    try:
        return joblib.load('rtk_classical_model.pkl')
    except Exception as e:
        st.error(f"Classical model error: {e}")
        return None

@st.cache_resource
def load_vit_model():
    try:
        # map_location='cpu' is vital for Streamlit Cloud deployment
        model = torch.load('rtk_vit_model.pth', map_location=torch.device('cpu'))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Waiting for ViT model file... (Ensure 'rtk_vit_model.pth' is uploaded via Git LFS)")
        return None
    except Exception as e:
        st.error(f"Error loading ViT model: {e}")
        return None

# --- 3. PREPROCESSING LOGIC ---
def get_hog_features(img):
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img_res = cv2.resize(img_gray, (128, 128))
    features, _ = hog(img_res, orientations=9, pixels_per_cell=(16, 16), 
                      cells_per_block=(2, 2), visualize=True)
    return features.reshape(1, -1)

def transform_for_vit(img):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).unsqueeze(0)

# --- 4. SIDEBAR SETUP ---
with st.sidebar:
    st.header("⚙️ Settings")
    use_deep_learning = st.toggle("Enable Deep Learning (ViT)", value=False)
    
    st.divider()
    if use_deep_learning:
        st.success("Mode: Vision Transformer (Global Context)")
    else:
        st.info("Mode: Random Forest (HOG Features)")

# --- 5. MAIN INTERFACE ---
categories = ['Asphalt', 'Paved', 'Unpaved']
uploaded_file = st.file_uploader("Upload a road image for classification...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Road Scene', use_container_width=True)

    if st.button('Analyze Road Surface'):
        if use_deep_learning:
            # --- DEEP LEARNING PATH ---
            vit = load_vit_model()
            if vit:
                input_tensor = transform_for_vit(image)
                with torch.no_grad():
                    output = vit(input_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = torch.nn.functional.softmax(output, dim=1).max().item()
                
                st.subheader(f"ViT Result: **{categories[prediction]}**")
                st.write(f"Confidence: {confidence:.2%}")
        else:
            # --- CLASSICAL ML PATH ---
            rf = load_classical_model()
            if rf:
                features = get_hog_features(image)
                prediction = rf.predict(features)[0]
                # If your RF supports predict_proba
                try:
                    confidence = np.max(rf.predict_proba(features))
                    st.subheader(f"RF Result: **{categories[prediction]}**")
                    st.write(f"Confidence: {confidence:.2%}")
                except:
                    st.subheader(f"RF Result: **{categories[prediction]}**")

# --- 6. FOOTER ---
st.divider()
st.caption("Project for AI601 Foundations of Artificial Intelligence.")
