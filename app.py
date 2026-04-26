import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
import torch  # Required for ViT
from torchvision import transforms # Required for ViT preprocessing
import os
import torch

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI602: Road Surface Classifier", page_icon="🛣️")
st.title("🛣️ Autonomous Vehicle Perception")
st.markdown("**Student:** Muhammad Hamza Khan")

# --- 2. SIDEBAR TOGGLE ---
with st.sidebar:
    st.header("⚙️ Model Settings")
    # This creates the switch for the user
    use_deep_learning = st.toggle("Enable Deep Learning (ViT)", value=False)
    
    if use_deep_learning:
        st.success("🚀 Mode: Vision Transformer")
    else:
        st.info("📊 Mode: Random Forest")

# --- 3. DYNAMIC MODEL LOADING ---
# --- STEP 3: PROFESSIONAL MODEL LOADER ---
@st.cache_resource
def load_vit_model():
    # This is your specific File ID from Google Drive
    file_id = '1a7UtWhQ0nMZEGhRrHqFcdpYIyuzkS2vU'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'rtk_vit_model.pth'

    # Download only if the file doesn't exist in the current session
    if not os.path.exists(output):
        with st.status("📥 Initializing Deep Learning Model (330MB)..."):
            st.write("Connecting to Model Registry on Google Drive...")
            # gdown handles the large file download seamlessly
            gdown.download(url, output, quiet=False, fuzzy=True)
            st.write("✅ Download Complete!")

    # Load the PyTorch model
    # map_location='cpu' ensures it works on Streamlit's servers without a GPU
    model = torch.load(output, map_location=torch.device('cpu'))
    model.eval()
    return model

# --- USAGE IN THE APP ---
# When the user flips the toggle, this function is called
if use_deep_learning:
    model_vit = load_vit_model()
    # Now use model_vit for your predictions
    
model = load_models(use_deep_learning)
categories = ['Asphalt', 'Paved', 'Unpaved']

# --- 4. IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Upload a road image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Road Scene', use_container_width=True)
    
    if st.button('Analyze Road Surface'):
        if use_deep_learning:
            # --- DEEP LEARNING PREPROCESSING (ViT) ---
            # ViT needs 224x224 and specific normalization
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(image).unsqueeze(0) # Add batch dimension
            
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                # Simple confidence mock for ViT (or use Softmax)
                confidence = torch.nn.functional.softmax(output, dim=1).max().item()
        else:
            # --- CLASSICAL ML PREPROCESSING (Random Forest) ---
            img_array = np.array(image.convert('RGB'))
            img_resized = cv2.resize(img_array, (128, 128))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            features = hog(img_gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
            
            prediction = model.predict(features.reshape(1, -1))[0]
            confidence = np.max(model.predict_proba(features.reshape(1, -1)))

        # --- DISPLAY RESULTS ---
        st.success(f"**Prediction:** {categories[prediction]}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.progress(float(confidence))
