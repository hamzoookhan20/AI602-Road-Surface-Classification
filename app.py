import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
import torch  # Required for ViT
from torchvision import transforms # Required for ViT preprocessing

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
@st.cache_resource
def load_models(is_deep):
    if is_deep:
        try:
            # Loading the PyTorch model (you must upload 'rtk_vit_model.pth' to GitHub)
            model = torch.load('rtk_vit_model.pth', map_location=torch.device('cpu'))
            model.eval() # Set to evaluation mode
            return model
        except Exception as e:
            st.error(f"ViT Model not found! Error: {e}")
            return None
    else:
        # Loading your existing Classical ML model
        return joblib.load('rtk_classical_model.pkl')

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
