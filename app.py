import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog

# 1. Page Configuration (App Title and Icon)
st.set_page_config(page_title="AI602: Road Surface Classifier", page_icon="🛣️")

# Header Section
st.title("🛣️ Autonomous Vehicle Perception")
st.subheader("Classical ML: Road Surface Recognition")
st.markdown("""
This application uses a **Random Forest** model with **HOG features** to classify road textures.
**Student:** Muhammad Hamza Khan
""")

# 2. Load the Saved Model (Cached for performance)
@st.cache_resource
def load_trained_model():
    # Ensure this filename matches exactly what you saved in Phase 1
    return joblib.load('rtk_classical_model.pkl')

try:
    model = load_trained_model()
    st.sidebar.success("✅ Model Loaded Successfully")
except:
    st.sidebar.error("❌ Model file 'rtk_classical_model.pkl' not found.")

# Define the category names for display
categories = ['Asphalt', 'Paved', 'Unpaved']

# 3. Image Upload Section
uploaded_file = st.file_uploader("Upload a road image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Road Scene', use_column_width=True)
    
    # 4. Preprocessing for Classical ML
    # We must replicate exactly how the model was trained
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (128, 128))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # Feature Extraction using HOG
    features = hog(img_gray, orientations=9, pixels_per_cell=(16, 16), 
                   cells_per_block=(2, 2))
    
    # 5. Prediction Logic
    if st.button('Analyze Road Surface'):
        # Reshape features for a single sample prediction
        prediction = model.predict(features.reshape(1, -1))
        confidence = model.predict_proba(features.reshape(1, -1))
        
        # Display Result
        result = categories[prediction[0]]
        st.success(f"**Prediction:** {result}")
        
        # Show confidence levels as a progress bar
        st.write("### Confidence Score")
        score = np.max(confidence)
        st.progress(float(score))
        st.write(f"The model is {score*100:.2f}% sure of this classification.")

st.sidebar.info("Note: This is the Baseline Model. Compare these results with the Deep Learning model in the final report.")
