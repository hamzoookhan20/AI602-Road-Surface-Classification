import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
import torch 
from torchvision import transforms
import os
import gdown  # CRITICAL: Added for Google Drive downloads

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI602: Road Surface Classifier", page_icon="🛣️")
st.title("🛣️ Autonomous Vehicle Perception")
st.markdown("**Student:** Muhammad Hamza Khan")

# --- 2. MODEL LOADING FUNCTIONS ---

@st.cache_resource
def load_classical_model():
    """Loads the Random Forest model from the local directory."""
    try:
        return joblib.load('rtk_classical_model.pkl')
    except Exception as e:
        st.error(f"Classical model not found: {e}")
        return None

@st.cache_resource
def load_vit_model():
    """Downloads and loads the Vision Transformer from Google Drive."""
    file_id = '1a7UtWhQ0nMZEGhRrHqFcdpYIyuzkS2vU'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'rtk_vit_model.pth'

    if not os.path.exists(output):
        try:
            with st.status("📥 Downloading ViT Model (330MB)..."):
                gdown.download(url, output, quiet=False, fuzzy=True)
                st.write("✅ Download Complete!")
