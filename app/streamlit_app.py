import sys
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure the src folder is accessible for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.m4_live_inference import FaceRecognitionPipeline

# Initialize backend pipeline once and cache it
@st.cache_resource
def load_pipeline():
    return FaceRecognitionPipeline()

pipeline = load_pipeline()

# Persistent state for our reference database
if 'reference_db' not in st.session_state:
    st.session_state['reference_db'] = {}

st.set_page_config(page_title="Open-Set Face Recognition", layout="wide")
st.title("Live Face Recognition System")

# Sidebar: Register Known Faces
with st.sidebar:
    st.header("1. Register a Person")
    st.write("Upload a clear photo of a face to add them to the system's memory.")
    
    register_name = st.text_input("Person's Name")
    register_file = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"], key="reg")
    
    if st.button("Add to Database"):
        if register_name and register_file:
            img = np.array(Image.open(register_file).convert('RGB'))
            faces = pipeline.detect_faces(img)
            
            if len(faces) == 1:
                emb = pipeline.get_embedding(faces[0]["crop"])
                st.session_state['reference_db'][register_name] = emb
                st.success(f"Added {register_name} successfully!")
            elif len(faces) == 0:
                st.error("No faces found in this image.")
            else:
                st.error("Multiple faces found! Please upload an image with only one person.")
        else:
            st.warning("Please provide both a name and an image.")
            
    st.divider()
    st.subheader("Registered Profiles")
    for name in st.session_state['reference_db'].keys():
        st.write(f"✅ {name}")

# Main Window: Inference
st.header("2. Real-Time Detection")
st.write("Test the model against your registered profiles. Unrecognized faces will be flagged as 'Non-Defined'.")

tab1, tab2 = st.tabs(["Use Webcam", "Upload Test Image"])

with tab1:
    st.write("Click 'Start' to capture a frame from your webcam.")
    camera_image = st.camera_input("Live Camera feed")
    
    if camera_image is not None:
        img = np.array(Image.open(camera_image).convert('RGB'))
        processed_img = pipeline.process_frame(img, st.session_state['reference_db'])
        st.image(processed_img, caption="Inference Result", use_column_width=True)

with tab2:
    test_file = st.file_uploader("Upload an image to test", type=["jpg", "jpeg", "png"], key="test")
    
    if test_file is not None:
        img = np.array(Image.open(test_file).convert('RGB'))
        processed_img = pipeline.process_frame(img, st.session_state['reference_db'])
        st.image(processed_img, caption="Inference Result", use_column_width=True)