######################
# Imported Libraries #
######################
import streamlit as st
import torch
from PIL import Image
import numpy as np
import os

##############
# Page Setup #
##############
st.set_page_config(
    page_title="The Neural Force Project",
    page_icon="🤖",
    layout="centered"
)

st.title("The Neural Force Project")
st.header("CSS181-2 Deep Learning Project")
st.subheader("")

#########################
# Load YOLOv8 Model     #
#########################
@st.cache_resource
def load_model():
    model_path = "./model/best.pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

##############################
# Upload + Predict Interface #
##############################
st.divider()
st.subheader("🧩 Test the Model")

uploaded_file = st.file_uploader("Upload a weld image (JPG or PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔮 Predict", key="predict_button"):
        with st.spinner("Running model prediction..."):
            try:
                # Convert image for YOLO
                results = model(image, size=640)
                pred_class = "No weld detected"

                if len(results.xyxy[0]) > 0:
                    cls_idx = int(results.xyxy[0][0][-1].item())
                    pred_class = model.names[cls_idx]

                st.success(f"✅ Predicted Class: **{pred_class.upper()}**")

                # Annotated results
                results.render()
                st.image(results.ims[0], caption="Detection Result", use_column_width=True)
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
else:
    st.warning("Please upload a weld image to begin.")
