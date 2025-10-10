######################
# Imported Libraries #
######################
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

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

#########################
# Load YOLOv8 Model     #
#########################
@st.cache_resource
def load_model():
    model_path = "./model/best.pt"
    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found! Please place `best.pt` inside the `/model` folder.")
        st.stop()
    return YOLO(model_path)

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

uploaded_file = st.file_uploader("Upload a weld image:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔮 Predict", key="predict_button"):
        with st.spinner("Running model prediction..."):
            try:
                # Save uploaded image temporarily
                file_ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                    image.save(tmp.name)
                    results = model(tmp.name)

                # Extract prediction
                boxes = results[0].boxes
                pred_class = (
                    results[0].names[int(boxes.cls[0])]
                    if len(boxes) > 0
                    else "No weld detected"
                )

                st.success(f"✅ Predicted Class: **{pred_class.upper()}**")

                # Show annotated result
                annotated = results[0].plot()
                st.image(annotated, caption="Detection Result", use_column_width=True)

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
else:
    st.info("Please upload an image to begin.")
