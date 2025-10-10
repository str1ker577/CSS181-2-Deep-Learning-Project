######################
# Imported Libraries #
######################

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

##############
# Page Setup #
##############

st.set_page_config(
    page_title="The Neural Force Project",
    page_icon="🤖",
    layout="centered"
)

# Page Header
st.title("The Neural Force Project")
st.header("CSS181-2 Deep Learning Project")
st.subheader("")

#########################
# Load YOLOv8 Model     #
#########################

@st.cache_resource
def load_model():
    model_path = "./model/best.pt"
    model = YOLO(model_path)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

##############################
# Upload + Predict Interface #
##############################

st.divider()
st.subheader("🧩 Test the Model")

# Let user upload a file
uploaded_file = st.file_uploader(
    "Upload a weld image (JPG or PNG):",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict Button
    if st.button("🔮 Predict", key="predict_button"):
        with st.spinner("Running model prediction..."):
            try:
                # Run YOLO prediction directly on the image
                results = model(image)

                # Get predictions
                boxes = results[0].boxes
                if len(boxes) > 0:
                    # Get the first predicted class
                    pred_class = results[0].names[int(boxes.cls[0])]
                else:
                    pred_class = "No weld detected"

                # Display results
                st.success(f"✅ Predicted Class: **{pred_class.upper()}**")

                # Display annotated result
                annotated = results[0].plot()  # numpy array
                st.image(annotated, caption="Detection Result", use_column_width=True)

            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")

else:
    st.warning("Please upload a weld image to begin.")
