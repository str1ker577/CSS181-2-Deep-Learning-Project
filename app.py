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

############################
# Sidebar Navigation Setup #
############################
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📘 Project Description", "🔮 Predictions"]
)

################
# Page 1: Home #
################
if page == "🏠 Home":
    st.title("🤖 The Neural Force Project")
    st.subheader("CSS181-2: Deep Learning Project")

    st.markdown("""
    ### Welcome to *The Neural Force Project*
    This project focuses on using **YOLOv8** (You Only Look Once) for detecting and classifying welding defects
    into three main categories:
    - ✅ Good
    - ⚠️ Defective
    - ❌ Bad

    Our model aims to support **automated weld quality inspection**, helping minimize human error in manufacturing environments.
    """)

    st.divider()
    st.subheader("📘 Class Details")
    st.markdown("""
    **Course:** CSS181-2 — Deep Learning  
    **Section:** AM1
    """)

    st.divider()
    st.subheader("👥 Group Members")
    st.markdown("""
    - Celles, Aaron
    - Leviste, Lee
    - Lim, Kyle
    - Santeco, Enrique
    """)

    st.info("Use the sidebar to navigate to the Project Description or the Prediction page.")


############################
# Page 2: Project Description #
############################
elif page == "📘 Project Description":
    st.title("📘 Project Description")
    st.markdown("""
    ### Overview  
    The **Neural Force Project** is a deep learning-based inspection system designed to analyze welding images
    and classify them based on defect severity using YOLOv8 object detection.

    ### Objectives  
    1. Develop an AI model capable of identifying welding defects.  
    2. Improve industrial efficiency by automating defect detection.  
    3. Provide a user-friendly interface for testing model predictions.  

    ### Dataset  
    - **Source:** Roboflow (Welding Project Dataset)  
    - **Classes:** Bad, Defect, Good  
    - **License:** CC BY 4.0  
    - [View Dataset on Roboflow](https://universe.roboflow.com/welding-vnkkh/welding-project-iaruw/dataset/1)

    ### Model  
    The model was trained using **YOLOv8**, a state-of-the-art object detection framework by Ultralytics.
    It uses convolutional layers and attention mechanisms to detect features relevant to weld quality.

    ### Expected Output  
    The system classifies uploaded images as one of the following:
    - ✅ Good weld
    - ⚠️ Defective weld
    - ❌ Bad weld
    """)


############################
# Page 3: Predictions       #
############################
elif page == "🔮 Predictions":
    st.title("🔮 Test the Model")

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

    st.divider()
    uploaded_file = st.file_uploader("Upload a weld image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict", key="predict_button"):
            with st.spinner("Running model prediction..."):
                try:
                    file_ext = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                        image.save(tmp.name)
                        results = model(tmp.name)

                    boxes = results[0].boxes
                    pred_class = (
                        results[0].names[int(boxes.cls[0])]
                        if len(boxes) > 0
                        else "No weld detected"
                    )

                    st.success(f"✅ Predicted Class: **{pred_class.upper()}**")

                    annotated = results[0].plot()
                    st.image(annotated, caption="Detection Result", use_column_width=True)

                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.info("Please upload an image to begin.")
