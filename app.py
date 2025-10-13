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

###################
# Sidebar Layout  #
###################
st.sidebar.title("🔧 Navigation")

# Use clickable buttons instead of selectbox/radio
pages = {
    "🏠 Home": "home",
    "📘 Project Description": "description",
    "🔍 Predictions": "predictions"
}

# Track which page user clicks on
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home"

for name, key in pages.items():
    if st.sidebar.button(name):
        st.session_state.current_page = name

page = st.session_state.current_page

########################################
# PAGE 1: Home - Intro and Team Details #
########################################
if page == "🏠 Home":
    st.title("The Neural Force Project")
    st.subheader("A Deep Learning Approach to Weld Quality Classification")
    st.markdown("---")

    st.write("""
    **Course:** CSS181-2  
    **Section:** AM1  

    **Group Name:** The Neural Force  
    **Members:**  
    - Celles, Aaron  
    - Leviste, Lee  
    - Lim, Kyle  
    - Santeco, Enrique  
    """)

    st.markdown("---")
    st.info("Use the sidebar to explore the project description or test predictions.")

#############################################
# PAGE 2: Project Description and Background #
#############################################
elif page == "📘 Project Description":
    st.title("📘 Project Description")
    st.markdown("---")

    st.markdown("""
    ### **Abstract**
    Using image data, this study utilized a deep learning model (**YOLOv12**) to predict and classify weld quality, 
    categorizing welds as **good**, **bad**, or **defect**. Evaluation metrics such as **precision**, **recall**, and 
    **mean Average Precision (mAP)** were used to assess model performance through segmentation-based detection.  

    Results indicated that due to fewer training samples and less distinct features in the *defect* class, 
    the model had lower accuracy identifying certain classes. Nonetheless, it demonstrated balanced detection 
    and segmentation performance, with slightly higher accuracy in bounding box prediction.  

    **Future work** should focus on expanding the dataset—especially for defect classes—and improving image quality 
    for enhanced feature learning and model robustness.  

    **Keywords:** Weld quality prediction, deep learning, image segmentation, object detection, computer vision, defect classification, automated inspection, quality control, mAP.
    """)

    st.markdown("---")
    st.markdown("""
    ### **I. Introduction**
    Ensuring the structural integrity of welded joints is vital in manufacturing and construction. Even minor flaws 
    can compromise safety and durability. Manual inspections are time-consuming and prone to human error, underscoring 
    the need for automated systems that can classify weld quality with precision.

    This project applies the **YOLOv12 object detection framework** to annotate and classify welds into:
    1. **Good** — smooth and consistent seams without spatters  
    2. **Bad** — irregular seams with excess spatters  
    3. **Defect** — welds that appear good but have critical cracks at the seam  

    Automating weld assessment enhances reliability, supports production optimization, and promotes safer engineering practices.
    """)

    st.markdown("---")
    st.markdown("""
    ### **II. Related Work**
    **Cengil [1]** used the YOLOv10n model to detect weld flaws on the Kaggle *Welding Defect – Object Detection* dataset.  
    The model achieved **0.939 precision** and **0.91 recall**, demonstrating its strong detection capability.  

    **Truong et al. [2]** also utilized YOLOv10 for weld flaw detection, comparing predicted and actual bounding boxes 
    using Intersection over Union (IoU), showing YOLOv10’s efficiency for fast and accurate weld detection.
    """)

###################################
# PAGE 3: Model Prediction (YOLO) #
###################################
elif page == "🔍 Predictions":
    st.title("🔍 Weld Quality Prediction")
    st.markdown("Upload an image to test the trained YOLOv12 model.")
    st.divider()

    #########################
    # Load YOLOv12 Model    #
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

    #########################
    # Upload + Predict UI   #
    #########################
    uploaded_file = st.file_uploader("📤 Upload a weld image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        if st.button("🔮 Predict"):
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

                    st.success(f"✅ **Predicted Class:** {pred_class.upper()}")

                    annotated = results[0].plot()
                    st.image(annotated, caption="🧠 Detection Result", use_column_width=True)

                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.info("📎 Please upload an image to begin prediction.")
