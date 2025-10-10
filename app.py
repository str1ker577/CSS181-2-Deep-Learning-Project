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
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    ("🏠 Home", "📘 Project Description", "🔍 Predictions")
)

########################################
# PAGE 1: Home - Intro and Team Details #
########################################
if page == "🏠 Home":
    st.title("The Neural Force Project")
    st.subheader("A Deep Learning Approach to Weld Quality Classification")
    st.markdown("---")

    st.write("""
    **Course:** CSS181-2 — Deep Learning  
    **Institution:** Mapúa Malayan Colleges Mindanao  
    **Term:** A.Y. 2025  

    **Group Name:** The Neural Force  
    **Section:** CS181-AI2  
    **Members:**  
    - Kyle Lim  
    - [Add your other members here]
    """)

    st.markdown("---")
    st.info("Use the sidebar to explore the project description and try live predictions.")

#############################################
# PAGE 2: Project Description and Background #
#############################################
elif page == "📘 Project Description":
    st.title("Project Description")
    st.markdown("---")

    st.markdown("""
    ### **Abstract**
    Using image data, this study utilized a deep learning model (YOLOv12) to predict and classify weld quality, 
    classifying welds as **good**, **bad**, or **defect**. Precision, recall, and mean Average Precision (mAP), among 
    other important detection and segmentation metrics, were used to assess performance using a segmentation-based model.  
    Because there were fewer training samples and less pronounced visual patterns in the defect class, the model performed 
    poorly in identifying both good and bad welds, according to the results. With a marginally higher accuracy in bounding 
    box prediction than in mask segmentation, the model showed balanced detection and segmentation performance.  

    **Future work** should concentrate on growing the dataset, especially for defect classes, and enhancing image quality 
    to guarantee better feature learning and detection reliability to increase model robustness.  

    **Keywords:** Weld quality prediction, deep learning, image segmentation, object detection, computer vision, defect classification, automated inspection, quality control, mAP.
    """)

    st.markdown("---")
    st.markdown("""
    ### **I. Introduction**
    Ensuring the structural integrity of welded joints is a critical aspect of industrial manufacturing and construction, 
    as even minor flaws in welds can compromise safety, durability, and overall performance. Traditional manual inspection 
    methods are often time-consuming, subjective, and prone to human error, highlighting the need for automated solutions 
    that can accurately classify weld quality.

    In this study, we focus on developing a deep learning–based approach using the **YOLOv12 object detection framework** 
    to annotate and classify welds into three categories:
    1. **Good** — smooth, consistent weld seams without spatters  
    2. **Bad** — irregular or wavy welds with excess spatters  
    3. **Defect** — welds with critical cracks at the seam center  

    By automating weld assessment, this research aims to provide a reliable method for verifying weld secureness, thereby 
    reducing risks of structural failures and ensuring compliance with safety standards in engineering applications.

    The implementation of automated weld classification has significant implications for productivity and cost-efficiency. 
    Automated systems can provide rapid and consistent evaluations across large volumes of welds, enabling real-time monitoring 
    and quality control within production pipelines.

    The **objective** of this project is threefold:
    1. Explore annotation techniques that best cover welded metal and its surrounding area  
    2. Build predictive models using the **YOLOv12 deep learning algorithm**  
    3. Identify the weld status of submitted images  

    The outcome supports production optimization and promotes safer and more efficient practices in welding-related industries.
    """)

    st.markdown("---")
    st.markdown("""
    ### **II. Related Work**
    The YOLOv10n deep learning model was employed in the study of **Cengil [1]** to automatically identify weld flaws. 
    The model used bounding boxes to mark the locations of welds and classified them as good, bad, or defective using the 
    Kaggle Welding Defect dataset. The model achieved **0.939 precision** and **0.91 recall**, demonstrating its strong performance.  

    Similarly, **Truong et al. [2]** confirmed the effectiveness of YOLOv10 for bounding box-based detection of weld defects, 
    evaluating performance via Intersection over Union (IoU). Results showed YOLOv10 provided a fast and accurate method 
    for weld flaw identification.
    """)

###################################
# PAGE 3: Model Prediction (YOLO) #
###################################
elif page == "🔍 Predictions":
    st.title("Weld Quality Prediction")
    st.markdown("Upload an image to test the trained YOLOv12 model.")
    st.divider()

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

    #########################
    # Upload + Predict UI   #
    #########################
    uploaded_file = st.file_uploader("Upload a weld image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔮 Predict", key="predict_button"):
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
