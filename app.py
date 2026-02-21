######################
# Imported Libraries #
######################
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
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

pages = {
    "🔍 Predictions": "predictions",
    "📘 Project Description": "description",
    "📈 Results & Summary": "results"
}

if "current_page" not in st.session_state:
    st.session_state.current_page = "🔍 Predictions"

# --- Sidebar Buttons (Consistent Size) ---
for name, key in pages.items():
    if st.sidebar.button(name, use_container_width=True):
        st.session_state.current_page = name

st.markdown("""
    <style>
    div[data-testid="stSidebar"] button {
        width: 100% !important;
        height: 50px !important;
        font-size: 16px !important;
        margin-bottom: 5px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Home/Team Info Section (Sidebar Bottom) ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🧠 **The Neural Force Project**
**A Deep Learning Approach to Weld Quality Classification**

**Course:** CSS181-2  
**Section:** AM1  

**Members:**  
- Celles, Aaron Kent
- Leviste, Lee Ryan
- Lim, Kyle Hendrik L.
- Santeco, Enrique  
""")

###################################
# PAGE 1: Model Prediction (Home) #
###################################
page = st.session_state.current_page

if page == "🔍 Predictions":
    st.title("🔍 Weld Defect Detection")
    st.markdown("Upload an image to detect weld lines, spatters, porosity, and cracks using YOLOv12 segmentation.")
    st.divider()

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

    uploaded_file = st.file_uploader("📤 Upload a weld image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        if st.button("🔮 Predict"):
            with st.spinner("Running model prediction..."):
                try:
                    import cv2
                    import numpy as np
                    import tempfile

                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        image.save(tmp.name)
                        results = model(tmp.name)

                    res = results[0]

                    if res.masks is None or len(res.masks.data) == 0:
                        st.warning("⚠️ No weld features detected in this image.")
                        st.stop()

                    img_np = np.array(image)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    # Color map in BGR format for OpenCV
                    # Green for weld line, Cyan for spatters, Purple for porosity, Orange for cracks
                    color_map = {
                        "welding line": (0, 255, 0),     # Green
                        "weld line": (0, 255, 0),        # Green
                        "weldingline": (0, 255, 0),      # Green
                        "weld_line": (0, 255, 0),        # Green
                        "welding_line": (0, 255, 0),     # Green
                        "spatters": (255, 255, 0),       # Cyan
                        "spatter": (255, 255, 0),        # Cyan
                        "splatters": (255, 255, 0),      # Cyan (alternative spelling)
                        "splatter": (255, 255, 0),       # Cyan (alternative spelling)
                        "porosity": (128, 0, 128),       # Purple
                        "porosities": (128, 0, 128),     # Purple
                        "cracks": (0, 165, 255),         # Orange
                        "crack": (0, 165, 255)           # Orange
                    }

                    def draw_label_with_bg(img, text, pos, color):
                        x, y = pos
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.6
                        thickness = 2
                        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                        bg_color = (40, 40, 40)
                        cv2.rectangle(img, (x - 3, y - th - 6), (x + tw + 3, y + 3), bg_color, -1)
                        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

                    def smooth_mask(mask, kernel_size=5, epsilon_factor=0.001):
                        mask = (mask * 255).astype(np.uint8)
                        h, w = img_np.shape[:2]
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        mask_filtered = cv2.bilateralFilter(mask, 9, 75, 75)
                        mask_blurred = cv2.GaussianBlur(mask_filtered, (kernel_size, kernel_size), 0)
                        _, mask_thresh = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                        mask_closed = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                        mask_smooth = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)
                        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        smoothed_contours = []
                        for contour in contours:
                            if cv2.contourArea(contour) < 50:
                                continue
                            epsilon = epsilon_factor * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            smoothed_contours.append(approx)
                        return smoothed_contours

                    def normalize_label(label):
                        """Normalize label names for consistent counting and display"""
                        label_lower = label.lower().replace(" ", "").replace("_", "")
                        
                        if "weld" in label_lower or "line" in label_lower:
                            return "weld line"
                        elif "spat" in label_lower:  # Catches both spatters and splatters
                            return "spatters"
                        elif "poros" in label_lower:
                            return "porosity"
                        elif "crack" in label_lower:
                            return "cracks"
                        
                        return label.lower()

                    masks = res.masks.data.cpu().numpy()
                    class_ids = res.boxes.cls.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy()
                    
                    # Label counts
                    label_counts = {"weld line": 0, "spatters": 0, "porosity": 0, "cracks": 0}

                    for i, mask in enumerate(masks):
                        cls_id = class_ids[i]
                        label = res.names[cls_id].lower()
                        conf = confs[i]
                        
                        # Get normalized label for counting
                        normalized_label = normalize_label(label)
                        
                        # Get color from map (check various forms of the label)
                        label_clean = label.replace(" ", "").replace("_", "")
                        color = color_map.get(label, color_map.get(label_clean, (200, 200, 200)))
                        
                        smoothed_contours = smooth_mask(mask, kernel_size=9, epsilon_factor=0.003)
                        
                        if not smoothed_contours:
                            continue
                        
                        overlay = img_np.copy()
                        cv2.drawContours(overlay, smoothed_contours, -1, color, -1)
                        img_np = cv2.addWeighted(overlay, 0.4, img_np, 0.6, 0)
                        cv2.drawContours(img_np, smoothed_contours, -1, color, 3, cv2.LINE_AA)
                        
                        x, y, w, h = cv2.boundingRect(smoothed_contours[0])
                        text = f"{normalized_label.capitalize()} ({conf:.2f})"
                        draw_label_with_bg(img_np, text, (x, y - 5), color)
                        
                        if normalized_label in label_counts:
                            label_counts[normalized_label] += 1

                    annotated_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    st.image(annotated_image, caption="🧠 Segmentation Result", use_column_width=True)

                    total = sum(label_counts.values())
                    st.markdown("---")
                    st.subheader("📊 Detection Summary")
                    st.markdown(f"**Total Features Detected:** {total}")
                    
                    cols = st.columns(4)
                    cols[0].metric("🟢 Weld Lines", label_counts["weld line"])
                    cols[1].metric("🔵 Spatters", label_counts["spatters"])
                    cols[2].metric("🟣 Porosity", label_counts["porosity"])
                    cols[3].metric("🟠 Cracks", label_counts["cracks"])



                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.info("📎 Please upload an image to begin detection.")

#############################################
# PAGE 2: Project Description and Background #
#############################################
#############################################
# PAGE 2: Project Description and Background #
#############################################
elif page == "📘 Project Description":
    st.title("📘 Project Description")
    st.markdown("---")
    
    st.markdown("""
    ### **Abstract**
    This study employs a deep learning model, **YOLOv12**, to automatically detect and analyze critical weld 
    features from image data, specifically **welding line**, **porosity**, **spatters**, and **cracks**. Precision, 
    recall, and mean Average Precision (mAP), among other important detection and segmentation metrics, were used 
    to assess performance using a segmentation-based model. Due to their unique visual features, spatter and porosity 
    detections produced the highest mAP scores, demonstrating the model's strong overall accuracy. Crack detection, 
    on the other hand, performed worse, probably because of small training samples and subtle visual characteristics. 
    With bounding box prediction exhibiting marginally better accuracy than mask segmentation, the model showed 
    balanced detection and segmentation capabilities. Future work should concentrate on enhancing image quality to 
    guarantee better feature learning, masking, and detection reliability to increase model robustness.
    
    **Keywords:** Weld feature prediction, deep learning, image segmentation, object detection, computer vision, 
    defect classification, automated inspection, quality control, mAP.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### **I. Introduction**
    Ensuring the structural integrity of welded joints is a critical aspect of industrial manufacturing and construction, 
    as even minor flaws in welds can compromise safety, durability, and overall performance. Traditional manual inspection 
    methods are often time-consuming, subjective, and prone to human error, highlighting the need for automated solutions 
    that can accurately classify weld quality. 
    
    In this study, we focus on developing a deep learning–based approach using the **YOLOv12** object detection framework 
    to annotate and classify welds into four categories:
    
    1. **Welding line** - representing the seam continuity
    2. **Porosity** - indicating trapped gas voids or bubbles within the weld
    3. **Spatters** - referring to scattered molten droplets around the weld area
    4. **Cracks** - which are critical discontinuities that threaten structural integrity
    
    The implementation of automated weld classification has significant implications for productivity and cost-efficiency. 
    Automated systems can provide rapid and consistent evaluations across large volumes of welds, enabling real-time 
    monitoring and quality control within production pipelines. Furthermore, the ability of deep learning models to detect 
    subtle variations that may not be visible to the human eye enhances the reliability of inspections compared to 
    traditional methods.
    
    ---
    
    ### **Objectives**
    The objective of this project is threefold:
    
    1. Explore a different annotation technique that best covers welded metal and its surrounding area
    2. Build predictive models using YOLOv12 deep learning algorithm
    3. Analyze and interpret detection outputs to evaluate weld integrity
    
    The outcome of this study supports production optimization, promotes safer and more efficient welding practices, and 
    contributes to the growing body of research in intelligent inspection systems within industrial and manufacturing 
    contexts, thereby reducing the risks of structural failure and improving compliance with safety standards in 
    engineering applications.
    """)
    
    st.markdown("---")
    
    # Download IEEE Paper Button (after abstract)
    ieee_paper_path = "./CSS181-2_IEEE.pdf"
    if os.path.exists(ieee_paper_path):
        with open(ieee_paper_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        st.download_button(
            label="📄 Download Full IEEE Paper (PDF)",
            data=pdf_bytes,
            file_name="CSS181-2_IEEE.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.info("📎 IEEE Paper not found. Please place 'CSS181-2_IEEE.pdf' in the project root directory.")

###########################################
# PAGE 3: Results & Model Evaluation Charts
###########################################
elif page == "📈 Results & Summary":
    import pandas as pd
    import time

    st.title("📊 Model Evaluation Results")
    st.markdown("---")

    results_dir = "./results"
    results_path = os.path.join(results_dir, "results.csv")

    start = time.time()

    # --- Step 1: Try to load CSV safely ---
    if os.path.exists(results_path):
        try:
            st.info("📄 Loading CSV file...")
            df = pd.read_csv(results_path, engine="python", encoding_errors="ignore")

            st.success(f"✅ Loaded CSV successfully! ({len(df)} rows × {len(df.columns)} columns)")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download CSV Button
            with open(results_path, "rb") as csv_file:
                csv_bytes = csv_file.read()
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"❌ CSV loading failed: {e}")
    else:
        st.warning("⚠️ No results.csv found in the results folder!")

    st.markdown("---")

    st.success(f"✅ Finished loading everything in {time.time() - start:.2f} seconds")