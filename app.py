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
    st.title("🔍 Weld Quality Prediction")
    st.markdown("Upload an image to test the trained YOLOv12 segmentation model.")
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
                        st.warning("⚠️ No welds detected in this image.")
                        st.stop()

                    img_np = np.array(image)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    color_map = {
                        "good": (0, 255, 0),
                        "bad": (0, 0, 255),
                        "defect": (0, 165, 255)
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

                    masks = res.masks.data.cpu().numpy()
                    class_ids = res.boxes.cls.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy()
                    label_counts = {"good": 0, "bad": 0, "defect": 0}

                    for i, mask in enumerate(masks):
                        cls_id = class_ids[i]
                        label = res.names[cls_id].lower()
                        conf = confs[i]
                        color = color_map.get(label, (255, 255, 255))
                        smoothed_contours = smooth_mask(mask, kernel_size=9, epsilon_factor=0.003)
                        if not smoothed_contours:
                            continue
                        overlay = img_np.copy()
                        cv2.drawContours(overlay, smoothed_contours, -1, color, -1)
                        img_np = cv2.addWeighted(overlay, 0.4, img_np, 0.6, 0)
                        cv2.drawContours(img_np, smoothed_contours, -1, color, 3, cv2.LINE_AA)
                        x, y, w, h = cv2.boundingRect(smoothed_contours[0])
                        text = f"{label.capitalize()} ({conf:.2f})"
                        draw_label_with_bg(img_np, text, (x, y - 5), color)
                        if label in label_counts:
                            label_counts[label] += 1

                    annotated_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    st.image(annotated_image, caption="🧠 Segmentation Result", use_column_width=True)

                    total = sum(label_counts.values())
                    st.markdown("---")
                    st.subheader("📊 Prediction Summary")
                    st.markdown(f"**Total Welds Detected:** {total}")
                    cols = st.columns(3)
                    cols[0].metric("🟢 Good", label_counts["good"])
                    cols[1].metric("🔴 Bad", label_counts["bad"])
                    cols[2].metric("🟠 Defect", label_counts["defect"])

                    if total > 0:
                        st.markdown("### 🧾 Percentages")
                        st.write(f"- 🟢 Good: {label_counts['good'] / total * 100:.1f}%")
                        st.write(f"- 🔴 Bad: {label_counts['bad'] / total * 100:.1f}%")
                        st.write(f"- 🟠 Defect: {label_counts['defect'] / total * 100:.1f}%")

                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.info("📎 Please upload an image to begin prediction.")

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
    """)

############################
# PAGE 3: Results & Charts #
############################
elif page == "📊 Results & Summary":
    import pandas as pd
    st.title("📊 Model Evaluation Results")
    st.markdown("---")

    results_path = "./results/results.csv"

    @st.cache_data
    def load_results(path):
        """Load small results.csv once."""
        return pd.read_csv(path)

    if os.path.exists(results_path):
        df = load_results(results_path)

        st.subheader("📋 Model Performance Summary")
        st.dataframe(df, use_container_width=True)
        st.caption(f"Showing {len(df)} total results from model predictions.")

        # Optional summary stats
        if {"good", "bad", "defective"}.issubset(df.columns):
            st.markdown("### 🧾 Summary Statistics")
            summary = df[["good", "bad", "defective"]].sum()
            st.write(summary)
    else:
        st.warning("⚠️ No results.csv file found in the `results/` folder.")

    st.markdown("---")

    # --- Display charts ---
    st.subheader("📈 Visual Charts")
    results_dir = "./results"

    chart_files = [f for f in os.listdir(results_dir) if f.endswith((".png", ".jpg"))]

    if chart_files:
        cols = st.columns(2)
        for i, chart in enumerate(chart_files):
            img_path = os.path.join(results_dir, chart)
            with cols[i % 2]:
                st.image(img_path, caption=f"📊 {chart}", use_container_width=True)
    else:
        st.info("📎 No charts found in the `results/` folder.")

