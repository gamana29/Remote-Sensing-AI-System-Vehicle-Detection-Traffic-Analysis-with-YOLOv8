import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np
import cv2
import time
from collections import Counter

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="🛰️ Remote Sensing AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3)
frame_skip = st.sidebar.slider("Process Every Nth Frame (Video)", 1, 5, 2)
model_choice = st.sidebar.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
use_gpu = st.sidebar.checkbox("Use GPU if available", value=True)
real_scale_m_per_pixel = st.sidebar.number_input("Meters per Pixel (for speed/size estimation)", value=0.05)

vehicle_classes = ["car", "truck", "bus"]

# ------------------ LOAD MODEL ------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name, device):
    return YOLO(model_name)  # device handled automatically by ultralytics latest

device = "cuda:0" if use_gpu else "cpu"
model = load_model(model_choice, device)

# ------------------ MAIN UI ------------------
st.title("🛰️ Remote Sensing AI System")
st.markdown("### Vehicle Detection, Speed & Size Estimation with YOLOv8")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])

# ------------------ IMAGE PROCESSING ------------------
if uploaded_file is not None and "image" in uploaded_file.type:
    st.subheader("📷 Image Processing")

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("⚠️ Cannot read this image.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)

        # Temp file for YOLO
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            cv2.imwrite(tmp_file.name, img)
            temp_path = tmp_file.name

        start_time = time.time()
        results = model(temp_path)
        result_img = results[0].plot()
        end_time = time.time()

        with col2:
            st.image(result_img, caption="Detection Result", use_container_width=True)

        # Vehicle analysis + size estimation
        names = model.names
        boxes = results[0].boxes
        vehicle_counts = Counter()
        total_vehicles = 0
        sizes_m2 = []

        for box in boxes:
            cls = int(box.cls.item())
            label = names[cls]
            conf = box.conf.item()
            if label in vehicle_classes and conf > confidence:
                vehicle_counts[label] += 1
                total_vehicles += 1

                # Approximate vehicle size (area in m²)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width_m = (x2 - x1) * real_scale_m_per_pixel
                height_m = (y2 - y1) * real_scale_m_per_pixel
                sizes_m2.append(width_m * height_m)

        st.markdown("### 📊 Vehicle Counts & Sizes")
        st.metric("Total Vehicles Detected", total_vehicles)
        st.bar_chart(vehicle_counts)
        if sizes_m2:
            st.metric("Average Vehicle Size (m²)", round(np.mean(sizes_m2), 2))

        st.info(f"⏱ Processing Time: {end_time - start_time:.2f}s")

# ------------------ VIDEO PROCESSING ------------------
elif uploaded_file is not None and "video" in uploaded_file.type:
    st.subheader("🎥 Video Processing & Tracking")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("⚠️ Cannot read this video.")
    else:
        st.video(uploaded_file)

        prev_positions = {}
        frame_id = 0
        vehicle_ids = set()
        frame_display = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        vehicle_counts_total = Counter()
        speed_list = []
        size_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % frame_skip != 0:
                continue

            results = model(frame)
            boxes = results[0].boxes
            current_positions = {}

            for i, box in enumerate(boxes):
                cls = int(box.cls.item())
                label = model.names[cls]
                conf = box.conf.item()
                if label in vehicle_classes and conf > confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2)//2, (y1 + y2)//2
                    current_positions[i] = (cx, cy)
                    vehicle_ids.add(i)
                    vehicle_counts_total[label] += 1

                    # Size estimation
                    width_m = (x2 - x1) * real_scale_m_per_pixel
                    height_m = (y2 - y1) * real_scale_m_per_pixel
                    size_list.append(width_m * height_m)

                    # Speed estimation
                    if i in prev_positions:
                        px, py = prev_positions[i]
                        distance_m = np.linalg.norm([cx - px, cy - py]) * real_scale_m_per_pixel
                        speed_m_s = distance_m * 30 / frame_skip  # assuming 30 FPS
                        speed_list.append(speed_m_s)
                        cv2.putText(frame, f"S:{speed_m_s:.1f} m/s", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            prev_positions = current_positions
            frame_display.image(frame, channels="BGR")
            progress_bar.progress(min(frame_id / total_frames, 1.0))

        cap.release()
        st.success(f"🚗 Total Vehicles Detected: {len(vehicle_ids)}")
        st.bar_chart(vehicle_counts_total)
        if size_list:
            st.metric("Average Vehicle Size (m²)", round(np.mean(size_list), 2))
        if speed_list:
            st.metric("Average Speed (m/s)", round(np.mean(speed_list), 2))

st.markdown("---")
st.subheader("🏙 Smart City Applications")
st.write("""
- Traffic Monitoring  
- Urban Planning  
- Parking Analysis  
- Intelligent Transportation Systems  
""")

