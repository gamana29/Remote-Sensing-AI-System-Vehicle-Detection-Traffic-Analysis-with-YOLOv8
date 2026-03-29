import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np
import cv2
import time
from collections import Counter

# ------------------ UI ------------------
st.title("🛰️ Remote Sensing AI System")
st.markdown("### Detection, Tracking, Speed & Size Analysis")

st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3)

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "png", "jpeg", "mp4"]
)

vehicle_classes = ["car", "truck", "bus"]

# ------------------ MAIN ------------------
if uploaded_file is not None:

    file_type = uploaded_file.type

    # ================= IMAGE =================
    if "image" in file_type:

        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            path = tmp.name

        results = model(path)
        result_img = results[0].plot()

        st.image(result_img, caption="Detection Result", use_container_width=True)

        boxes = results[0].boxes
        names = model.names

        count = 0

        st.subheader("📊 Vehicle Info")

        for box in boxes:
            cls = int(box.cls.item())
            label = names[cls]
            conf = box.conf.item()

            if label in vehicle_classes and conf > confidence:
                count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1

                st.write(f"{label} | Size: {width}x{height}px | Conf: {conf:.2f}")

        st.success(f"🚗 Total Vehicles: {count}")

    # ================= VIDEO =================
    elif "video" in file_type:

        st.video(uploaded_file)

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        prev_positions = {}
        vehicle_ids = set()
        frame_display = st.empty()

        st.write("Processing video... ⏳")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            boxes = results[0].boxes
            names = model.names

            current_positions = {}

            for i, box in enumerate(boxes):
                cls = int(box.cls.item())
                label = names[cls]

                if label in vehicle_classes:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    width = x2 - x1
                    height = y2 - y1

                    current_positions[i] = (cx, cy)
                    vehicle_ids.add(i)

                    # -------- SPEED --------
                    if i in prev_positions:
                        px, py = prev_positions[i]
                        distance = ((cx - px)**2 + (cy - py)**2) ** 0.5
                        speed = distance * 0.1
                    else:
                        speed = 0

                    # -------- DRAW --------
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    cv2.putText(frame,
                                f"{label} | S:{speed:.1f} | {width}x{height}",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,255,0),
                                2)

            prev_positions = current_positions

            frame_display.image(frame, channels="BGR")

        cap.release()

        st.subheader("📊 Final Stats")
        st.write(f"🚗 Total Vehicles Detected: {len(vehicle_ids)}")
        st.warning("⚠️ Speed & size are approximate (pixel-based)")
