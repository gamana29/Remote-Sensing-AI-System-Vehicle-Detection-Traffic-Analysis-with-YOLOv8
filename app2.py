import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np

# ------------------ UI ------------------
st.title("🛰️ Remote Sensing AI System")
st.markdown("### Detection, Tracking, Speed & Size Analysis")

st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3)
frame_skip = st.sidebar.slider("Process Every Nth Frame", 1, 5, 2)  # skip frames

# Load YOLOv8 on GPU if available
model = YOLO("yolov8n.pt")  # add device="0" if GPU

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

        count = 0
        st.subheader("📊 Vehicle Info")
        for box in results[0].boxes:
            cls = int(box.cls.item())
            label = model.names[cls]
            conf = box.conf.item()
            if label in vehicle_classes and conf > confidence:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                st.write(f"{label} | Size: {x2-x1}x{y2-y1}px | Conf: {conf:.2f}")

        st.success(f"🚗 Total Vehicles: {count}")

    # ================= VIDEO =================
    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        prev_positions = {}
        vehicle_ids = set()
        frame_id = 0

        st.write("Processing video... ⏳")
        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            # Process every Nth frame
            if frame_id % frame_skip != 0:
                out.write(frame)
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

                    # Speed estimation
                    if i in prev_positions:
                        px, py = prev_positions[i]
                        distance = np.linalg.norm([cx - px, cy - py])
                        speed = distance * 0.1  # pixel-to-speed factor
                    else:
                        speed = 0

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame,
                                f"{label} | S:{speed:.1f} | {x2-x1}x{y2-y1}",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,255,0),
                                2)
            prev_positions = current_positions
            out.write(frame)
            progress_bar.progress(min(frame_id/total_frames,1.0))

        cap.release()
        out.release()

        st.video(output_file)
        st.subheader("📊 Final Stats")
        st.write(f"🚗 Total Vehicles Detected: {len(vehicle_ids)}")
        st.warning("⚠️ Speed & size are approximate (pixel-based)")
