# 🛰️ Remote Sensing AI System  
### Vehicle Detection, Speed & Size Estimation using YOLOv8

## 👩‍💻 Author
**Chirumamilla Gamana**  
Roll No: 2320040075  

---

## 📌 Project Overview

This project presents a **Remote Sensing AI System** that performs **vehicle detection, tracking, speed estimation, and size analysis** using images and videos.

It leverages **YOLOv8 (You Only Look Once)** for real-time object detection and provides an interactive dashboard using **Streamlit**.

---

## 🎯 Objectives

- Detect vehicles (car, truck, bus) using AI  
- Count and classify vehicles  
- Estimate **vehicle speed** from video  
- Estimate **vehicle size** using pixel-to-meter conversion  
- Provide a **real-time dashboard visualization**

---

## ⚙️ Technologies Used

- Python  
- Streamlit  
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  
- Pandas  

---

## 🧠 How It Works

### 📷 Image Processing
1. Upload an image  
2. YOLO detects vehicles  
3. Bounding boxes are drawn  
4. System calculates:
   - Total vehicles  
   - Vehicle type distribution  
   - Average vehicle size  

---

### 🎥 Video Processing
1. Video is split into frames  
2. YOLO runs on each frame  
3. Vehicles are detected and tracked  
4. System calculates:
   - Vehicle count  
   - Speed (based on movement between frames)  
   - Size (based on bounding box area)  

---

## 📊 Features

- Real-time vehicle detection  
- Vehicle counting & classification  
- Speed estimation (km/h)  
- Size estimation (m²)  
- Interactive dashboard  
- Vehicle data table  

---

## 🎛️ User Controls

- **Confidence Threshold** → Controls detection accuracy  
- **Frame Skip** → Improves performance for videos  
- **YOLO Model Selection** → Speed vs accuracy  
- **Meters per Pixel** → Converts pixel values to real-world units  

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/gamana29/Remote-Sensing-AI-System-Vehicle-Detection-Traffic-Analysis-with-YOLOv8.git
cd Remote-Sensing-AI-System-Vehicle-Detection-Traffic-Analysis-with-YOLOv8
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Application
```bash
streamlit run app.py
```

### 📁 Project Structure
```bash
├── app.py
├── requirements.txt
├── sample_images/
├── README.md
📈 Performance
```
---
Real-time detection using YOLOv8

Processing time in milliseconds per frame

Works efficiently on both CPU and GPU
---
### ⚠️ Limitations
Speed estimation is approximate

Depends on camera angle and calibration

Basic tracking (ID may change across frames)

Real-world accuracy may vary
---
### 🔮 Future Improvements
Advanced tracking (DeepSORT)

Lane detection

Real-time alerts

Cloud deployment

Integration with smart traffic systems
---
### 🏙 Applications
Traffic monitoring

Smart city systems

Parking analysis

Urban planning

Intelligent transportation
---
### 📌 Conclusion
This project demonstrates how AI and computer vision can be used to transform visual data into meaningful traffic insights, enabling smarter and more efficient urban systems.
---
