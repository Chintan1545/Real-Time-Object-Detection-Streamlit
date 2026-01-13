# Real-Time-Object-Detection-Streamlit (Image • Video • Webcam)

A complete Object Detection system built using MobileNet-SSD + OpenCV DNN, with an interactive Streamlit web application that supports image upload, video upload, and real-time webcam detection.

This project demonstrates an end-to-end computer vision pipeline, optimized for real-time performance on CPU.

## ✨ Features
- ✅ Object detection on images
- ✅ Object detection on video files
- ✅ Real-time webcam detection
- ✅ Bounding boxes with class labels & confidence scores
- ✅ Lightweight & fast MobileNet-SSD model
- ✅ Interactive Streamlit UI
- ✅ CPU-friendly (no GPU required)

## 🗂 Project Structure
```bash
Object-Detection-App/
│
├── screenshots/
│   ├── image_detection.png
│   ├── video_detection.png
│   └── webcam_detection.png
│
├── MobileNetSSD_deploy.prototxt.txt
├── MobileNetSSD_deploy.caffemodel
│
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/object-detection-app.git
cd object-detection-app
```
## 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
## requirements.txt
```bash
streamlit
opencv-python
numpy
imutils
```
## ▶️ Run the Application
```bash
streamlit run app.py
```

## 🧠 How It Works (Pipeline)

1. Input is taken from image / video / webcam
2. Frame is converted into a blob for normalization
3. Blob is passed through MobileNet-SSD
4. Model outputs:
  - Bounding boxes
  - Class IDs
  - Confidence scores
5. Weak detections are filtered
6. Bounding boxes and labels are drawn
7. Output is displayed in real time using Streamlit

## 🚀 Optimizations Used

- Frame resizing before inference
- Confidence threshold filtering
- Streamlit model caching (@st.cache_resource)
- Lightweight SSD-based architecture for real-time performance

## 🧪 Technologies Used

- Python
- OpenCV (DNN Module)
- MobileNet-SSD
- Streamlit
- NumPy
- Computer Vision

## ⭐ If you like this project

Give it a star ⭐ and feel free to fork!
