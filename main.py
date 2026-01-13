import streamlit as st
import cv2
import numpy as np
import tempfile
import time

# ---------------- CONFIG ----------------
CONFIDENCE_THRESHOLD = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromCaffe(
        "MobileNetSSD_deploy.prototxt.txt",
        "MobileNetSSD_deploy.caffemodel"
    )
    return net

net = load_model()

# ---------------- DETECTION FUNCTION ----------------
def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), COLORS[idx], 2)
            y = startY - 10 if startY - 10 > 10 else startY + 20
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return frame

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Object Detection App", layout="wide")
st.title("🧠 Object Detection (Image • Video • Webcam)")
st.write("Powered by MobileNet SSD + OpenCV DNN")

mode = st.radio("Choose Mode", ["Image", "Video", "Webcam"])

# ---------------- IMAGE ----------------
if mode == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR", caption="Original Image")
        result = detect_objects(image)
        st.image(result, channels="BGR", caption="Detected Objects")

# ---------------- VIDEO ----------------
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect_objects(frame)
            stframe.image(frame, channels="BGR")

        cap.release()

# ---------------- WEBCAM ----------------
else:
    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")
    stframe = st.empty()

    if start:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break

            frame = detect_objects(frame)
            stframe.image(frame, channels="BGR")

        cap.release()
