import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# ------------------ CONFIG ------------------
YOLO_DIR = "yolo-coco"
CONFIDENCE = 0.5
THRESHOLD = 0.3

labelsPath = os.path.join(YOLO_DIR, "coco.names")
weightsPath = os.path.join(YOLO_DIR, "yolov3.weights")
configPath = os.path.join(YOLO_DIR, "yolov3.cfg")

LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, ln

net, ln = load_model()

# ------------------ DETECTION FUNCTION ------------------
def detect_objects(image):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    boxes, confidences, classIDs = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            color = [int(c) for c in COLORS[classIDs[i]]]
            label = f"{LABELS[classIDs[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🧠 YOLO Object Detection (Image & Video)")
st.write("Detect objects using YOLOv3 + OpenCV")

option = st.radio("Choose Input Type", ["Image", "Video"])

# ------------------ IMAGE MODE ------------------
if option == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR", caption="Original Image")
        result = detect_objects(image)
        st.image(result, channels="BGR", caption="Detected Objects")

# ------------------ VIDEO MODE ------------------
else:
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
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
