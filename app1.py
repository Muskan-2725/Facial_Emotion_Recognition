import streamlit as st
import cv2
import numpy as np
import torch
import joblib
from skimage.feature import hog, local_binary_pattern
import torch.nn as nn

#  CONFIG 
IMG_SIZE = 48

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

label_map = joblib.load("labels.pkl")
EMOTIONS = list(label_map.values())

#  LOAD FILES 
scaler = joblib.load("scaler.pkl")
selected_indices = joblib.load("features.pkl")


# MODEL
class ImprovedDNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)



@st.cache_resource
def load_model():
    m = ImprovedDNN(len(selected_indices), len(EMOTIONS))
    m.load_state_dict(torch.load("emotion_model.pth", map_location="cpu"))
    m.eval()
    return m


model = load_model()


#  FACE DETECTION 
def detect_and_crop_face(gray_img):
    """
    Runs the Haar cascade that was already being loaded but never used.
    Returns (cropped_face, num_faces_found).
    If no face is found, falls back to the full image so the app still
    works, but the UI tells the user this happened.
    """
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return gray_img, 0
    # If multiple faces are found, use the largest one (closest to camera /
    # most likely subject) rather than silently picking the first detection.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray_img[y:y + h, x:x + w], len(faces)


# FEATURE EXTRACTION 
def extract_features(img):
    hog_feat = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern(img, P=8, R=1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return np.concatenate([hog_feat, lbp_hist])


#  UI 
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("😊 Facial Emotion Recognition")
st.write("Upload a face image and detect emotion instantly!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Couldn't read that file as an image. Please upload a valid JPG/PNG.")
        st.stop()

    face_img, n_faces = detect_and_crop_face(img)

    if n_faces == 0:
        st.warning(
            "No face detected — using the full image. Results may be less "
            "reliable if there's background clutter."
        )
    elif n_faces > 1:
        st.info(f"{n_faces} faces detected — using the largest one.")

    resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    st.image(resized, caption="Processed face", use_container_width=True)

    if st.button("Detect Emotion"):
        try:
            # Feature extraction
            feat = extract_features(resized)

            
            expected_len = scaler.mean_.shape[0]
            if feat.shape[0] != expected_len:
                st.error(
                    f"Feature vector length mismatch: got {feat.shape[0]}, "
                    f"scaler expects {expected_len}. The feature extraction "
                    f"code no longer matches what the model was trained on."
                )
                st.stop()

            # Scale
            feat = scaler.transform([feat])

            # Feature selection (QIGA-selected indices)
            feat = feat[:, selected_indices]

            # Prediction
            with torch.no_grad():
                tensor = torch.tensor(feat, dtype=torch.float32)
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, 1).item()

            # Result
            st.success(f"Predicted Emotion: {EMOTIONS[pred]}")

            # Confidence scores
            st.subheader("Confidence Scores:")
            for i, emo in enumerate(EMOTIONS):
                st.write(f"{emo}: {probs[0][i] * 100:.2f}%")

        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")