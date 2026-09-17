import streamlit as st
import cv2
import numpy as np
import torch
import joblib
from skimage.feature import hog, local_binary_pattern
import torch.nn as nn

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------ CONFIG ------------------
IMG_SIZE = 48
label_map = joblib.load("labels.pkl")
EMOTIONS = list(label_map.values())

# ------------------ LOAD FILES ------------------
scaler = joblib.load("scaler.pkl")
selected_indices = joblib.load("features.pkl")

# ------------------ MODEL ------------------
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

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = ImprovedDNN(len(selected_indices), len(EMOTIONS))
model.load_state_dict(torch.load("emotion_model.pth", map_location="cpu"))
model.eval()

# ------------------ FEATURE EXTRACTION ------------------
def extract_features(img):
    hog_feat = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)

    lbp = local_binary_pattern(img, P=8, R=1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))

    return np.concatenate([hog_feat, lbp_hist])

# ------------------ UI ------------------
st.set_page_config(page_title="Emotion Detector", layout="centered")

st.title("😊 Facial Emotion Recognition")
st.write("Upload a face image and detect emotion instantly!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Emotion"):
        # Feature extraction
        feat = extract_features(img)

        # Scale
        feat = scaler.transform([feat])

        # Feature selection
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
            st.write(f"{emo}: {probs[0][i]*100:.2f}%")