# app.py
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import requests
import os
from io import BytesIO

# ---------- CONFIG ----------
# Google Drive file ID from your link:
DRIVE_FILE_ID = "1OwWgl_R5Ff8vxyqjQ7JXKtL4l8C5jvmy"
MODEL_PATH_ON_DISK = "model.pth"
NUM_CLASSES = 15

# ---------- UTIL: download from Google Drive (handles large files) ----------
def download_file_from_google_drive(file_id: str, destination: str):
    """
    Downloads a file from Google Drive, handling the "virus scan / confirm" step for large files.
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    # fallback: try to find token in response content (rare)
    return None

def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

# ---------- Define your model architecture (must match how model was saved) ----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # NOTE: if your saved model used a different final feature size, change 64*56*56 here
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ---------- Caching: download & load model once per session ----------
@st.cache_resource(show_spinner=False)
def load_model():
    # download if not present
    if not os.path.exists(MODEL_PATH_ON_DISK):
        st.info("Downloading model from Google Drive...")
        try:
            download_file_from_google_drive(DRIVE_FILE_ID, MODEL_PATH_ON_DISK)
        except Exception as e:
            st.error("Failed to download model: " + str(e))
            raise

    # load into memory
    model = SimpleCNN(num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH_ON_DISK, map_location="cpu")
    # if the uploaded file contains a dict with 'model_state_dict' adjust accordingly.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# ---------- Class names (change order if your model uses a different mapping) ----------
CLASS_NAMES_
