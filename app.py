import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import gdown

# ----------------------------
# 1. Download model from Google Drive
# ----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1OwWgl_R5Ff8vxyqjQ7JXKtL4l8C5jvmy"
MODEL_PATH = "best_vegetable_model.pth"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ----------------------------
# 2. Define your model architecture
# ----------------------------
class VegetableModel(nn.Module):
    def __init__(self):
        super(VegetableModel, self).__init__()
        # Replace with your actual architecture
        self.fc = nn.Linear(224*224*3, 10)  # example for 10 classes

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------------------
# 3. Load the model
# ----------------------------
model = VegetableModel()

try:
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    else:
        # If torch.load returns a full model (not state_dict)
        model = state_dict
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ----------------------------
# 4. Define image transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------------------------
# 5. Streamlit UI
# ----------------------------
st.title("Vegetable Classification App")
st.write("Upload an image to classify the vegetable")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        predicted = output.argmax(1).item()

    st.success(f"Predicted Class: **{predicted}**")
