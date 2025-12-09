import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# ----------------------------
# 1. App settings
# ----------------------------
st.set_page_config(
    page_title="Vegetable Classification App",
    page_icon="🥦",
    layout="centered"
)

st.title("🥦 Vegetable Classification App")
st.write("Upload an image of a vegetable and get its class prediction.")

# ----------------------------
# 2. Google Drive model download
# ----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1OwWgl_R5Ff8vxyqjQ7JXKtL4l8C5jvmy"
MODEL_PATH = "best_vegetable_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model, please wait..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded!")

# ----------------------------
# 3. Define vegetable classes
# ----------------------------
class_names = [
    "Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli",
    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
]
num_classes = len(class_names)

# ----------------------------
# 4. Load model architecture
# ----------------------------
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Load ResNet50 architecture
model = models.resnet50(pretrained=False)

# Replace classifier to match number of classes
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes)
)

# Load checkpoint weights
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
st.success("Model loaded successfully!")

# ----------------------------
# 5. Image transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------------------------
# 6. Streamlit UI
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Image (JPG, PNG, JPEG)", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_t = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(img_t)
        predicted_idx = output.argmax(1).item()
        predicted_class = class_names[predicted_idx]

    # Display prediction
    st.markdown(f"""
        <div style='padding:10px; border-radius:10px; background-color:#E6F4EA'>
            <h3 style='color:#2E7D32;'>Predicted Class: {predicted_class}</h3>
        </div>
    """, unsafe_allow_html=True)

# ----------------------------
# 7. Footer
# ----------------------------
st.markdown("---")
st.markdown("Developed by Md. Abdullah Al Maruf | Powered by PyTorch & Streamlit")
