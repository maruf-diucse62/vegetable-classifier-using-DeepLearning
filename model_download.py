import gdown
import os

def download_model():
    model_path = "best_vegetable_model.pth"

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1OwWgl_R5Ff8vxyqjQ7JXKtL4l8C5jvmy"
        print("Downloading model...")
        gdown.download(url, model_path, quiet=False, fuzzy=True)
    else:
        print("Model already exists.")

if __name__ == "__main__":
    download_model()
