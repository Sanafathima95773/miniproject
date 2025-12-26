import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ======================================================
# Device
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# Load labels (MUST be 38)
# ======================================================
def load_labels():
    path = r"C:\Users\admin\Image based\Mini_project\ai_engine\data\class_mapping.json"
    with open(path, "r") as f:
        data = json.load(f)
    return [data[str(i)] for i in range(len(data))]

CLASSES = load_labels()
NUM_CLASSES = len(CLASSES)

print("Loaded classes:", NUM_CLASSES)  # MUST PRINT 38

# ======================================================
# Load trained ResNet18 (MATCH CHECKPOINT)
# ======================================================
def load_model(model_path):

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, NUM_CLASSES)  # MUST BE 38

    state = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model

# ======================================================
# Init model
# ======================================================
MODEL_PATH = r"C:\Users\admin\Image based\Mini_project\ai_engine\models\plant_resnet18.pth"
model = load_model(MODEL_PATH)

# ======================================================
# Transform
# ======================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================
# Prediction
# ======================================================
def predict_disease(image):

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)

    return CLASSES[idx.item()], conf.item()
