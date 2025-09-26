import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as slt
from PIL import Image

class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.pool  = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc1   = nn.Linear(32 * 8 * 8, 128)
            self.fc2   = nn.Linear(128, 2)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)
        
model = CNN()
model.load_state_dict(torch.load("models/cnn_cat_dog.pth"))

slt.title("cats or dogs?")
user_image = slt.file_uploader("upload an image!", type=["jpg", "png", "jpeg"])

transform = transforms.Compose([
    transforms.Resize((32, 32)),          
    transforms.ToTensor(),              
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    tensor = transform(image).unsqueeze(0)  
    return tensor

if user_image is not None:
    img_tensor = preprocess_image(user_image)
    with torch.no_grad():
        outputs = model(img_tensor)
    probs = F.softmax(outputs, dim=1)[0]
    pred_class = torch.argmax(probs).item()
    labels = ["Cat 🐱", "Dog 🐶"]

    if probs[pred_class] < 0.8:
        slt.text(f"I don't think this is a cat or a dog (probability: {probs[pred_class]:.2f})")
        slt.image(user_image)
    else:
        slt.text(f"Prediction: {labels[pred_class]} (probability: {probs[pred_class]:.2f})")
        slt.image(user_image)

    slt.subheader("Confidence")
    cols = slt.columns(len(labels))

    for col, lbs, p in zip(cols, labels, probs):
        col.write(f"**{lbs}**")
        col.progress(int(p * 100))
        col.caption(f"{p:.2%}")
