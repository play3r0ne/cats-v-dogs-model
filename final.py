import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image
import argparse
import os.path
import streamlit as slt
from PIL import Image
from torchvision import models

#nn

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
        
model_rn  = models.resnet18(weights=None)
model_rn.fc = nn.Linear(model_rn.fc.in_features, 2)
model_rn.load_state_dict(torch.load("models/resnet_cat_dog.pth", map_location= "cuda"))
model_rn.eval()

model_cnn  = CNN()
model_cnn.load_state_dict(torch.load("models/cnn_cat_dog.pth"))
model_cnn.eval()


transform_rn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]) 
])

transform_cnn = transforms.Compose([
    transforms.Resize((32, 32)),          
    transforms.ToTensor(),              
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def preprocess_rn(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    tensor = transform_rn(image).unsqueeze(0)
    return tensor

def preprocess_cnn(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    tensor = transform_cnn(image).unsqueeze(0)  
    return tensor

#streamlit

slt.title("cats or dogs?")
chosen_model = slt.selectbox("Choose your model: ", ("ResNet18", "CustomCNN"))
user_image = slt.file_uploader("Upload an Image!", type=["jpeg", "jpg", "png"])

if user_image is not None:

    if chosen_model == "ResNet18":
        img_tensor = preprocess_rn(user_image)
        with torch.no_grad():
            outputs = model_rn(img_tensor)
    else:
        img_tensor = preprocess_cnn(user_image)
        with torch.no_grad():
            outputs = model_cnn(img_tensor)

    probs = F.softmax(outputs, dim=1)[0]
    pred_class = torch.argmax(probs).item()
    labels = ["cat!", "dog!"]

    if probs[pred_class] < 0.8:
        slt.text(f"I don't think this is a cat or dog. probability: {probs[pred_class]:.2f}")
        slt.image(user_image)

    else:
        slt.text(f"Prediction: {labels[pred_class]} probability: {probs[pred_class]:.2f}")
        slt.image(user_image)
        
    slt.subheader("Confidence")
    cols = slt.columns(len(labels))

    for col, lbs, p in zip(cols, labels, probs):
        col.write(f"**{lbs}**")
        col.progress(int(p * 100))
        col.caption(f"{p:.2f}")