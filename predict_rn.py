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

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("models/resnet_cat_dog.pth", map_location= "cuda"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]) 
])

def preprocess(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor

#streamlit

slt.title("cats or dogs?")
user_image = slt.file_uploader("Upload an Image!", type=["jpeg", "jpg", "png"])

if user_image is not None:
    img_tensor = preprocess(user_image)
    with torch.no_grad():
        outputs = model(img_tensor)

    probs = F.softmax(outputs, dim=1)[0]
    pred_class = torch.argmax(probs).item()
    labels = ["cat!", "dog!"]

    if probs[pred_class] < 0.8:
        slt.text(f"I don't think this is a cat or dog. (probability: {probs[pred_class]:.2f}")
        slt.image(user_image)

    else:
        slt.text(f"Prediction: {labels[pred_class]} (probability: {probs[pred_class]:.2f})")
        slt.image(user_image)
        
    slt.subheader("Confidence")
    cols = slt.columns(len(labels))

    for col, lbs, p in zip(cols, labels, probs):
        col.write(f"**{lbs}**")
        col.progress(int(p * 100))
        col.caption(f"{p:.2f}")