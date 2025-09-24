import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image
import argparse
import os.path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="Path to the input image")
args = parser.parse_args()

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

transform = transforms.Compose([
    transforms.Resize((32, 32)),          
    transforms.ToTensor(),              
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  
    return tensor

img_tensor = preprocess_image(args.image_path)

with torch.no_grad():
    outputs = model(img_tensor)

probs = F.softmax(outputs, dim=1)
pred_class = torch.argmax(probs, dim=1).item()
labels = ["Cat 🐱", "Dog 🐶"]
print(f"Prediction: {labels[pred_class]} (probability: {probs[0][pred_class]:.2f})")

