import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]) 
])

train_data = datasets.CIFAR10("data/", train=True, transform=transform)
val_data   = datasets.CIFAR10("data/", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data, batch_size=512, shuffle=False)

model = models.resnet18(weights= models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

params_to_update = model.fc.parameters()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = 0.001)


epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for inputs, labels in train_loader:
        labels = (labels == 5).long()

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = (labels == 5).long()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()  
    epoch_time = end_time - start_time

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} "
          f"Loss: {running_loss/len(train_loader):.4f}, "
          f"Val Acc: {acc:.2f}%, "
          f"Time: {epoch_time:.2f}")

    
torch.save(model.state_dict(), "resnet_cat_dog.pth")
print("Model saved successfully!")
print("Training complete ✅")