import torch
import torchvision
import time
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

torch.backends.cudnn.benchmark = True

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    cat_dog_train_idx = [i for i, label in enumerate(trainset.targets) if label in [3, 5]]
    cat_dog_test_idx  = [i for i, label in enumerate(testset.targets) if label in [3, 5]]
    trainset_cd = Subset(trainset, cat_dog_train_idx)
    testset_cd  = Subset(testset,  cat_dog_test_idx)

    trainloader = DataLoader(trainset_cd, batch_size=1024, shuffle=True,
                             num_workers=4, pin_memory=True, prefetch_factor=4)
    testloader  = DataLoader(testset_cd,  batch_size=1024, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"Training samples: {len(trainset_cd)} | Test samples: {len(testset_cd)}")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            labels = (labels == 5).long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        end_time = time.time()  
        epoch_time = end_time - start_time

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss/len(trainloader):.4f}, "
              f"Accuracy: {100*correct/total:.2f}%"
              f"Time: {epoch_time:.2f} seconds")
    
    torch.save(model.state_dict(), "cnn_cat_dog.pth")
    print("Model saved successfully!")
            
if __name__ == "__main__":
    main()
