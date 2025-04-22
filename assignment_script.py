# For reading data
import os
import pandas as pd
import urllib.request
import gzip
import numpy as np

# For visualizing
import plotly.express as px

# For model building
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"{filename} already exists.")

def read_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)  # Skip header
        return np.frombuffer(f.read(), dtype=np.uint8)

def read_images(filename):
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)

    
class CustomMNIST(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0) / 255.0  # normalize
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label
    
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.LazyConv2d(6, 5, padding=2)
        self.conv2 = nn.LazyConv2d(16, 5)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# === Accuracy Calculator ===
def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# === Training Loop ===
def train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, device):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_acc = calculate_accuracy(model, train_loader, device)
        test_acc = calculate_accuracy(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# === Data Setup ===
urls = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
}

for name, url in urls.items():
    download_file(url, f"{name}.gz")

train_images = read_images("train_images.gz")
train_labels = read_labels("train_labels.gz")
test_images = read_images("test_images.gz")
test_labels = read_labels("test_labels.gz")

train_data = CustomMNIST(train_images, train_labels)
test_data = CustomMNIST(test_images, test_labels)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# === Model Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
epochs = 5

train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, device)

# === Save Model ===
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, "model.pt")

# === Load Model ===
model = LeNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
checkpoint = torch.load("model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Model loaded successfully from epoch {checkpoint['epoch']}!")