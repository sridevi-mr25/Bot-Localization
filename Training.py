import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from constants import *

# Load Entire Mixed Dataset
data = np.load('mixed_train_data.npz')
X = data['X'].astype(np.float32)  
y = data['y'].astype(np.float32)  

#  Normalize values
X[X == -1] = 0.5  
X_tensor = torch.tensor(X).unsqueeze(1)  
y_tensor = torch.tensor(y).unsqueeze(1)  


full_dataset = TensorDataset(X_tensor, y_tensor)

# Split into Train/Test Sets 
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

#  Create Dataladers
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# CNN Model
class ConvLocalizationRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, 15, 15)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 7, 7)
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = ConvLocalizationRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_losses = []
    test_losses = []
    epochs = 100
    #Training the model
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                total_test_loss += loss.item() * xb.size(0)

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

    # Save model
    torch.save(model.state_dict(), "localization_cnn_model.pt")
