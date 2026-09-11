from environment import create_maze
from part_2_belief_sets import *
from Part_4 import estimate_cost
from L_next import bot_move
from Training import ConvLocalizationRegressor
from constants import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import random
import matplotlib.pyplot as plt


#Creates a belief map
def get_belief_map(maze, L):
    belief_map = np.zeros((DIMENSION, DIMENSION), dtype=np.int8)
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            if maze[i][j] == CELL_OBSTACLE:
                belief_map[i][j] = 1
    for x, y in L:
        belief_map[x][y] = -1
    return belief_map

# Generate Training Data from π₁ 
def generate_pi1_training_data(model, num_episodes=200):
    data = []
    for _ in range(num_episodes):
        maze = create_maze()
        open_cells = [(i, j) for i in range(DIMENSION) for j in range(DIMENSION) if maze[i][j] == 0]
        L = set(random.sample(open_cells, random.randint(2, 30)))

        directions = ['Up', 'Down', 'Left', 'Right']
        costs = {}

        for dir in directions:
            L_next = bot_move(maze, L, dir)
            cost = estimate_cost(model, L_next)
            costs[dir] = cost

        best_action = min(costs, key=costs.get)
        L_updated = bot_move(maze, L, best_action)

        labeled_data = run_localization_with_intermediate_data(maze, L_updated)
        first_map = get_belief_map(maze, L)
        data.append((first_map, len(labeled_data) + 1))
        data.extend(labeled_data)
    return data

#Train a new model pi_1
def train_pi1_model(data):
    X = np.array([d[0] for d in data], dtype=np.float32)
    y = np.array([d[1] for d in data], dtype=np.float32)

    # Normalize belief maps
    X[X == -1] = 0.5
    X[X == 1] = 1.0

    X_tensor = torch.tensor(X).unsqueeze(1)
    y_tensor = torch.tensor(y).unsqueeze(1)
    # Create training and testing datasets
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = ConvLocalizationRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    #Train for 50 epochs
    for epoch in range(50):
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)

        train_losses.append(total_train_loss / len(train_loader.dataset))
        #Evaluating on test set
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                total_test_loss += loss.item() * xb.size(0)
        test_losses.append(total_test_loss / len(test_loader.dataset))
       
    # Save model
    torch.save(model.state_dict(), "localization_pi1_model.pt")

    return model

if __name__ == "__main__":
    
    pi0_model = ConvLocalizationRegressor()
    pi0_model.load_state_dict(torch.load("localization_cnn_model.pt"))
    pi0_model.eval()
    pi1_data = generate_pi1_training_data(pi0_model, num_episodes=500)
    np.savez_compressed("pi1_train_data.npz", X=[x for x, _ in pi1_data], y=[y for _, y in pi1_data])
    train_pi1_model(pi1_data)