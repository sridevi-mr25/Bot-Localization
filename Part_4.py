import torch
from environment import create_maze
from part_2_belief_sets import * 
import matplotlib.pyplot as plt
import numpy as np
import random
from Training import ConvLocalizationRegressor
from L_next import bot_move
from constants import DIMENSION, CELL_OBSTACLE
from directions import DIRECTIONS

#estimate how many steps the bot will need to localize from a given belief set
def estimate_cost(model, belief_set, map_shape=(30, 30)):
    grid = np.zeros(map_shape, dtype=np.float32)
    for (i, j) in belief_set:
        grid[i][j] = -1  # belief
    grid[grid == 0] = 0.0
    grid[grid == -1] = 0.5
    input_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(input_tensor).item()
    return pred
#Policy pi1
def policy_pi1(L, model, maze):
    directions = ['Up', 'Down', 'Left', 'Right']
    costs = {}

    for dir in directions:
        L_next = bot_move(maze, L, dir)
        cost = estimate_cost(model, L_next)
        costs[dir] = cost

    best_action = min(costs, key=costs.get)
    L_updated = bot_move(maze, L, best_action)

    # Run π₀ from new belief set
    labeled_data = run_localization_with_intermediate_data(maze, L_updated)
    total_moves = 1 + len(labeled_data)
    return total_moves

#  Comparison  
def compare_pi0_pi1(model, num_trials=30):
    avg_moves_pi0 = []
    avg_moves_pi1 = []

    maze = create_maze()

    for size in range(2, 31):  
        moves_pi0 = []
        moves_pi1 = []

        for _ in range(num_trials):
            open_cells = [(i, j) for i in range(DIMENSION) for j in range(DIMENSION) if maze[i][j] == 0]
            initial_L = random.sample(open_cells, size)

            # π₀ strategy
            pi0_data = run_localization_with_intermediate_data(maze, initial_L)
            moves_pi0.append(len(pi0_data))

            # π₁ strategy
            moves = policy_pi1(initial_L, model, maze)
            moves_pi1.append(moves)

        avg_moves_pi0.append(np.mean(moves_pi0))
        avg_moves_pi1.append(np.mean(moves_pi1))

    return avg_moves_pi0, avg_moves_pi1


if __name__ == "__main__":
    # Load  trained model 
    model = ConvLocalizationRegressor()
    model.load_state_dict(torch.load("localization_cnn_model.pt"))
    model.eval()
# Run comparison 
    avg_moves_pi0, avg_moves_pi1 = compare_pi0_pi1(model)

    
        




