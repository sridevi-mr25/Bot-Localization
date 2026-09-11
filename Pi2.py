import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from environment import create_maze
from part_2_belief_sets import *
from L_next import *
from PI1_model import ConvLocalizationRegressor, get_belief_map, estimate_cost
from constants import *


#  Load models 
C0_model = ConvLocalizationRegressor()
C0_model.load_state_dict(torch.load("localization_cnn_model.pt"))
C0_model.eval()

C1_model = ConvLocalizationRegressor()
C1_model.load_state_dict(torch.load("localization_pi1_model.pt"))
C1_model.eval()

# π₁ strategy
def run_pi1(L, model, maze):
    directions = ['Up', 'Down', 'Left', 'Right']
    costs = {}
    for dir in directions:
        L_next = bot_move(maze, L, dir)
        cost = estimate_cost(model, L_next)
        costs[dir] = cost

    best_action = min(costs, key=costs.get)
    L_updated = bot_move(maze, L, best_action)
    labeled_data = run_localization_with_intermediate_data(maze, L_updated)
    return 1 + len(labeled_data)

# π₂ strategy 
def run_pi2(L, C1_model, C0_model, maze):
    directions = ['Up', 'Down', 'Left', 'Right']
    costs = {}
    for dir in directions:
        L_next = bot_move(maze, L, dir)
        cost = estimate_cost(C1_model, L_next)
        costs[dir] = cost

    best_action = min(costs, key=costs.get)
    L_updated = bot_move(maze, L, best_action)

    # Follow π₁ from updated belief set using C0
    directions_pi1 = ['Up', 'Down', 'Left', 'Right']
    inner_costs = {}
    for dir in directions_pi1:
        inner_L = bot_move(maze, L_updated, dir)
        cost = estimate_cost(C0_model, inner_L)
        inner_costs[dir] = cost

    inner_best = min(inner_costs, key=inner_costs.get)
    final_L = bot_move(maze, L_updated, inner_best)

    labeled_data = run_localization_with_intermediate_data(maze, final_L)
    return 2 + len(labeled_data)

#  π₀ Strategy run
def run_pi0(L, maze):
    labeled_data = run_localization_with_intermediate_data(maze, L)
    return len(labeled_data)

#  Compare all 3 
def compare_all_policies(num_trials=20):
    results_pi0, results_pi1, results_pi2 = [], [], []

    for size in range(2, 31):
        moves0, moves1, moves2 = [], [], []

        for _ in range(num_trials):
            maze = create_maze()
            open_cells = [(i, j) for i in range(DIMENSION) for j in range(DIMENSION) if maze[i][j] == 0]
            L = set(random.sample(open_cells, size))

            try:
                moves0.append(run_pi0(L, maze))
                moves1.append(run_pi1(L, C0_model, maze))
                moves2.append(run_pi2(L, C1_model, C0_model, maze))
            except Exception:
                continue  # Skip malformed runs

        results_pi0.append(np.mean(moves0))
        results_pi1.append(np.mean(moves1))
        results_pi2.append(np.mean(moves2))

    return results_pi0, results_pi1, results_pi2


if __name__ == "__main__":
    
    avg0, avg1, avg2 = compare_all_policies(num_trials=20)


   
