import numpy as np
import random
from queue import PriorityQueue
from constants import *
from directions import *
from L_next import bot_move
from environment import create_maze
from part_2_belief_sets import *

#This function generates two types of belief sets a full belief set and a partial belief set(75% open cells)
def generate_mixed_data(num_mazes=200, fraction=0.75,mix_ratio=0.5, return_numpy=True, save_name='mixed_train_data.npz'):
    data = []

    for maze_idx in range(num_mazes):
        maze = create_maze()
        open_cells = [(i, j) for i in range(DIMENSION) for j in range(DIMENSION) if maze[i][j] == CELL_OPEN]

        if len(open_cells) < 2:
            continue
        # Randomly picks between partial belief set or full belief set
        use_partial_belief = random.random() < mix_ratio
        if use_partial_belief:
            sample_size = int(len(open_cells) * fraction)
            belief_set = random.sample(open_cells, sample_size)
            belief_type = 'partial'
        else:
            belief_set = open_cells
            belief_type = 'full'
        # Run the localization process and record data at each step
        episodes = run_localization_with_intermediate_data(maze, belief_set)
        # For every step, save the belief map and how many steps were left
        for belief_map, steps_left in episodes:
            data.append({
                "maze_id": maze_idx + 1,
                "belief_size": len(belief_set),
                "belief_type": belief_type,
                "belief_map": belief_map,
                "steps_left": steps_left
            })

    if return_numpy:
        X = np.stack([d["belief_map"] for d in data])  
        y = np.array([d["steps_left"] for d in data]) 
        np.savez(save_name, X=X, y=y)
        return data, (X, y)
    else:
        return data

# Generate a dataset with both full and partial belief sets
data, (X, y) = generate_mixed_data(num_mazes=200, fraction=0.75, mix_ratio=0.5,save_name='mixed_train_data.npz')
