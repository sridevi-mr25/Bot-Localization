import numpy as np
from constants import *
from directions import *
import random

def create_maze():
    grid = np.full((DIMENSION, DIMENSION), CELL_OPEN, dtype=int)
    grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1] = CELL_OBSTACLE, CELL_OBSTACLE, CELL_OBSTACLE, CELL_OBSTACLE

    for i in range(1, DIMENSION - 1):
        for j in range(1, DIMENSION - 1):
            grid[i, j] = CELL_OBSTACLE

    start_x, start_y = random.randint(1, DIMENSION - 2), random.randint(1, DIMENSION - 2)
    grid[start_x, start_y] = CELL_OPEN

    while True:
        rows, cols = np.where(grid[1:-1, 1:-1] == CELL_OBSTACLE)
        rows += 1
        cols += 1
        if not rows.size:
            break

        candidates = []
        for i, j in zip(rows, cols):
            neighbors = grid[i + DIRECTION_ARRAY[:, 0], j + DIRECTION_ARRAY[:, 1]]
            if np.sum(neighbors == CELL_OPEN) == 1:
                candidates.append((i, j))

        if not candidates:
            break

        x, y = random.choice(candidates)
        grid[x, y] = CELL_OPEN

    # Handle dead ends
    rows, cols = np.where(grid[1:-1, 1:-1] == CELL_OPEN)
    rows += 1
    cols += 1
    dead_ends = []
    for i, j in zip(rows, cols):
        if i in (0, DIMENSION - 1) or j in (0, DIMENSION - 1):
            continue
        neighbors = grid[i + DIRECTION_ARRAY[:, 0], j + DIRECTION_ARRAY[:, 1]]
        if np.sum(neighbors == CELL_OPEN) == 1:
            dead_ends.append((i, j))

    if dead_ends:
        num_dead_ends_to_open = len(dead_ends) // 2
        dead_ends_to_open = random.sample(dead_ends, num_dead_ends_to_open)
        for x, y in dead_ends_to_open:
            neighbors = grid[x + DIRECTION_ARRAY[:, 0], y + DIRECTION_ARRAY[:, 1]]
            closed_neighbors = np.where(neighbors == CELL_OBSTACLE)[0]
            if closed_neighbors.size:
                dx, dy = DIRECTION_ARRAY[random.choice(closed_neighbors)]
                if 1 <= x + dx < DIMENSION - 1 and 1 <= y + dy < DIMENSION - 1:
                    grid[x + dx, y + dy] = CELL_OPEN

    return grid
