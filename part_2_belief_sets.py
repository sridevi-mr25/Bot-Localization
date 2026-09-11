import numpy as np
import random
from queue import PriorityQueue
from constants import *
from directions import *
from L_next import bot_move
from environment import create_maze  

# This function finds all the corners and dead-ends in the maze
def find_corners_or_dead_ends(maze):
    candidates = []
    for i in range(1, DIMENSION - 1):
        for j in range(1, DIMENSION - 1):
            if maze[i][j] == CELL_OPEN:
                open_neighbors = 0
                for dx, dy in DIRECTION_ARRAY:
                    ni, nj = i + dx, j + dy
                    if maze[ni][nj] == CELL_OPEN:
                        open_neighbors += 1
                if open_neighbors <= 1:
                    candidates.append((i, j))
    return candidates
# Heuristic function for A*
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
# A* search algorithm to find the shortest path from start to goal in the maze
def astar(maze, start, goal):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        _, current = frontier.get()
        if current == goal:
            break

        for i, (dx, dy) in enumerate(DIRECTION_ARRAY):
            next_cell = (current[0] + dx, current[1] + dy)
            if (0 <= next_cell[0] < DIMENSION and 0 <= next_cell[1] < DIMENSION and 
                maze[next_cell[0]][next_cell[1]] == CELL_OPEN):
                new_cost = cost_so_far[current] + 1
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + heuristic(goal, next_cell)
                    frontier.put((priority, next_cell))
                    came_from[next_cell] = current

   # Reconstruct the path from start to goal
    path = []
    current = goal
    while current != start:
        if current not in came_from:
            return []  # No path found
        prev = came_from[current]
        dx, dy = current[0] - prev[0], current[1] - prev[1]
        path.append(list(map(tuple, DIRECTION_ARRAY)).index((dx, dy)))  
        current = prev

    return path[::-1]

#simulates the bot's steps and records intermediate belief maps
def run_localization_with_intermediate_data(maze, initial_locations):
    L = set(initial_locations)
    targets = find_corners_or_dead_ends(maze)
    if not targets:
        return []

    target = random.choice(targets)
    belief_step_data = []

    total_moves = []
    while len(L) > 1:
        start = random.choice(list(L))
        path = astar(maze, start, target)
        if not path:
            break

        for direction_index in path:
            direction = list(DIRECTIONS.keys())[direction_index]

            belief_map = np.zeros((DIMENSION, DIMENSION), dtype=np.int8)

            # Mark obstacles as 1
            for i in range(DIMENSION):
                for j in range(DIMENSION):
                    if maze[i][j] == CELL_OBSTACLE:
                        belief_map[i][j] = 1

            # Mark belief set cells as -1
            for x, y in L:
                belief_map[x, y] = -1


            # Save the current state and how many steps are left 
            belief_step_data.append(belief_map)

            # Perform the move and update L
            L = bot_move(maze, L, direction)
            total_moves.append(direction)

            if len(L) == 1:
                break

    steps_total = len(total_moves)
    
    labeled_data = [
        (belief_map, steps_total - idx)
        for idx, belief_map in enumerate(belief_step_data)
    ]
    return labeled_data


if __name__ == "__main__":
    maze = create_maze()
    initial_L = [(i, j) for i in range(DIMENSION) for j in range(DIMENSION) if maze[i][j] == CELL_OPEN]
    
    episodes = run_localization_with_intermediate_data(maze, initial_L)
   
    print(f"Total steps taken: {len(episodes)}")

    
    


