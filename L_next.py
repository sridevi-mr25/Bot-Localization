import numpy as np
from constants import *
from directions import *
from environment import *

def bot_move(maze,locations,direction) :
    
    dx , dy = DIRECTIONS[direction]
    locations_next = set()
    
    for x,y in locations:
        next_x, next_y = x + dx , y + dy
        if 0 <= next_x < DIMENSION and 0 <= next_y < DIMENSION and maze[next_x][next_y] != CELL_OBSTACLE:
            locations_next.add((next_x,next_y))
        else:
            locations_next.add((x,y))
    return locations_next




