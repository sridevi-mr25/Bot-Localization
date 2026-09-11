import numpy as np

# --- Directions ---

DIRECTIONS = {
    'Right': (0, 1),
    'Left': (0, -1),
    'Down': (1, 0),
    'Up': (-1, 0)
}
DIRECTION_ARRAY = np.array(list(DIRECTIONS.values()))
OPPOSITE = {'Up': 'Down', 'Down': 'Up', 'Left': 'Right', 'Right': 'Left'}