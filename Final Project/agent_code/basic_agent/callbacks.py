import os
import pickle
import random

import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression

# ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

DIRECTIONS = {"LEFT": 0, "RIGHT": 1, "DOWN": 2, "UP": 3}

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        X = np.zeros((1, 5))
        y = np.zeros((1, 5))
        self.model = LinearRegression()
        self.model.fit(X, y)


    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    epsilon = .1
    feature_vector = state_to_features(game_state)

    if self.train and random.random() < epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2]) # Reduced action space for coin heaven

    self.logger.debug("Querying model for action.")
    q_vector = self.model.predict([feature_vector])
    argmax = np.argmax(q_vector, axis=1)[0]

    return ACTIONS[argmax]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Good features include:
    #   Situational awareness features, e.g. whether or not there is a wall to the left of your agent.
    #   Pathfinding features, e.g. the direction to move which brings you closest to the nearest coin.
    #   Life-saving features

    pathfinding = direction_to_nearest_coin(game_state['self'], game_state['coins'], game_state['field'])
    walls = get_adjacent_tiles(game_state['self'][3], game_state['field'])

    # Feature Vector
    feature_vector = [pathfinding, walls[0][2], walls[1][2], walls[2][2], walls[3][2]]

    return feature_vector

# Get the direction (left, right, up, down) that leads to the closest coin using bfs
def direction_to_nearest_coin(agent, coins, field):
    queue = deque(get_adjacent_tiles(agent[3], field))
    visited = []

    # BFS to find a coin
    while queue:
        current = queue.popleft()
        if current[0] in coins:
            return current[1]
        if current[2] != -1:
            neighbors = get_adjacent_tiles(current[0], field)
        else:
            continue
        for neighbor in neighbors:
            if neighbor[2] == 0: # If there's no wall/crate here, we can move into the tile
                if neighbor[0] not in visited:
                    queue.append((neighbor[0], current[1], neighbor[2])) # always know tile type is 0 from earlier
        visited.append(current[0])

    return -1 # No coin found in BFS

# Returns the tiles left, right, up, down from the given x,y coordinate from the field
# format of return is (coords, direction, tile type)
def get_adjacent_tiles(coord, field):
    # left, right, up, down tile types
    left = field[coord[0] - 1, coord[1]]
    right = field[coord[0] + 1, coord[1]]
    down = field[coord[0], coord[1] + 1]
    up = field[coord[0], coord[1] - 1]

    # Tiles are 0 if movable too, 1 for crates, and -1 for walls.
    # for adj_tiles[1] using 0 for LEFT, 1 for RIGHT, 2 for DOWN, and 3 for UP, -1 for no coin direction
    adjacent_tiles = [((coord[0] - 1, coord[1]), DIRECTIONS['LEFT'], left),
                      ((coord[0] + 1, coord[1]), DIRECTIONS['RIGHT'], right),
                      ((coord[0], coord[1] + 1), DIRECTIONS['DOWN'], down),
                      ((coord[0], coord[1] - 1), DIRECTIONS['UP'], up)]

    return adjacent_tiles