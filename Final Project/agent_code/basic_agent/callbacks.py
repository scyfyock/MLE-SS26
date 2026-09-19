import os
import pickle
import random

import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression, Ridge

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'] # Action space for classic scenario
# ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT'] # Action space for coin-heaven

DIRECTIONS = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}

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
    if self.train or not os.path.isfile("q-learning-model.pt"):
        self.logger.info("Setting up model from scratch.")
        X = np.zeros((1, 19))
        y = np.zeros((1,))
        self.models = {action: Ridge(alpha=0.01) for action in ACTIONS}
        for action in ACTIONS:
            self.models[action].fit(X, y)

    else:
        self.logger.info("Loading model from saved state.")
        with open("q-learning-model.pt", "rb") as file:
            self.models = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    epsilon = .1 # was .1

    if self.train and random.random() < epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.18, .18, .18, .18, .08, .2])
        # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2]) # Reduced action space for coin heaven

    feature_vector = state_to_features(game_state)

    # np.argwhere(feature_vector[10:14] == -1 or feature_vector[10:14] == 1, feature_vector[a] = -999)
    self.logger.debug("Querying model for action.")
    q_vector = []
    for action in ACTIONS:
        q_vector.append(self.models[action].predict([feature_vector])[0])

    # Action masking in inference to block out blocks and crates to not pick those options
    for i in range(4):
        if feature_vector[i + 10] == 1 or feature_vector[i + 10] == -1:
            q_vector[i] = -999

    argmax = np.argmax(q_vector)

    self.logger.debug(f"Model chose action: {ACTIONS[argmax]}")
    self.logger.debug(f"Coin_direction currently is: {feature_vector[0:4]}")

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
    coin_direction_onehot = [0, 0, 0, 0]
    coin_pathfinding = direction_to_nearest_coin(game_state['self'], game_state['coins'], game_state['field'])

    if coin_pathfinding != -1:
        coin_direction_onehot[coin_pathfinding] = 1

    # Get the field's adjacent tiles, being 0 if movable too, 1 for crates, and -1 for walls.
    adj_tiles = get_adjacent_tiles(game_state['self'][3], game_state['field'])

    # Get the neighboring tiles explosion_map
    adj_explosions = get_adjacent_tiles(game_state['self'][3], game_state['explosion_map'])

    # Get bomb countdown for current agent's tile
    current_agent_countdown = current_bomb_timer(game_state['self'][3], game_state['field'], game_state['bombs'])

    # Get the number of bombable crates if the agent dropped a bomb at it's current tile
    bombable_crates = check_bombable_crates(game_state['self'][3], game_state['field'])

    # Prevent agent from making invalid move and bombing right after placing a bomb
    bombing_possible = game_state['self'][2]
    if bombing_possible:
        bombing_possible = 1
    else:
        bombing_possible = -1

    # Get the direction out of a bomb's way
    safety_direction_onehot = [0, 0, 0, 0]
    safety_direction = -1
    if current_agent_countdown != -1:
        safety_direction = direction_to_safety(game_state['self'][3], game_state['field'], game_state['bombs'], game_state['explosion_map'])

    if safety_direction != -1:
        safety_direction_onehot[safety_direction] = 1

    # Feature Vector
    feature_vector = coin_direction_onehot + safety_direction_onehot + [bombable_crates, current_agent_countdown, adj_tiles[0][2], adj_tiles[1][2], adj_tiles[2][2], adj_tiles[3][2], adj_explosions[0][2], adj_explosions[1][2], adj_explosions[2][2], adj_explosions[3][2], bombing_possible]

    return feature_vector

# Similar to direction to nearest coin, get the direction to a tile that is not currently being bombed
def direction_to_safety(coord, board, bombs, explosion_map):
    queue = deque(get_adjacent_tiles(coord, board))
    visited = set()

    danger_map = set(tuple(coord) for coord in np.argwhere(explosion_map > 0))
    bomb_coords = [b for b, _ in bombs]

    for (x,y), timer in bombs:
        # Add the bomb directly
        danger_map.add((x, y))

        # Check four blast paths of bomb
        for lt in range(1,4):
            if board[x - lt, y] == -1:
                break
            else:
                danger_map.add((x - lt, y))

        for rt in range(1,4):
            if board[x + rt, y] == -1:
                break
            else:
                danger_map.add((x + rt, y))

        for dn in range(1,4):
            if board[x, y + dn] == -1:
                break
            else:
                danger_map.add((x, y + dn))

        for up in range(1,4):
            if board[x, y - up] == -1:
                break
            else:
                danger_map.add((x, y - up))

    while queue:
        current = queue.popleft()

        if current[2] == 0 and explosion_map[current[0]] == 0 and current[0] not in bomb_coords:
            if current[0] not in danger_map:
                return current[1]
            neighbors = get_adjacent_tiles(current[0], board)
        else:
            continue

        for neighbor in neighbors:
            if neighbor[2] == 0: # If there's no wall/crate here, we can move into the tile
                if neighbor[0] not in visited:
                    queue.append((neighbor[0], current[1], neighbor[2])) # always know tile type is 0 from earlier

        visited.add(current[0]) # Add to visited bfs queue

    return -1

# Get the direction (left, right, up, down) that leads to the closest coin using bfs
def direction_to_nearest_coin(agent, coins, field):
    queue = deque(get_adjacent_tiles(agent[3], field))
    visited = set()

    # BFS to find a coin
    while queue:
        current = queue.popleft()

        if current[0] in coins: # if current tile is a coin, return the direction its in
            return current[1]

        if current[2] == 0: # if current tile is not a wall (-1) or a crate (1) we add to queue
            neighbors = get_adjacent_tiles(current[0], field)
        else:
            continue

        for neighbor in neighbors:
            if neighbor[2] == 0: # If there's no wall/crate here, we can move into the tile
                if neighbor[0] not in visited:
                    queue.append((neighbor[0], current[1], neighbor[2])) # always know tile type is 0 from earlier

        visited.add(current[0]) # Add to visited bfs queue

    return -1 # No coin found in BFS

# Returns the tiles left, right, up, down from the given x,y coordinate from the field
# format of return is (coords, direction, tile type)
# Board can now be either the field or explosion_map, if given explosion map adjacent_tiles[2] is an explosion countdown
def get_adjacent_tiles(coord, board):
    # left, right, up, down tile types
    left = board[coord[0] - 1, coord[1]]
    right = board[coord[0] + 1, coord[1]]
    down = board[coord[0], coord[1] + 1]
    up = board[coord[0], coord[1] - 1]

    # If given a field, Tile types are 0 if movable too, 1 for crates, and -1 for walls.
    # for adj_tiles[1] using 0 for LEFT, 1 for RIGHT, 2 for DOWN, and 3 for UP, -1 for no coin direction
    adjacent_tiles = [((coord[0], coord[1] - 1), DIRECTIONS['UP'], up),
                      ((coord[0] + 1, coord[1]), DIRECTIONS['RIGHT'], right),
                      ((coord[0], coord[1] + 1), DIRECTIONS['DOWN'], down),
                      ((coord[0] - 1, coord[1]), DIRECTIONS['LEFT'], left)]
    return adjacent_tiles

# Get the number of bombable crates if the agent dropped a bomb at it's current tile
# based on the get_blast_coords function in items.py
def check_bombable_crates(coord, board):
    # dir = [left, right, down, up]
    directions = [0, 0, 0, 0]

    for lt in range(1,4):
        if board[coord[0] - lt, coord[1]] == -1:
            break
        elif board[coord[0] - lt, coord[1]] == 1:
            directions[0] += 1

    for rt in range(1,4):
        if board[coord[0] + rt, coord[1]] == -1:
            break
        elif board[coord[0] + rt, coord[1]] == 1:
            directions[1] += 1

    for dn in range(1,4):
        if board[coord[0], coord[1] + dn] == -1:
            break
        elif board[coord[0], coord[1] + dn] == 1:
            directions[2] += 1

    for up in range(1,4):
        if board[coord[0] , coord[1] - up] == -1:
            break
        elif board[coord[0], coord[1] - up] == 1:
            directions[3] += 1

    return sum(directions)

# Look through all the active bombs on the board and get the timer of the first bomb found that intersects with the agent
def current_bomb_timer(coord, board, bombs):
    live_bombs = []

    for (x,y), timer in bombs:
        # Check directly on the bomb
        if (x, y) == coord:
            live_bombs.append(timer)

        # Check four blast paths of bomb
        for lt in range(1,4):
            if board[x - lt, y] == -1:
                break
            elif (x - lt, y) == coord:
                live_bombs.append(timer)
                break

        for rt in range(1,4):
            if board[x + rt, y] == -1:
                break
            elif (x + rt, y) == coord:
                live_bombs.append(timer)
                break

        for dn in range(1,4):
            if board[x, y + dn] == -1:
                break
            elif (x, y + dn) == coord:
                live_bombs.append(timer)
                break

        for up in range(1,4):
            if board[x, y - up] == -1:
                break
            elif (x, y - up) == coord:
                live_bombs.append(timer)
                break

    if len(live_bombs) == 0:
        return -1
    else:
        return min(live_bombs)