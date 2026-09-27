import os
import pickle
import random
from collections import deque

import numpy as np

# ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]


ACTION_TO_INDEX = {
    action: index
    for index, action in enumerate(ACTIONS)
}

MODEL_FILE = os.environ.get("DYNA_MODEL_FILE", "dyna-model.pt")
MODEL_SCHEMA_VERSION = 3
MAX_COIN_DISTANCE_BUCKET = 4

def setup(self):
    """Load an existing Dyna agent or initialize an empty Q-table."""

    # generate a random number generator with a fixed seed for reproducibility
    self.action_rng = random.Random(0)

    self.epsilon = 0.2
    self.q_table = {}
    self.world_model = {}
    self.previous_position = None
    self.current_state_key = None

    if os.path.isfile(MODEL_FILE):
        self.logger.info("Loading existing Dyna model.")

        with open(MODEL_FILE, "rb") as file:
            saved_data = pickle.load(file)

        saved_version = saved_data.get("model_schema_version", 1)

        if saved_version != MODEL_SCHEMA_VERSION:
            self.logger.warning(
                "Ignoring incompatible Dyna model schema "
                f"{saved_version}; expected {MODEL_SCHEMA_VERSION}. "
                "A fresh model will be trained."
            )
            return

        self.q_table = saved_data["q_table"]
        self.world_model = saved_data["world_model"]
        self.epsilon = saved_data.get("epsilon", self.epsilon)
    else:
        self.logger.info("Initializing an empty Q-table.")

def act(self, game_state: dict) -> str:
    state = state_to_key(
        game_state,
        previous_position=self.previous_position,
    )
    valid_actions = get_valid_actions(game_state)
    self.current_state_key = state
    self.previous_position = game_state["self"][3]

    # Exploration nur während des Trainings.
    if self.train and self.action_rng.random() < self.epsilon:
        action = self.action_rng.choice(valid_actions)
        self.logger.debug(f"Exploration selected {action}.")
        return action

    q_values = get_q_values(self, state)

    valid_q_values = {
        action: q_values[ACTION_TO_INDEX[action]]
        for action in valid_actions
    }

    best_value = max(valid_q_values.values())

    # Zufällige Auswahl bei Gleichstand verhindert den alten UP-Bias.
    best_actions = [
        action
        for action, value in valid_q_values.items()
        if np.isclose(value, best_value)
    ]

    action = self.action_rng.choice(best_actions)

    self.logger.debug(
        f"State={state}, valid={valid_actions}, "
        f"Q={q_values.tolist()}, selected={action}"
    )

    return action

def state_to_features(
    game_state: dict,
    previous_position=None,
) -> np.array:
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

    coin_direction, coin_distance = nearest_coin_info(
        game_state["self"],
        game_state["coins"],
        game_state["field"],
    )
    if coin_direction != -1:
        coin_direction_onehot[coin_direction] = 1

    position = game_state["self"][3]
    walls = get_adjacent_tiles(position, game_state["field"])
    previous_tile_direction_onehot = [0, 0, 0, 0]

    if previous_position is not None:
        for neighbor, direction, _tile_type in walls:
            if neighbor == previous_position:
                previous_tile_direction_onehot[direction] = 1
                break

    # The previous tile distinguishes immediate backtracking without tying the
    # policy to absolute coordinates. Distance is clipped to keep the table
    # compact while preserving the useful near/far signal.
    feature_vector = coin_direction_onehot + [
        walls[0][2],
        walls[1][2],
        walls[2][2],
        walls[3][2],
    ] + previous_tile_direction_onehot + [
        min(coin_distance, MAX_COIN_DISTANCE_BUCKET),
    ]

    return feature_vector

def state_to_key(game_state, previous_position=None):
    """Convert a game state into a hashable tabular state."""
    if game_state is None:
        return None

    features = state_to_features(
        game_state,
        previous_position=previous_position,
    )
    return tuple(int(value) for value in features)

def get_q_values(self, state):
    """Return all action values for a state, creating them if necessary."""

    if state not in self.q_table:
        self.q_table[state] = np.zeros(
            len(ACTIONS),
            dtype=np.float32,
        )

    return self.q_table[state]

def nearest_coin_info(agent, coins, field):
    """Return the first action and shortest-path distance to the nearest coin."""

    if not coins:
        return -1, 0

    start = agent[3]
    queue = deque()
    visited = {start}

    for position, direction, tile_type in get_adjacent_tiles(start, field):
        if tile_type == 0 and position not in visited:
            queue.append((position, direction, 1))
            visited.add(position)

    while queue:
        position, first_direction, distance = queue.popleft()

        if position in coins:
            return first_direction, distance

        for neighbor, _direction, tile_type in get_adjacent_tiles(
            position,
            field,
        ):
            if tile_type == 0 and neighbor not in visited:
                queue.append(
                    (
                        neighbor,
                        first_direction,
                        distance + 1,
                    )
                )
                visited.add(neighbor)

    return -1, 0


# Kept as a small compatibility helper for callers interested only in direction.
def direction_to_nearest_coin(agent, coins, field):
    direction, _distance = nearest_coin_info(agent, coins, field)
    return direction

def get_valid_actions(game_state):
    """Return actions that can mechanically be executed."""

    field = game_state["field"]
    x, y = game_state["self"][3]

    occupied = {
        position
        for position, _timer in game_state["bombs"]
    }

    occupied.update(
        other[3]
        for other in game_state["others"]
    )

    destinations = {
        "UP": (x, y - 1),
        "RIGHT": (x + 1, y),
        "DOWN": (x, y + 1),
        "LEFT": (x - 1, y),
    }

    valid_actions = []

    for action, destination in destinations.items():
        if field[destination] == 0 and destination not in occupied:
            valid_actions.append(action)

    # WAIT ist mechanisch immer möglich.
    valid_actions.append("WAIT")

    return valid_actions

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
    adjacent_tiles = [((coord[0] - 1, coord[1]), ACTION_TO_INDEX['LEFT'], left),
                      ((coord[0] + 1, coord[1]), ACTION_TO_INDEX['RIGHT'], right),
                      ((coord[0], coord[1] + 1), ACTION_TO_INDEX['DOWN'], down),
                      ((coord[0], coord[1] - 1), ACTION_TO_INDEX['UP'], up)]

    return adjacent_tiles
