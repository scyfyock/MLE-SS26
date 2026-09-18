import os
import pickle
import random
from collections import deque
import settings as s

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MVMT_ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]


ACTION_TO_INDEX = {
    action: index
    for index, action in enumerate(ACTIONS)
}

MODEL_FILE = os.environ.get("DYNA_MODEL_FILE", "dyna-model.pt")
MODEL_SCHEMA_VERSION = 12
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

    # Block out tiles where the bomb timer is 0, agent keeps walking into explosions and killing itself for some reason
    x, y = game_state["self"][3]

    destinations = {
        "UP": (x, y - 1),
        "RIGHT": (x + 1, y),
        "DOWN": (x, y + 1),
        "LEFT": (x - 1, y),
        "WAIT": (x, y),
        "BOMB": (x, y),
    }

    invalid_actions = []

    for action, destination in destinations.items():
        if current_bomb_timer(destination, game_state['field'], game_state['bombs']) == 0:
            invalid_actions.append(action)

    # valid_actions = set(valid_actions) - set(invalid_actions)
    valid_actions_filter = [va for va in valid_actions if va not in invalid_actions]

    if valid_actions_filter:
        valid_actions = valid_actions_filter

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

    # Block out other players and pretend they are crates (tile_type = 1)
    field_with_players = game_state['field'].copy()
    for player in game_state['others']:
        field_with_players[player[3]] = 1


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

    crate_direction, crate_distance = nearest_crate_info(
        game_state["self"],
        field_with_players,
        game_state['explosion_map'],
        game_state['bombs'],
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



    # Get bomb countdown for current agent's tile
    current_agent_countdown = current_bomb_timer(game_state['self'][3], game_state['field'], game_state['bombs'])

    # Get the number of bombable crates if the agent dropped a bomb at it's current tile
    bombable_crates_or_opponents = check_bombable_crates(game_state['self'][3], game_state['field'])

    # treat an opponent as a crate to see if that incentivizes it to bomb then (rename to bombable_crates_or_opponents
    if bomb_would_hit_opponent(game_state):
        bombable_crates_or_opponents += 1

    # flatten nearby bombable crates for simplicity to not grow qtable
    bombable_crates_or_opponents = min(bombable_crates_or_opponents, 3)


    # Get the direction out of a bomb's way
    safety_directions = [0, 0, 0, 0]
    if current_agent_countdown != -1:
        safety_directions = safe_directions(game_state['self'][3], field_with_players, game_state['bombs'], game_state['explosion_map'])

    # Check if a safe spot exists if a bomb is placed this instant
    safety_exists = 0
    if bombable_crates_or_opponents != 0:
        fake_bombs = game_state['bombs'] + [(game_state['self'][3], 3)]
        if direction_to_safety(game_state['self'][3], field_with_players, fake_bombs, game_state['explosion_map'], move_timer=4) != -1:
            safety_exists = 1
        else:
            safety_exists = 2

    valid_check = [0, 0, 0, 0]
    valid_actions = get_valid_actions(game_state)
    for a in valid_actions:
        if a in MVMT_ACTIONS:
            valid_check[MVMT_ACTIONS.index(a)] = 1

    bomb_ticking = 0 if game_state['self'][2] else 1

    # Ignore bombable crates, crate direction and safe bombing exists when currently in a blast
    if bomb_ticking == 1:
        if any(safety_directions):
            bombable_crates_or_opponents, safety_exists, crate_direction = 0, 0, -1

    # The previous tile distinguishes immediate backtracking without tying the
    # policy to absolute coordinates. Distance is clipped to keep the table
    # compact while preserving the useful near/far signal.
    feature_vector = coin_direction_onehot + [
        bombable_crates_or_opponents,
    ] + valid_check + previous_tile_direction_onehot + safety_directions + [
        safety_exists,
        crate_direction,
        bomb_ticking
    ]


    """
    feature_vector key
    
    coin_direction_onehot: array of 4 integers (0/1) indicating the direction to the nearest coin, uses BFS. All 0s
        indicates no reachable coins
    bombable_crates_or_opponents: integer indicating the number of bombable crates/opponents if the agent were
        to bomb right now (0 to 3), where 3 means 3 or more crates are bombable currently. Enemy agents are counted in the total.
    valid_check: array of 4 integers (0/1) indicating whether nearby tiles can be moved into (get_valid_actions), e.g.
        not a bomb, crate, wall, agent, or explosion
    previous_tile_direction_onehot: array of 4 integers (0/1) indicating the direction it just came from
    safety_directions: array of 4 integers (0/1) indicating the directions to a safe tile out of a blast before the bomb
        explodes, all 0s indicates that no danger exists or the agent is trapped. Opponents block routes. Only computed
        when agent is within a blast.
    safety_exists: An integer (0/1/2) indicating if the agent were to bomb here if a safe escape exists. 
        0 = no crates/opponents in bomb range, 1 = crates/opponents in range, escape exists, 2 = crates/opponents
        in range, bomb cuts off escapes. The existing escape has to be escapable within 4 moves (bomb fuse)
    crate_direction: An integer (-1/0/1/2/3) indicating the direction to the nearest tile that the agent can
        bomb where the bomb will hit a crate. -1 = agent on tile where bombs will hit a crate and an escape exists OR 
        no tile with a bombable crate is reachable, OR while escaping agent's own bomb, 0/1/2/3 = directions to crate.
    bomb_ticking: An integer (0/1) indicating if the agent's bomb is currently placed on the environment, where 0 means
        bomb is available, and 1 means it is unavailable. When 1, we ignore bombable crates, crate direction and 
        safe bombing exists when agent's bomb is ticking and an escape exists.
    """

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

def nearest_crate_info(agent, field, explosion_map, bombs):
    """Return the first action and shortest-path distance to the nearest tile where placing a bomb on that tile
        hits a crate, and leaves an escape route for the agent"""

    bomb_array = [b for b, _ in bombs]
    start = agent[3]
    queue = deque()
    visited = {start}

    # Checks if the starting spot of the agent is a good spot to bomb right now
    # (Note: direction_to_safety checks fuse now)
    if (check_bombable_crates(start, field) > 0
            and direction_to_safety(start, field, bombs + [(start, 3)], explosion_map, move_timer=4) != -1):
        return -1, 0

    # Go through tiles that are walkable (not wall/crate or explosions/bombs) and add them into the queue
    for position, direction, tile_type in get_adjacent_tiles(start, field):
        if tile_type == 0 and position not in visited and position not in bomb_array and explosion_map[position] == 0:
            queue.append((position, direction, 1))
            visited.add(position)

    # BFS and for each tile recheck if the tile is a good spot to bomb right now
    while queue:
        position, first_direction, distance = queue.popleft()
        fake_bombs = bombs + [(position, 3)]

        # Same as first initial check: Should I bomb here?
        if check_bombable_crates(position, field) > 0 and direction_to_safety(position, field, fake_bombs, explosion_map, move_timer=4) != -1:
            return first_direction, distance

        for neighbor, _direction, tile_type in get_adjacent_tiles(
            position,
            field,
        ):
            if tile_type == 0 and neighbor not in visited and neighbor not in bomb_array:
                if field[neighbor] == 0 and explosion_map[neighbor] == 0:
                    queue.append(
                        (
                            neighbor,
                            first_direction,
                            distance + 1,
                        )
                    )
                    visited.add(neighbor)

    # Return if no viable tiles are found
    return -1, 0

# Reimplement this function for fighting enemies, but make enemy specific
def bomb_would_hit_opponent(game_state: dict) -> bool:
    """Return whether a bomb here can hit a current opponent.

    The ray casting deliberately matches ``Bomb.get_blast_coords`` in
    items.py: stone walls stop a blast, while crates do not stop subsequent
    blast tiles in this version of the environment.
    """

    field = game_state["field"]
    x, y = game_state["self"][3]
    opponent_positions = {
        other[3]
        for other in game_state["others"]
    }

    directions = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    )

    for dx, dy in directions:
        for distance in range(1, s.BOMB_POWER + 1):
            position = (x + dx * distance, y + dy * distance)
            tile_type = field[position]

            if tile_type == -1:
                break

            if position in opponent_positions:
                return True

    return False

# Kept as a small compatibility helper for callers interested only in direction.
def direction_to_nearest_coin(agent, coins, field):
    direction, _distance = nearest_coin_info(agent, coins, field)
    return direction

def get_valid_actions(game_state):
    """Return actions that can mechanically be executed."""

    field = game_state["field"]
    explosions = game_state["explosion_map"]
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
        if field[destination] == 0 and destination not in occupied and explosions[destination] == 0:
            valid_actions.append(action)

    # WAIT ist mechanisch immer möglich.
    valid_actions.append("WAIT")

    # The environment accepts BOMB only while the agent has one available.
    # Checking the current tile as well keeps this helper consistent with the
    # occupied-tile representation used for movement.
    if game_state["self"][2] and (x, y) not in occupied:
        valid_actions.append("BOMB")

    return valid_actions

# Returns the tiles left, right, up, down from the given x,y coordinate from the field
# format of return is (coords, direction, tile type)
# Return is [((x,y), ACTION_TO_INDEX['direction'], tile type), ...]
def get_adjacent_tiles(coord, field):
    # left, right, up, down tile types
    left = field[coord[0] - 1, coord[1]]
    right = field[coord[0] + 1, coord[1]]
    down = field[coord[0], coord[1] + 1]
    up = field[coord[0], coord[1] - 1]

    # Tiles are 0 if movable too, 1 for crates, and -1 for walls.
    # for adj_tiles[1] using 0 for UP, 1 for RIGHT, 2 for DOWN, and 3 for LEFT, -1 for no coin direction
    adjacent_tiles = [((coord[0] - 1, coord[1]), ACTION_TO_INDEX['LEFT'], left),
                      ((coord[0] + 1, coord[1]), ACTION_TO_INDEX['RIGHT'], right),
                      ((coord[0], coord[1] + 1), ACTION_TO_INDEX['DOWN'], down),
                      ((coord[0], coord[1] - 1), ACTION_TO_INDEX['UP'], up)]

    return adjacent_tiles

# Look through all the active bombs on the board and get the timer of the first bomb found that intersects with the agent
def current_bomb_timer(coord, board, bombs):
    live_bombs = []

    for (x,y), timer in bombs:
        # Check directly on the bomb
        if (x, y) == coord:
            live_bombs.append(timer)

        # Check four blast paths of bomb
        for lt in range(1, s.BOMB_POWER + 1):
            if board[x - lt, y] == -1:
                break
            elif (x - lt, y) == coord:
                live_bombs.append(timer)
                break

        for rt in range(1, s.BOMB_POWER + 1):
            if board[x + rt, y] == -1:
                break
            elif (x + rt, y) == coord:
                live_bombs.append(timer)
                break

        for dn in range(1, s.BOMB_POWER + 1):
            if board[x, y + dn] == -1:
                break
            elif (x, y + dn) == coord:
                live_bombs.append(timer)
                break

        for up in range(1, s.BOMB_POWER + 1):
            if board[x, y - up] == -1:
                break
            elif (x, y - up) == coord:
                live_bombs.append(timer)
                break

    if len(live_bombs) == 0:
        return -1
    else:
        return min(live_bombs)

# Get the number of bombable crates if the agent dropped a bomb at it's current tile
# based on the get_blast_coords function in items.py
def check_bombable_crates(coord, board):
    # dir = [left, right, down, up]
    directions = [0, 0, 0, 0]

    for lt in range(1, s.BOMB_POWER + 1):
        if board[coord[0] - lt, coord[1]] == -1:
            break
        elif board[coord[0] - lt, coord[1]] == 1:
            directions[0] += 1

    for rt in range(1, s.BOMB_POWER + 1):
        if board[coord[0] + rt, coord[1]] == -1:
            break
        elif board[coord[0] + rt, coord[1]] == 1:
            directions[1] += 1

    for dn in range(1, s.BOMB_POWER + 1):
        if board[coord[0], coord[1] + dn] == -1:
            break
        elif board[coord[0], coord[1] + dn] == 1:
            directions[2] += 1

    for up in range(1, s.BOMB_POWER + 1):
        if board[coord[0], coord[1] - up] == -1:
            break
        elif board[coord[0], coord[1] - up] == 1:
            directions[3] += 1

    return sum(directions)

# Check the adjacent tiles from the agent's tile to find every direction that an escape from a blast exists in
def safe_directions(coord, field, bombs, explosion_map):
    safety_dirs = [0, 0, 0, 0]
    bomb_coords = [b for b, _ in bombs]

    bomb_timer = current_bomb_timer(coord, field, bombs)

    for neighbor in get_adjacent_tiles(coord, field):
        if neighbor[2] == 0 and neighbor[0] not in bomb_coords and explosion_map[neighbor[0]] == 0: # tile type isnt a crate/wall, a bomb doesnt sit here, no explosion here
            if current_bomb_timer(neighbor[0], field, bombs) == -1: # tile is outside all blasts
                safety_dirs[neighbor[1]] = 1
            elif direction_to_safety(neighbor[0], field, bombs, explosion_map, move_timer = bomb_timer) != -1: # BFS from that tile to find if safety exists
                safety_dirs[neighbor[1]] = 1

    return safety_dirs

# Similar to direction to nearest coin, get the direction to a tile that is not currently being bombed
# Now tracks the bomb fuse and if it can be escaped in time : - )
def direction_to_safety(coord, board, bombs, explosion_map, move_timer = None):
    # Add a timer to the adj tiles to track the bomb timer
    adj_tiles = get_adjacent_tiles(coord, board)
    adj_tiles = [t + (1,) for t in adj_tiles]

    queue = deque(adj_tiles)
    visited = set()

    danger_map = set(tuple(coord) for coord in np.argwhere(explosion_map > 0))
    bomb_coords = [b for b, _ in bombs]

    # Build a danger map that covers every bomb's blast on the map
    for (x,y), timer in bombs:
        # Add the bomb directly
        danger_map.add((x, y))

        # Check four blast paths of bomb
        for lt in range(1, s.BOMB_POWER + 1):
            if board[x - lt, y] == -1:
                break
            else:
                danger_map.add((x - lt, y))

        for rt in range(1, s.BOMB_POWER + 1):
            if board[x + rt, y] == -1:
                break
            else:
                danger_map.add((x + rt, y))

        for dn in range(1, s.BOMB_POWER + 1):
            if board[x, y + dn] == -1:
                break
            else:
                danger_map.add((x, y + dn))

        for up in range(1, s.BOMB_POWER + 1):
            if board[x, y - up] == -1:
                break
            else:
                danger_map.add((x, y - up))

    # BFS through tiles
    while queue:
        current = queue.popleft()

        if current[2] == 0 and explosion_map[current[0]] == 0 and current[0] not in bomb_coords:
            if current[0] not in danger_map:
                if move_timer is None or current[3] <= move_timer: # if theres no move timer or the agent can move in time to escape the blast timer
                    return current[1]
                else:
                    return -1
            neighbors = get_adjacent_tiles(current[0], board)
        else:
            continue

        for neighbor in neighbors:
            if neighbor[2] == 0: # If there's no wall/crate here, we can move into the tile
                if neighbor[0] not in visited:
                    queue.append((neighbor[0], current[1], neighbor[2], current[3] + 1)) # always know tile type is 0 from earlier

        visited.add(current[0]) # Add to visited bfs queue

    return -1