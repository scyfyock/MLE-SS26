import os
import pickle
import random
from collections import deque

import numpy as np
import settings as s

from .q_network import NeuralQFunction

# ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


ACTION_TO_INDEX = {
    action: index
    for index, action in enumerate(ACTIONS)
}

MODEL_FILE = os.environ.get(
    "DYNA_MODEL_FILE",
    "world-model.pt",
)
MODEL_SCHEMA_VERSION = 16
DEATH_RISK_PENALTY = float(
    os.environ.get(
        "DYNA_DEATH_RISK_PENALTY",
        os.environ.get("DYNA_SUICIDE_RISK_PENALTY", "18.0"),
    )
)
DEATH_RISK_FREE_THRESHOLD = float(
    os.environ.get("DYNA_DEATH_RISK_FREE_THRESHOLD", "0.0")
)
BOMB_CRATE_OUTCOME_VALUE = float(
    os.environ.get("DYNA_BOMB_CRATE_OUTCOME_VALUE", "2.0")
)
BOMB_KILL_OUTCOME_VALUE = float(
    os.environ.get("DYNA_BOMB_KILL_OUTCOME_VALUE", "25.0")
)
BOMB_KILL_PROBABILITY_THRESHOLD = float(
    os.environ.get("DYNA_BOMB_KILL_PROBABILITY_THRESHOLD", "0.10")
)
BOMB_ESCAPE_OUTCOME_PENALTY = float(
    os.environ.get("DYNA_BOMB_ESCAPE_OUTCOME_PENALTY", "4.0")
)
MAX_COIN_DISTANCE_BUCKET = 4
MAX_OPPONENT_DISTANCE_BUCKET = 4
SAFE_TIME = -1

SURVIVAL_ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT")
ACTION_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
}


def death_risk_cost(probability):
    """Map calibrated death probability to an optional thresholded cost."""

    excess = max(0.0, float(probability) - DEATH_RISK_FREE_THRESHOLD)
    scale = max(1e-6, 1.0 - DEATH_RISK_FREE_THRESHOLD)
    return DEATH_RISK_PENALTY * excess / scale


def predicted_action_risk(world_model, state, action_index):
    """Combine learned all-cause and self-inflicted death estimates."""

    risks = [0.0]
    if getattr(world_model, "is_death_model_ready", False):
        risks.append(
            world_model.predict_death_probability(state, action_index)
        )
    if getattr(world_model, "is_suicide_model_ready", False):
        risks.append(
            world_model.predict_suicide_probability(state, action_index)
        )
    return float(max(risks))


def bomb_outcome_adjustment(world_model, state):
    """Convert learned delayed bomb outcomes into an action-value correction."""

    if not getattr(world_model, "is_bomb_outcome_ready", False):
        return 0.0, None
    prediction = world_model.predict_bomb_outcome(state)
    # The class-weighted kill classifier deliberately favours recall.  Its raw
    # probability is therefore not calibrated and used to award a sizeable
    # bonus even for ordinary non-kill bombs.  Only confidence above the
    # measured background rate contributes, rescaled back to [0, 1].
    kill_excess = max(
        0.0,
        prediction.kill_probability - BOMB_KILL_PROBABILITY_THRESHOLD,
    )
    effective_kill_probability = kill_excess / max(
        1e-6,
        1.0 - BOMB_KILL_PROBABILITY_THRESHOLD,
    )
    adjustment = (
        BOMB_CRATE_OUTCOME_VALUE * prediction.crate_probability
        + BOMB_KILL_OUTCOME_VALUE * effective_kill_probability
        - BOMB_ESCAPE_OUTCOME_PENALTY * prediction.escape_probability
    )
    return float(adjustment), prediction

def setup(self):
    """Load an existing Dyna agent or initialize a neural Q-function."""

    # generate a random number generator with a fixed seed for reproducibility
    self.action_rng = random.Random(0)

    self.epsilon = 0.2
    self.q_network = NeuralQFunction()
    self.world_model = None
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

        self.q_network = saved_data["q_network"]
        if hasattr(self.q_network, "ensure_rare_replay_state"):
            self.q_network.ensure_rare_replay_state()
        self.world_model = saved_data["world_model"]
        if hasattr(self.world_model, "ensure_bomb_outcome_state"):
            self.world_model.ensure_bomb_outcome_state()
        if hasattr(self.world_model, "ensure_opponent_model_state"):
            self.world_model.ensure_opponent_model_state()
        self.epsilon = saved_data.get("epsilon", self.epsilon)
    else:
        self.logger.info("Initializing a fresh Double-DQN policy.")

def act(self, game_state: dict) -> str:
    state = state_to_key(
        game_state,
        previous_position=self.previous_position,
    )
    valid_actions = get_valid_actions(game_state)
    self.current_state_key = state
    self.previous_position = game_state["self"][3]

    q_values = get_q_values(self, state)
    world_model = getattr(self, "world_model", None)
    death_risks = {
        action: (
            predicted_action_risk(
                world_model,
                state,
                ACTION_TO_INDEX[action],
            )
            if world_model is not None
            else 0.0
        )
        for action in valid_actions
    }
    bomb_adjustment, bomb_outcome = (
        bomb_outcome_adjustment(world_model, state)
        if "BOMB" in valid_actions and world_model is not None
        else (0.0, None)
    )
    if bomb_outcome is not None:
        # A delayed bomb-outcome sample contains the complete result of placing
        # this bomb, whereas the generic one-step death head can miss a death
        # several ticks later.  Combine both estimates conservatively.
        death_risks["BOMB"] = max(
            death_risks["BOMB"],
            1.0 - bomb_outcome.survived_probability,
        )

    # Exploration remains possible, but learned high-risk actions are sampled
    # less often. A small floor ensures that the risk model can still correct
    # an initially wrong estimate from new real experience.
    if self.train and self.action_rng.random() < self.epsilon:
        exploration_weights = []
        for action in valid_actions:
            weight = max(0.05, (1.0 - death_risks[action]) ** 2)
            if action == "BOMB" and bomb_outcome is not None:
                kill_excess = max(
                    0.0,
                    bomb_outcome.kill_probability
                    - BOMB_KILL_PROBABILITY_THRESHOLD,
                )
                opportunity = (
                    bomb_outcome.crate_probability
                    + 3.0 * kill_excess
                )
                weight *= 1.0 + opportunity
            exploration_weights.append(weight)
        action = self.action_rng.choices(
            valid_actions,
            weights=exploration_weights,
            k=1,
        )[0]
        self.logger.debug(
            f"Risk-aware exploration selected {action}; "
            f"death_risks={death_risks}."
        )
        return action

    valid_q_values = {
        action: (
            q_values[ACTION_TO_INDEX[action]]
            - death_risk_cost(death_risks[action])
            + (bomb_adjustment if action == "BOMB" else 0.0)
        )
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
        f"Q={q_values.tolist()}, death_risks={death_risks}, "
        f"bomb_outcome={bomb_outcome}, "
        f"risk_adjusted_Q={valid_q_values}, selected={action}"
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

    opponent_direction_onehot = [0, 0, 0, 0]
    opponent_direction, opponent_distance = nearest_opponent_info(
        game_state["self"],
        game_state["others"],
        game_state["field"],
    )
    if opponent_direction != -1:
        opponent_direction_onehot[opponent_direction] = 1

    position = game_state["self"][3]
    walls = get_adjacent_tiles(position, game_state["field"])
    previous_tile_direction_onehot = [0, 0, 0, 0]

    if previous_position is not None:
        for neighbor, direction, _tile_type in walls:
            if neighbor == previous_position:
                previous_tile_direction_onehot[direction] = 1
                break

    danger_map = build_danger_map(game_state)
    danger_level = danger_bucket(danger_map[position])
    danger_schedule = build_danger_schedule(game_state)
    safe_directions = safe_action_mask(game_state, danger_schedule)
    escape_direction = find_escape_direction(
        game_state,
        danger_schedule=danger_schedule,
    )
    escape_direction_onehot = [0, 0, 0, 0, 0]

    if escape_direction is not None:
        escape_direction_onehot[ACTION_TO_INDEX[escape_direction]] = 1

    explosion_map = game_state.get("explosion_map")
    hazard_present = bool(game_state["bombs"]) or (
        explosion_map is not None
        and bool(np.any(explosion_map > 0))
    )
    if hazard_present:
        # Coin chasing and arrival history split identical escape problems into
        # many sparse table entries. During a hazard phase, the safety-related
        # features below contain the information relevant to survival.
        coin_direction_onehot = [0, 0, 0, 0]
        coin_distance = 0
        previous_tile_direction_onehot = [0, 0, 0, 0]

    opponent_positions = {
        other[3]
        for other in game_state["others"]
    }
    adjacent_opponents = [
        int(
            (
                position[0] + ACTION_DELTAS[action][0],
                position[1] + ACTION_DELTAS[action][1],
            )
            in opponent_positions
        )
        for action in ("UP", "RIGHT", "DOWN", "LEFT")
    ]
    (
        opponent_blast_directions,
        opponent_escape_routes,
        opponent_trapped,
        own_escape_routes_after_bomb,
    ) = bomb_tactical_features(game_state)

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
        danger_level,
    ] + safe_directions + escape_direction_onehot + [
        int(bool(game_state["self"][2])),
        int(bomb_would_hit_target(game_state)),
    ] + opponent_direction_onehot + [
        min(opponent_distance, MAX_OPPONENT_DISTANCE_BUCKET),
    ] + adjacent_opponents
    feature_vector += opponent_blast_directions + [
        opponent_escape_routes,
        opponent_trapped,
        own_escape_routes_after_bomb,
    ]

    return feature_vector

def state_to_key(game_state, previous_position=None):
    """Convert a game state into a compact discrete neural-network input."""
    if game_state is None:
        return None

    features = state_to_features(
        game_state,
        previous_position=previous_position,
    )
    return tuple(int(value) for value in features)

def get_q_values(self, state):
    """Return all action values predicted by the online Q-network."""

    return self.q_network.predict(state)

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


def nearest_opponent_info(agent, opponents, field):
    """Return direction and path distance to the nearest opponent."""

    positions = {
        opponent[3]
        for opponent in opponents
    }
    if not positions:
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
        if position in positions:
            return first_direction, distance

        for neighbor, _direction, tile_type in get_adjacent_tiles(
            position,
            field,
        ):
            if tile_type == 0 and neighbor not in visited:
                queue.append((neighbor, first_direction, distance + 1))
                visited.add(neighbor)

    return -1, 0


# Kept as a small compatibility helper for callers interested only in direction.
def direction_to_nearest_coin(agent, coins, field):
    direction, _distance = nearest_coin_info(agent, coins, field)
    return direction


def blast_coordinates(field, bomb_position):
    """Return blast tiles using the same wall rules as the environment."""

    x, y = bomb_position
    blast_tiles = [(x, y)]

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, s.BOMB_POWER + 1):
            position = (x + dx * distance, y + dy * distance)

            if field[position] == -1:
                break

            blast_tiles.append(position)
            # Crates are destroyed by a blast, but shield everything behind
            # them during that explosion.
            if field[position] == 1:
                break

    return blast_tiles


def build_danger_map(game_state):
    """Return the earliest time at which each tile becomes dangerous.

    ``SAFE_TIME`` denotes tiles not threatened by any currently known bomb.
    Active explosion tiles use time zero. A bomb timer is shifted by one
    because the environment explodes a timer-zero bomb after the next action.
    """

    field = game_state["field"]
    danger_map = np.full(field.shape, SAFE_TIME, dtype=np.int8)
    explosion_map = game_state.get("explosion_map")

    if explosion_map is not None:
        for x, y in np.argwhere(explosion_map > 0):
            danger_map[x, y] = 0

    for bomb_position, timer in game_state["bombs"]:
        explosion_time = max(1, int(timer) + 1)

        for x, y in blast_coordinates(field, bomb_position):
            previous_time = int(danger_map[x, y])
            if previous_time == SAFE_TIME or explosion_time < previous_time:
                danger_map[x, y] = explosion_time

    return danger_map


def build_danger_schedule(game_state):
    """Return dangerous positions for each future action time."""

    bomb_explosion_times = [
        max(1, int(timer) + 1)
        for _position, timer in game_state["bombs"]
    ]
    horizon = max([2] + [time + 1 for time in bomb_explosion_times])
    danger_schedule = [set() for _ in range(horizon + 1)]
    explosion_map = game_state.get("explosion_map")

    # An explosion visible in the current state remains dangerous while the
    # agent's next action is resolved.
    if explosion_map is not None:
        danger_schedule[1].update(
            (int(x), int(y))
            for x, y in np.argwhere(explosion_map > 0)
        )

    field = game_state["field"]
    for (bomb_position, timer), explosion_time in zip(
        game_state["bombs"],
        bomb_explosion_times,
    ):
        blast_tiles = blast_coordinates(field, bomb_position)
        danger_schedule[explosion_time].update(blast_tiles)
        danger_schedule[explosion_time + 1].update(blast_tiles)

    return danger_schedule


def danger_bucket(danger_time):
    """Compress exact danger time into a small categorical feature."""

    danger_time = int(danger_time)
    if danger_time == SAFE_TIME:
        return 0
    if danger_time <= 1:
        return 1
    if danger_time == 2:
        return 2
    return 3


def position_is_in_danger(game_state, position=None):
    """Return whether a position is threatened now or by a known bomb."""

    if position is None:
        position = game_state["self"][3]
    return int(build_danger_map(game_state)[position]) != SAFE_TIME


def safe_action_mask(game_state, danger_schedule=None):
    """Mark survival actions whose destination is safe after one action."""

    if danger_schedule is None:
        danger_schedule = build_danger_schedule(game_state)

    field = game_state["field"]
    x, y = game_state["self"][3]
    bomb_positions = {
        position
        for position, _timer in game_state["bombs"]
    }
    opponent_positions = {
        other[3]
        for other in game_state["others"]
    }
    immediate_danger = danger_schedule[1]
    mask = []

    for action in SURVIVAL_ACTIONS:
        dx, dy = ACTION_DELTAS[action]
        destination = (x + dx, y + dy)
        mechanically_valid = (
            action == "WAIT"
            or (
                field[destination] == 0
                and destination not in bomb_positions
                and destination not in opponent_positions
            )
        )
        mask.append(
            int(mechanically_valid and destination not in immediate_danger)
        )

    return mask


def find_escape_direction(game_state, danger_schedule=None):
    """Find the first action of a time-aware path out of all blast zones."""

    escape_actions = survivable_escape_actions(
        game_state,
        danger_schedule=danger_schedule,
    )
    return escape_actions[0] if escape_actions else None


def survivable_escape_actions(game_state, danger_schedule=None):
    """Return actions that begin at least one complete safe escape path."""

    if danger_schedule is None:
        danger_schedule = build_danger_schedule(game_state)

    start = game_state["self"][3]
    if not any(start in danger for danger in danger_schedule):
        return []

    field = game_state["field"]
    opponent_positions = {
        other[3]
        for other in game_state["others"]
    }
    bomb_clear_times = {
        position: max(1, int(timer) + 1)
        for position, timer in game_state["bombs"]
    }
    horizon = len(danger_schedule) - 1
    queue = deque([(start, 0, None)])
    visited = {(start, 0, None)}
    survivable_first_actions = set()

    while queue:
        position, time, first_action = queue.popleft()
        if time >= horizon:
            continue

        for action in SURVIVAL_ACTIONS:
            dx, dy = ACTION_DELTAS[action]
            destination = (position[0] + dx, position[1] + dy)
            arrival_time = time + 1

            if action != "WAIT":
                if field[destination] != 0:
                    continue
                if destination in opponent_positions:
                    continue
                if arrival_time <= bomb_clear_times.get(destination, -1):
                    continue

            if destination in danger_schedule[arrival_time]:
                continue

            path_first_action = first_action or action
            safe_for_remaining_time = all(
                destination not in danger_schedule[future_time]
                for future_time in range(arrival_time, horizon + 1)
            )

            if safe_for_remaining_time:
                survivable_first_actions.add(path_first_action)
                continue

            search_state = (
                destination,
                arrival_time,
                path_first_action,
            )
            if search_state not in visited:
                visited.add(search_state)
                queue.append(
                    (destination, arrival_time, path_first_action)
                )

    return [
        action
        for action in SURVIVAL_ACTIONS
        if action in survivable_first_actions
    ]


def bomb_would_hit_target(game_state):
    """Return whether a bomb here can hit a crate or current opponent."""

    field = game_state["field"]
    position = game_state["self"][3]
    opponent_positions = {
        other[3]
        for other in game_state["others"]
    }

    return any(
        field[blast_position] == 1
        or blast_position in opponent_positions
        for blast_position in blast_coordinates(field, position)
        if blast_position != position
    )


def can_escape_after_placing_bomb(game_state):
    """Return whether a survivable route would exist after choosing BOMB.

    This is an observation used for features and reward shaping only. It does
    not remove BOMB from the action set.
    """

    position = game_state["self"][3]
    simulated_state = dict(game_state)
    simulated_state["bombs"] = list(game_state["bombs"]) + [
        (position, max(0, s.BOMB_TIMER - 1))
    ]
    name, score, _bombs_left, _position = game_state["self"]
    simulated_state["self"] = (name, score, False, position)

    return find_escape_direction(simulated_state) is not None


def _free_neighbor_count(game_state, position, excluded=()):
    """Count mechanically free movement exits from an arbitrary position."""

    field = game_state["field"]
    occupied = {
        bomb_position
        for bomb_position, _timer in game_state["bombs"]
    }
    occupied.update(excluded)
    return sum(
        field[
            position[0] + ACTION_DELTAS[action][0],
            position[1] + ACTION_DELTAS[action][1],
        ] == 0
        and (
            position[0] + ACTION_DELTAS[action][0],
            position[1] + ACTION_DELTAS[action][1],
        ) not in occupied
        for action in ("UP", "RIGHT", "DOWN", "LEFT")
    )


def bomb_tactical_features(game_state):
    """Describe whether a bomb can pressure and trap the nearest opponent.

    Returns four directional blast-line flags, the opponent's free exit count,
    a trapped flag, and the number of our survivable first moves after placing
    a bomb. All values are translation invariant and cheap to compute.
    """

    own_position = game_state["self"][3]
    opponents = list(game_state["others"])
    blast_directions = [0, 0, 0, 0]
    opponent_escape_routes = 0
    opponent_trapped = 0

    if opponents:
        nearest = min(
            opponents,
            key=lambda other: (
                abs(other[3][0] - own_position[0])
                + abs(other[3][1] - own_position[1])
            ),
        )
        opponent_position = nearest[3]
        dx = opponent_position[0] - own_position[0]
        dy = opponent_position[1] - own_position[1]
        direction = None
        if dx == 0 and 0 < abs(dy) <= s.BOMB_POWER:
            direction = "DOWN" if dy > 0 else "UP"
        elif dy == 0 and 0 < abs(dx) <= s.BOMB_POWER:
            direction = "RIGHT" if dx > 0 else "LEFT"

        if (
            direction is not None
            and opponent_position
            in blast_coordinates(game_state["field"], own_position)
        ):
            blast_directions[ACTION_TO_INDEX[direction]] = 1

        excluded = {
            own_position,
            *(
                other[3]
                for other in opponents
                if other is not nearest
            ),
        }
        opponent_escape_routes = _free_neighbor_count(
            game_state,
            opponent_position,
            excluded=excluded,
        )
        opponent_trapped = int(
            any(blast_directions) and opponent_escape_routes <= 1
        )

    own_escape_routes_after_bomb = 0
    if game_state["self"][2]:
        simulated_state = dict(game_state)
        simulated_state["bombs"] = list(game_state["bombs"]) + [
            (own_position, max(0, s.BOMB_TIMER - 1))
        ]
        name, score, _bombs_left, _position = game_state["self"]
        simulated_state["self"] = (
            name,
            score,
            False,
            own_position,
        )
        own_escape_routes_after_bomb = sum(
            action != "WAIT"
            for action in survivable_escape_actions(simulated_state)
        )

    return (
        blast_directions,
        min(4, opponent_escape_routes),
        opponent_trapped,
        min(4, own_escape_routes_after_bomb),
    )


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

    # The environment accepts BOMB only while the agent has one available.
    # Danger is deliberately not filtered here: the policy must learn from
    # danger features and rewards which mechanically valid actions are safe.
    if game_state["self"][2] and (x, y) not in occupied:
        valid_actions.append("BOMB")

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
