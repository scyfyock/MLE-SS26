import pickle
from typing import List
import random
from collections import deque
import os

import events as e
import settings as s

from .callbacks import (
    ACTIONS,
    ACTION_TO_INDEX,
    MODEL_FILE,
    MODEL_SCHEMA_VERSION,
    get_q_values,
    get_valid_actions,
    state_to_key,
)


# Custom events
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_NOT_TOWARD_COIN = "MOVED_NOT_TOWARD_COIN"
USEFUL_BOMB = "USEFUL_BOMB"
USELESS_BOMB = "USELESS_BOMB"

MOVEMENT_EVENTS = {
    e.MOVED_LEFT,
    e.MOVED_RIGHT,
    e.MOVED_UP,
    e.MOVED_DOWN,
}


# Hyperparameters
LEARNING_RATE = 0.1
PLANNING_LEARNING_RATE = float(
    os.environ.get("DYNA_PLANNING_ALPHA", "0.01")
)
DISCOUNT_FACTOR = 0.95

INITIAL_EPSILON = 0.2
MIN_EPSILON = 0.02
EPSILON_DECAY = 0.995

# This remains zero until the basic Q-learning agent works.
PLANNING_STEPS = int(os.environ.get("DYNA_PLANNING_STEPS", "10"))
MODEL_OUTCOMES_PER_PAIR = 100

def setup_training(self):
    """Initialize parameters used only during training."""

    self.alpha = LEARNING_RATE
    self.planning_alpha = PLANNING_LEARNING_RATE
    self.gamma = DISCOUNT_FACTOR
    self.planning_steps = PLANNING_STEPS
    self.planning_rng = random.Random(1)  # Random generator for planning steps

    if not hasattr(self, "world_model"):
        self.world_model = {}

    self.model_keys = list(self.world_model)

    # setup() may already have loaded epsilon from a saved model.
    if not hasattr(self, "epsilon"):
        self.epsilon = INITIAL_EPSILON

    self.logger.info(
        f"Training setup: alpha={self.alpha}, "
        f"planning_alpha={self.planning_alpha}, "
        f"gamma={self.gamma}, epsilon={self.epsilon}, "
        f"planning_steps={self.planning_steps}"
    )


def add_custom_events(
    old_game_state: dict,
    self_action: str,
    events: List[str],
):
    """Add events describing whether a movement followed the coin direction."""

    if old_game_state is None:
        return

    if MOVEMENT_EVENTS.intersection(events):
        old_state = state_to_key(old_game_state)
        coin_features = old_state[:4]

        coin_direction = next(
            (
                index
                for index, value in enumerate(coin_features)
                if value == 1
            ),
            None,
        )

        if coin_direction is not None:
            if ACTION_TO_INDEX[self_action] == coin_direction:
                events.append(MOVED_TOWARD_COIN)
            else:
                events.append(MOVED_NOT_TOWARD_COIN)

    # Only score bombs which the environment actually accepted. This avoids
    # classifying an attempted BOMB without ammunition as a useful placement.
    if self_action == "BOMB" and e.BOMB_DROPPED in events:
        if bomb_would_hit_target(old_game_state):
            events.append(USEFUL_BOMB)
        else:
            events.append(USELESS_BOMB)


def bomb_would_hit_target(game_state: dict) -> bool:
    """Return whether a bomb here can hit a crate or current opponent.

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

            if tile_type == 1 or position in opponent_positions:
                return True

    return False


def update_q(
    self,
    state,
    action: str,
    reward: float,
    next_state,
    done: bool,
    next_valid_actions=None,
    learning_rate=None,
):
    """Perform one tabular Q-learning update."""

    q_values = get_q_values(self, state)
    action_index = ACTION_TO_INDEX[action]

    current_q = q_values[action_index]

    if done:
        target = reward
    else:
        next_q_values = get_q_values(self, next_state)

        next_best_q = max(
            next_q_values[ACTION_TO_INDEX[next_action]]
            for next_action in next_valid_actions
        )

        target = reward + self.gamma * next_best_q

    td_error = target - current_q

    alpha = self.alpha if learning_rate is None else learning_rate
    q_values[action_index] += alpha * td_error

    return float(abs(td_error))

def store_transition(
    self,
    state,
    action,
    reward,
    next_state,
    done,
    next_valid_actions,
):
    """Store an observed transition in the learned world model."""

    model_key = (state, action)

    if model_key not in self.world_model:
        self.world_model[model_key] = deque(
            maxlen=MODEL_OUTCOMES_PER_PAIR
        )
        if not hasattr(self, "model_keys"):
            self.model_keys = []
        self.model_keys.append(model_key)

    stored_valid_actions = (
        tuple(next_valid_actions)
        if next_valid_actions is not None
        else tuple()
    )

    self.world_model[model_key].append(
        (
            reward,
            next_state,
            done,
            stored_valid_actions,
        )
    )

def perform_planning_updates(self):
    """Train Q-values from transitions sampled from the world model."""

    if not self.world_model:
        return []

    td_errors = []

    for _ in range(self.planning_steps):
        state, action = self.planning_rng.choice(self.model_keys)

        (
            reward,
            next_state,
            done,
            stored_valid_actions,
        ) = self.planning_rng.choice(
            self.world_model[(state, action)]
        )

        next_valid_actions = (
            list(stored_valid_actions)
            if not done
            else None
        )

        td_error = update_q(
            self=self,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            next_valid_actions=next_valid_actions,
            learning_rate=getattr(
                self,
                "planning_alpha",
                self.alpha,
            ),
        )

        td_errors.append(td_error)

    return td_errors

def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """Learn from one non-terminal real environment transition."""

    if old_game_state is None:
        return

    add_custom_events(
        old_game_state,
        self_action,
        events,
    )

    state = (
        getattr(self, "current_state_key", None)
        or state_to_key(old_game_state)
    )
    next_state = state_to_key(
        new_game_state,
        previous_position=old_game_state["self"][3],
    )
    reward = reward_from_events(self, events)

    next_valid_actions = get_valid_actions(new_game_state)

    td_error = update_q(
        self=self,
        state=state,
        action=self_action,
        reward=reward,
        next_state=next_state,
        done=False,
        next_valid_actions=next_valid_actions,
    )

    store_transition(
        self=self,
        state=state,
        action=self_action,
        reward=reward,
        next_state=next_state,
        done=False,
        next_valid_actions=next_valid_actions,
    )

    planning_td_errors = perform_planning_updates(self)

    mean_planning_error = (
        sum(planning_td_errors) / len(planning_td_errors)
        if planning_td_errors
        else 0.0
    )

    self.logger.debug(
        f"Real Q-update: state={state}, action={self_action}, "
        f"reward={reward}, next_state={next_state}, "
        f"td_error={td_error:.4f}, "
        f"planning_error={mean_planning_error:.4f}"
    )


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """Learn from the final transition and save the Q-table."""

    if last_game_state is not None and last_action in ACTION_TO_INDEX:
        add_custom_events(
            last_game_state,
            last_action,
            events,
        )

        state = (
            getattr(self, "current_state_key", None)
            or state_to_key(last_game_state)
        )
        reward = reward_from_events(self, events)

        td_error = update_q(
            self=self,
            state=state,
            action=last_action,
            reward=reward,
            next_state=None,
            done=True,
        )

        store_transition(
            self=self,
            state=state,
            action=last_action,
            reward=reward,
            next_state=None,
            done=True,
            next_valid_actions=None,
        )

        planning_td_errors = perform_planning_updates(self)

        mean_planning_error = (
            sum(planning_td_errors) / len(planning_td_errors)
            if planning_td_errors
            else 0.0
        )

        self.logger.debug(
            f"Terminal Q-update: state={state}, action={last_action}, "
            f"reward={reward}, td_error={td_error:.4f}, "
            f"planning_error={mean_planning_error:.4f}"
        )

    self.previous_position = None
    self.current_state_key = None

    self.epsilon = max(
        MIN_EPSILON,
        self.epsilon * EPSILON_DECAY,
    )

    save_model(self)


def reward_from_events(
    self,
    events: List[str],
) -> float:
    """Convert game events into a scalar training reward."""

    rewards = {
        e.COIN_COLLECTED: 5.0,
        e.CRATE_DESTROYED: 1.0,
        e.COIN_FOUND: 0.5,
        e.KILLED_OPPONENT: 5.0,
        e.KILLED_SELF: -10.0,
        e.GOT_KILLED: -5.0,
        e.SURVIVED_ROUND: 1.0,
        e.WAITED: -0.5,
        e.INVALID_ACTION: -1.0,
        MOVED_TOWARD_COIN: -0.1,
        MOVED_NOT_TOWARD_COIN: -0.5,
        USEFUL_BOMB: 0.5,
        USELESS_BOMB: -1.0,
    }

    reward = sum(
        rewards.get(event, 0.0)
        for event in events
    )

    self.logger.debug(
        f"Reward {reward} for events {events}"
    )

    return reward


def save_model(self):
    """Persist everything required to continue training."""

    saved_data = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "q_table": self.q_table,
        "world_model": self.world_model,
        "epsilon": self.epsilon,
    }

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(
            saved_data,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    self.logger.info(
        f"Saved {len(self.q_table)} Q-table states "
        f"{len(self.world_model)} model state-action pairs, "
        f"with epsilon={self.epsilon:.4f}."
    )
