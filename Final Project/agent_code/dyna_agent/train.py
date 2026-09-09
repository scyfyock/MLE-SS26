import pickle
from typing import List

import events as e

from .callbacks import (
    ACTIONS,
    ACTION_TO_INDEX,
    MODEL_FILE,
    get_q_values,
    get_valid_actions,
    state_to_key,
)


# Custom events
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_NOT_TOWARD_COIN = "MOVED_NOT_TOWARD_COIN"

MOVEMENT_EVENTS = {
    e.MOVED_LEFT,
    e.MOVED_RIGHT,
    e.MOVED_UP,
    e.MOVED_DOWN,
}


# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95

INITIAL_EPSILON = 0.2
MIN_EPSILON = 0.02
EPSILON_DECAY = 0.995

# This remains zero until the basic Q-learning agent works.
PLANNING_STEPS = 0


def setup_training(self):
    """Initialize parameters used only during training."""

    self.alpha = LEARNING_RATE
    self.gamma = DISCOUNT_FACTOR
    self.planning_steps = PLANNING_STEPS

    # setup() may already have loaded epsilon from a saved model.
    if not hasattr(self, "epsilon"):
        self.epsilon = INITIAL_EPSILON

    self.logger.info(
        f"Training setup: alpha={self.alpha}, "
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

    if not MOVEMENT_EVENTS.intersection(events):
        return

    if self_action not in ACTION_TO_INDEX:
        return

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

    if coin_direction is None:
        return

    if ACTION_TO_INDEX[self_action] == coin_direction:
        events.append(MOVED_TOWARD_COIN)
    else:
        events.append(MOVED_NOT_TOWARD_COIN)


def update_q(
    self,
    state,
    action: str,
    reward: float,
    next_state,
    done: bool,
    next_valid_actions=None,
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

    q_values[action_index] += self.alpha * td_error

    return float(abs(td_error))


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

    state = state_to_key(old_game_state)
    next_state = state_to_key(new_game_state)
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

    self.logger.debug(
        f"Real Q-update: state={state}, action={self_action}, "
        f"reward={reward}, next_state={next_state}, "
        f"td_error={td_error:.4f}"
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

        state = state_to_key(last_game_state)
        reward = reward_from_events(self, events)

        td_error = update_q(
            self=self,
            state=state,
            action=last_action,
            reward=reward,
            next_state=None,
            done=True,
        )

        self.logger.debug(
            f"Terminal Q-update: state={state}, "
            f"action={last_action}, reward={reward}, "
            f"td_error={td_error:.4f}"
        )

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
        e.WAITED: -0.5,
        e.INVALID_ACTION: -1.0,
        MOVED_TOWARD_COIN: -0.1,
        MOVED_NOT_TOWARD_COIN: -0.5,
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
        "q_table": self.q_table,
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
        f"with epsilon={self.epsilon:.4f}."
    )