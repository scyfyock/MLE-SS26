from collections import namedtuple, deque

import numpy as np
import pickle
import random
from typing import List

import events as e
from pyparsing import Empty

from .callbacks import state_to_features
from .callbacks import direction_to_nearest_coin
from .callbacks import DIRECTIONS, ACTIONS

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'episode'))


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 60000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 100

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_NOT_TOWARD_COIN = "MOVED_NOT_TOWARD_COIN"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.gamma = 0.9
    self.batch_size = BATCH_SIZE


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    # Idea: Add your own events to hand out rewards
    old_state_features = state_to_features(old_game_state)
    coin_direction_onehot = old_state_features[0:4]
    coin_direction = coin_direction_onehot.index(1) if any(coin_direction_onehot) else -1
    self.logger.debug(f'Coin direction: {coin_direction}')

    # Check to see if we moved in the direction of the closest coin
    if {e.MOVED_LEFT, e.MOVED_RIGHT, e.MOVED_UP, e.MOVED_DOWN} & set(events):
        if coin_direction != -1:
            if DIRECTIONS[self_action] == coin_direction:
                events.append(MOVED_TOWARD_COIN)
            else:
                events.append(MOVED_NOT_TOWARD_COIN)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(old_state_features, self_action, state_to_features(new_game_state), reward_from_events(self, events), new_game_state['round']))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events), last_game_state['round']))

    # In an effort to not pick multiple transitions from the same game:
    randomization = random.sample(self.transitions, len(self.transitions))
    seen = []
    batch = []
    for r in randomization:
        if r.episode not in seen:
            batch.append(r)
            seen.append(r.episode)
        # if len(seen) == self.batch_size:
        #     break

    # sample = random.sample(self.transitions, self.batch_size)
    targets_store = []
    states_store = []
    errors_store = []

    # 9.18 weighted sampling to fix agent getting stuck at bottom of board?
    for b in batch:
        pred = self.model.predict([b.state])[0]

        if b.next_state is None:
            target = b.reward
            td_error = abs(target - pred[ACTIONS.index(b.action)])
        else:
            target = b.reward + self.gamma * np.max(self.model.predict([b.next_state])[0])
            model_prediction = pred[ACTIONS.index(b.action)]
            td_error = np.abs(target - model_prediction)
        pred[ACTIONS.index(b.action)] = target      # Rewrite DIRECTIONS so we can use index?

        targets_store.append(pred)
        states_store.append(b.state)
        errors_store.append(td_error)

    combined = list(zip(states_store, targets_store, errors_store))
    highest_errors = sorted(combined, key=lambda x: x[2], reverse=True)[:self.batch_size]
    states_errors, targets_errors, _ = zip(*highest_errors)
    self.model.fit(states_errors, targets_errors)

    # Store the model
    with open("q-learning-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.WAITED: -0.5,
        e.INVALID_ACTION: -1,
        e.KILLED_OPPONENT: 5,
        MOVED_TOWARD_COIN: -0.1,
        MOVED_NOT_TOWARD_COIN: -0.5,
    }
    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum
