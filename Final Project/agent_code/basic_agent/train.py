import collections
from collections import namedtuple, deque

import numpy as np
import pickle
import random
from typing import List
import copy

import events as e
from pyparsing import Empty

from .callbacks import DIRECTIONS, ACTIONS, state_to_features, direction_to_nearest_coin

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'episode'))


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 60000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 100
MODEL_SYNC = 10

# Events
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_NOT_TOWARD_COIN = "MOVED_NOT_TOWARD_COIN"
CHOSE_TO_BOMB_CRATES = "CHOSE_TO_BOMB_CRATES"
BOMBED_MANY_CRATES = "BOMBED_MANY_CRATES"
STANDING_IN_BLAST = "STANDING_IN_BLAST"
MOVED_INTO_BLAST = "MOVED_INTO_BLAST"
ESCAPED_BLAST = "ESCAPED_BLAST"
MOVED_TOWARD_SAFETY = "MOVED_TOWARD_SAFETY"
MOVED_NOT_TOWARD_SAFETY = "MOVED_NOT_TOWARD_SAFETY"


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
    self.target_models = copy.deepcopy(self.models)
    self.sync = MODEL_SYNC


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
    new_state_features = state_to_features(new_game_state)

    coin_direction_onehot = old_state_features[0:4]
    coin_direction = coin_direction_onehot.index(1) if any(coin_direction_onehot) else -1
    self.logger.debug(f'Coin direction: {coin_direction}')

    safety_direction_onehot = old_state_features[4:8]
    safety_direction = safety_direction_onehot.index(1) if any(safety_direction_onehot) else -1
    self.logger.debug(f'Safe direction: {safety_direction}')

    # Punish agent for standing in blast zones, -1 is safe
    old_blast_timer = old_state_features[9]
    new_blast_timer = new_state_features[9]

    if old_blast_timer != -1 and new_blast_timer == -1: # Was in danger before, now safe
        events.append(ESCAPED_BLAST)
    elif old_blast_timer == -1 and new_blast_timer != -1: # Was safe before, now in danger
        events.append(MOVED_INTO_BLAST)
    elif old_blast_timer != -1 and new_blast_timer != -1: # Was in danger before, still is currently in danger
        events.append(STANDING_IN_BLAST)

    # Check to see if we moved in the direction of the closest coin
    if {e.MOVED_LEFT, e.MOVED_RIGHT, e.MOVED_UP, e.MOVED_DOWN} & set(events):
        if coin_direction != -1:
            if DIRECTIONS[self_action] == coin_direction:
                events.append(MOVED_TOWARD_COIN)
            else:
                if old_state_features[9] == -1:
                    events.append(MOVED_NOT_TOWARD_COIN)

        if safety_direction != -1:
            if old_blast_timer != -1: # unsafe previous position, in a blast zone
                if DIRECTIONS[self_action] == safety_direction:
                    events.append(MOVED_TOWARD_SAFETY)
                else:
                    events.append(MOVED_NOT_TOWARD_SAFETY)

    # Make it so waiting next to a bomb blast to avoid it does not incur a penalty
    if e.WAITED in events and any(t > 0 for t in old_state_features[14:18]) and old_state_features[9] == -1:
            events.remove(e.WAITED)

    # Reward agent if it chose to bomb several crates
    bombable_crates = old_state_features[8]

    if e.BOMB_DROPPED in events and bombable_crates > 0 and old_state_features[9] == -1:
        events.append(CHOSE_TO_BOMB_CRATES)
        # if bombable_crates > 3:
        #     events.append(BOMBED_MANY_CRATES)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(old_state_features, self_action, new_state_features, reward_from_events(self, events), new_game_state['round']))


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

    seen = collections.defaultdict(int)
    batch = []

    for r in randomization:
        if seen[r.episode] < 20:
            batch.append(r)
            seen[r.episode] += 1

    # sample = random.sample(self.transitions, self.batch_size)
    targets_store = []
    states_store = []
    errors_store = []
    actions_dict = {action: [] for action in ACTIONS}

    # 9.18 weighted sampling to fix agent getting stuck at bottom of board?
    for b in batch:
        pred = self.models[b.action].predict([b.state])[0]
        target = b.reward

        if b.next_state is None:
            td_error = abs(target - pred)
        else:
            state_actions = []
            for action in ACTIONS:
                state_actions.append(self.target_models[action].predict([b.next_state])) # Best action in that state, target model only for bootstrap

            target = b.reward + self.gamma * max(state_actions)[0]
            model_prediction = pred
            td_error = np.abs(target - model_prediction)

        actions_dict[b.action].append((b.state, max(-20, min(target, 20)), td_error)) # clamp target to prevent insane readings
        states_store.append(b.state)
        errors_store.append(td_error)

    rand_sampling_mix = 0.7
    for action in actions_dict:
        if not actions_dict[action]: continue

        combined = actions_dict[action]
        sorted_combined = sorted(combined, key=lambda x: x[2], reverse=True)
        highest_errors = sorted_combined[:int(self.batch_size * rand_sampling_mix)]
        sample_errors = random.sample(sorted_combined[len(highest_errors):],
                                      int(min(self.batch_size * (1 - rand_sampling_mix), len(sorted_combined) - len(highest_errors))))

        states_errors, targets_errors, _ = zip(*(highest_errors + sample_errors))
        self.models[action].fit(states_errors, targets_errors)

    if last_game_state['round'] % self.sync == 0:
        self.target_models = copy.deepcopy(self.models)

    # Store the model
    with open("q-learning-model.pt", "wb") as file:
        pickle.dump(self.models, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        # e.SURVIVED_ROUND: 0.1,
        e.WAITED: -2,
        e.INVALID_ACTION: -1,

        # e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,

        # e.COIN_COLLECTED: 5,
        # MOVED_TOWARD_COIN: 0.5,
        # MOVED_NOT_TOWARD_COIN: -0.2,

        e.CRATE_DESTROYED: 1,
        CHOSE_TO_BOMB_CRATES: 2,
        # BOMBED_MANY_CRATES: 1,

        STANDING_IN_BLAST: -2,
        ESCAPED_BLAST: 0.5,

        MOVED_INTO_BLAST: -1,
        MOVED_NOT_TOWARD_SAFETY: -2,
        MOVED_TOWARD_SAFETY: 2,
    }

    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum
