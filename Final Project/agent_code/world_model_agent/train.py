import pickle
from typing import List
import random
import os

import events as e

from .callbacks import (
    ACTIONS,
    ACTION_TO_INDEX,
    MODEL_FILE,
    MODEL_SCHEMA_VERSION,
    DEATH_RISK_PENALTY,
    bomb_would_hit_target,
    bomb_outcome_adjustment,
    can_escape_after_placing_bomb,
    death_risk_cost,
    get_q_values,
    get_valid_actions,
    position_is_in_danger,
    predicted_action_risk,
    safe_action_mask,
    state_to_key,
    survivable_escape_actions,
)
from .q_network import NeuralQFunction, QExperience
from .world_model import (
    LearnedWorldModel,
    REPLAY_BOMB,
    REPLAY_DANGER,
    REPLAY_NORMAL,
    REPLAY_TERMINAL,
    game_state_to_world_state,
)


# Custom events
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_NOT_TOWARD_COIN = "MOVED_NOT_TOWARD_COIN"
USEFUL_BOMB = "USEFUL_BOMB"
USELESS_BOMB = "USELESS_BOMB"
MOVED_OUT_OF_DANGER = "MOVED_OUT_OF_DANGER"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"
STAYED_IN_DANGER = "STAYED_IN_DANGER"
MOVED_TOWARD_SAFETY = "MOVED_TOWARD_SAFETY"
MOVED_INTO_IMMEDIATE_DANGER = "MOVED_INTO_IMMEDIATE_DANGER"
BOMB_WITHOUT_ESCAPE = "BOMB_WITHOUT_ESCAPE"

MOVEMENT_EVENTS = {
    e.MOVED_LEFT,
    e.MOVED_RIGHT,
    e.MOVED_UP,
    e.MOVED_DOWN,
}


# Hyperparameters
DISCOUNT_FACTOR = 0.95

INITIAL_EPSILON = 0.2
MIN_EPSILON = 0.02
EPSILON_DECAY = 0.995

PLANNING_ROLLOUTS = int(
    os.environ.get(
        "DYNA_PLANNING_ROLLOUTS",
        os.environ.get("DYNA_PLANNING_STEPS", "1"),
    )
)
# Start with one model-generated transition per rollout. The evaluator can be
# used to justify increasing this via DYNA_PLANNING_HORIZON once open-loop
# position and terminal predictions remain reliable beyond the first step.
PLANNING_HORIZON = int(os.environ.get("DYNA_PLANNING_HORIZON", "1"))
WORLD_MODEL_REPLAY_SIZE = int(
    os.environ.get("DYNA_WORLD_MODEL_REPLAY_SIZE", "20000")
)
WORLD_MODEL_BATCH_SIZE = int(
    os.environ.get("DYNA_WORLD_MODEL_BATCH_SIZE", "32")
)
WORLD_MODEL_WARMUP = int(
    os.environ.get("DYNA_WORLD_MODEL_WARMUP", "256")
)
WORLD_MODEL_TRAIN_EVERY = int(
    os.environ.get("DYNA_WORLD_MODEL_TRAIN_EVERY", "4")
)
WORLD_MODEL_DONE_THRESHOLD = float(
    os.environ.get("DYNA_WORLD_MODEL_DONE_THRESHOLD", "0.5")
)
Q_REPLAY_SIZE = int(os.environ.get("DYNA_Q_REPLAY_SIZE", "30000"))
Q_BATCH_SIZE = int(os.environ.get("DYNA_Q_BATCH_SIZE", "64"))
Q_WARMUP = int(os.environ.get("DYNA_Q_WARMUP", "128"))
Q_HIDDEN_SIZE = int(os.environ.get("DYNA_Q_HIDDEN_SIZE", "64"))
Q_LEARNING_RATE = float(os.environ.get("DYNA_Q_LEARNING_RATE", "0.0005"))
Q_TARGET_UPDATE = int(os.environ.get("DYNA_Q_TARGET_UPDATE", "250"))
SAVE_EVERY_ROUNDS = max(1, int(os.environ.get("DYNA_SAVE_EVERY", "5")))
FREEZE_Q_NETWORK = os.environ.get(
    "DYNA_FREEZE_Q_NETWORK",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}

def setup_training(self):
    """Initialize parameters used only during training."""

    self.gamma = DISCOUNT_FACTOR
    self.planning_rollouts = PLANNING_ROLLOUTS
    self.planning_horizon = PLANNING_HORIZON
    self.world_model_done_threshold = WORLD_MODEL_DONE_THRESHOLD
    self.world_model_train_every = WORLD_MODEL_TRAIN_EVERY
    self.death_risk_penalty = DEATH_RISK_PENALTY
    self.planning_rng = random.Random(1)  # Random generator for planning steps

    if not isinstance(getattr(self, "q_network", None), NeuralQFunction):
        self.q_network = NeuralQFunction(
            replay_capacity=Q_REPLAY_SIZE,
            batch_size=Q_BATCH_SIZE,
            min_samples=Q_WARMUP,
            hidden_size=Q_HIDDEN_SIZE,
            learning_rate=Q_LEARNING_RATE,
            gamma=self.gamma,
            target_update_interval=Q_TARGET_UPDATE,
        )
    self.q_network.ensure_rare_replay_state()

    if not isinstance(
        getattr(self, "world_model", None),
        LearnedWorldModel,
    ):
        self.world_model = LearnedWorldModel(
            replay_capacity=WORLD_MODEL_REPLAY_SIZE,
            batch_size=WORLD_MODEL_BATCH_SIZE,
            min_samples=WORLD_MODEL_WARMUP,
        )
    self.world_model.ensure_bomb_outcome_state()
    self.world_model.ensure_opponent_model_state()

    # setup() may already have loaded epsilon from a saved model.
    if not hasattr(self, "epsilon"):
        self.epsilon = INITIAL_EPSILON

    self.logger.info(
        f"Training setup: Double-DQN lr={Q_LEARNING_RATE}, "
        f"gamma={self.gamma}, epsilon={self.epsilon}, "
        f"q_batch={self.q_network.batch_size}, "
        f"q_warmup={self.q_network.min_samples}, "
        f"target_update={self.q_network.target_update_interval}, "
        f"planning_rollouts={self.planning_rollouts}, "
        f"planning_horizon={self.planning_horizon}, "
        f"world_model_warmup={self.world_model.min_samples}, "
        f"world_model_train_every={self.world_model_train_every}, "
        f"death_risk_penalty={self.death_risk_penalty}, "
        f"freeze_q_network={FREEZE_Q_NETWORK}"
    )


def add_custom_events(
    old_game_state: dict,
    self_action: str,
    events: List[str],
    new_game_state: dict = None,
):
    """Add coin, bomb-utility, and bomb-safety training signals."""

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
        if not can_escape_after_placing_bomb(old_game_state):
            events.append(BOMB_WITHOUT_ESCAPE)

    action_was_executed = (
        bool(MOVEMENT_EVENTS.intersection(events))
        or e.WAITED in events
    )
    if not action_was_executed:
        return

    old_in_danger = position_is_in_danger(old_game_state)
    escape_actions = survivable_escape_actions(old_game_state)
    action_safety = safe_action_mask(old_game_state)

    if not action_safety[ACTION_TO_INDEX[self_action]]:
        events.append(MOVED_INTO_IMMEDIATE_DANGER)

    if old_in_danger and self_action not in escape_actions:
        events.append(STAYED_IN_DANGER)

    if new_game_state is None:
        return

    new_in_danger = position_is_in_danger(new_game_state)

    if old_in_danger and not new_in_danger:
        events.append(MOVED_OUT_OF_DANGER)
    elif not old_in_danger and new_in_danger:
        events.append(MOVED_INTO_DANGER)
    elif old_in_danger and new_in_danger:
        if self_action in escape_actions and self_action != "WAIT":
            events.append(MOVED_TOWARD_SAFETY)


def update_q(
    self,
    state,
    action: str,
    reward: float,
    next_state,
    done: bool,
    next_valid_actions=None,
    store_in_replay=True,
):
    """Store and train one real Double-DQN transition."""

    if FREEZE_Q_NETWORK:
        return 0.0

    next_indices = tuple(
        ACTION_TO_INDEX[next_action]
        for next_action in (next_valid_actions or ())
    )
    if store_in_replay:
        errors = self.q_network.observe(
            state=state,
            action_index=ACTION_TO_INDEX[action],
            reward=reward,
            next_state=next_state,
            done=done,
            next_valid_actions=next_indices,
        )
    else:
        errors = self.q_network.train_imagined(
            [
                QExperience(
                    state=tuple(state),
                    action_index=ACTION_TO_INDEX[action],
                    reward=float(reward),
                    next_state=(
                        tuple(next_state)
                        if next_state is not None
                        else None
                    ),
                    done=bool(done),
                    next_valid_actions=next_indices,
                )
            ]
        )
    return float(sum(errors) / len(errors)) if errors else 0.0

def store_transition(
    self,
    state,
    action,
    reward,
    next_state,
    done,
    game_state,
    next_game_state,
    valid_actions,
    next_valid_actions,
    death=False,
    suicide=False,
):
    """Store a real transition as supervised world-model experience."""

    if done:
        replay_category = REPLAY_TERMINAL
    elif (
        position_is_in_danger(game_state)
        or (
            next_game_state is not None
            and position_is_in_danger(next_game_state)
        )
    ):
        replay_category = REPLAY_DANGER
    elif (
        action == "BOMB"
        or bool(game_state.get("bombs"))
        or (
            next_game_state is not None
            and bool(next_game_state.get("bombs"))
        )
    ):
        replay_category = REPLAY_BOMB
    else:
        replay_category = REPLAY_NORMAL

    self.world_model.observe(
        state=state,
        world_state=game_state_to_world_state(game_state),
        action_index=ACTION_TO_INDEX[action],
        reward=reward,
        next_state=next_state,
        next_world_state=game_state_to_world_state(next_game_state),
        done=done,
        valid_actions=(
            ACTION_TO_INDEX[valid_action]
            for valid_action in valid_actions
        ),
        next_valid_actions=(
            ACTION_TO_INDEX[next_action]
            for next_action in (next_valid_actions or [])
        ),
        replay_category=replay_category,
        death=death,
        suicide=suicide,
    )


def record_bomb_outcome_events(self, state, action, events, done=False):
    """Forward accepted bomb placements and their delayed outcomes."""

    self.world_model.observe_bomb_events(
        state=state,
        bomb_dropped=(action == "BOMB" and e.BOMB_DROPPED in events),
        crate_destroyed=e.CRATE_DESTROYED in events,
        killed_opponent=e.KILLED_OPPONENT in events,
        death=(e.GOT_KILLED in events or e.KILLED_SELF in events),
        exploded=e.BOMB_EXPLODED in events,
        done=done,
    )

def perform_planning_updates(self):
    """Train the Q-network on recursively imagined model rollouts."""

    if FREEZE_Q_NETWORK or not self.world_model.is_ready:
        return []

    imagined_experiences = []
    for _ in range(self.planning_rollouts):
        start = self.world_model.sample_start(self.planning_rng)
        state = start.state
        world_state = start.world_state
        valid_action_indices = list(start.valid_actions)
        imagined_trajectory = []

        for _depth in range(self.planning_horizon):
            if not valid_action_indices:
                break

            q_values = get_q_values(self, state)
            bomb_adjustment, bomb_outcome = (
                bomb_outcome_adjustment(self.world_model, state)
                if ACTION_TO_INDEX["BOMB"] in valid_action_indices
                else (0.0, None)
            )
            action_death_risks = {
                action_index: predicted_action_risk(
                    self.world_model,
                    state,
                    action_index,
                )
                for action_index in valid_action_indices
            }
            bomb_index = ACTION_TO_INDEX["BOMB"]
            if bomb_outcome is not None:
                action_death_risks[bomb_index] = max(
                    action_death_risks[bomb_index],
                    1.0 - bomb_outcome.survived_probability,
                )
            risk_adjusted_values = {
                action_index: (
                    q_values[action_index]
                    - death_risk_cost(action_death_risks[action_index])
                    + (
                        bomb_adjustment
                        if action_index == bomb_index
                        else 0.0
                    )
                )
                for action_index in valid_action_indices
            }
            best_value = max(
                risk_adjusted_values[action_index]
                for action_index in valid_action_indices
            )
            best_actions = [
                action_index
                for action_index in valid_action_indices
                if abs(
                    risk_adjusted_values[action_index] - best_value
                ) < 1e-8
            ]
            action_index = self.planning_rng.choice(best_actions)
            prediction = self.world_model.predict(
                world_state,
                state,
                action_index,
            )
            next_state = prediction.next_state
            done = (
                prediction.done_probability
                >= self.world_model_done_threshold
            )
            next_valid_action_indices = (
                []
                if done
                else list(prediction.next_valid_actions)
            )
            imagined_trajectory.append(
                QExperience(
                    state=tuple(state),
                    action_index=action_index,
                    reward=prediction.reward,
                    next_state=tuple(next_state),
                    done=done,
                    next_valid_actions=tuple(next_valid_action_indices),
                )
            )

            if done:
                break

            state = next_state
            world_state = prediction.next_world_state
            valid_action_indices = next_valid_action_indices

        imagined_experiences.extend(reversed(imagined_trajectory))

    return self.q_network.train_imagined(imagined_experiences)

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
        new_game_state,
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
    record_bomb_outcome_events(
        self,
        state,
        self_action,
        events,
        done=False,
    )

    next_valid_actions = get_valid_actions(new_game_state)
    valid_actions = get_valid_actions(old_game_state)

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
        game_state=old_game_state,
        next_game_state=new_game_state,
        valid_actions=valid_actions,
        next_valid_actions=next_valid_actions,
    )

    train_dynamics = (
        self.world_model.total_observations
        % self.world_model_train_every
        == 0
    )
    world_model_loss = self.world_model.train_step(
        train_dynamics=train_dynamics
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
        f"world_model_loss={world_model_loss}, "
        f"planning_error={mean_planning_error:.4f}"
    )


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """Learn from the final transition and save both neural models."""

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
        record_bomb_outcome_events(
            self,
            state,
            last_action,
            events,
            done=True,
        )

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
            game_state=last_game_state,
            next_game_state=None,
            valid_actions=get_valid_actions(last_game_state),
            next_valid_actions=None,
            death=e.GOT_KILLED in events,
            suicide=e.KILLED_SELF in events,
        )

        train_dynamics = (
            self.world_model.total_observations
            % self.world_model_train_every
            == 0
        )
        world_model_loss = self.world_model.train_step(
            train_dynamics=train_dynamics
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
            f"world_model_loss={world_model_loss}, "
            f"planning_error={mean_planning_error:.4f}"
        )

    self.previous_position = None
    self.current_state_key = None
    # Normally the terminal transition flushes all pending n-step returns.
    # This also prevents accidental cross-round returns after an abnormal end.
    self.q_network.pending.clear()

    self.epsilon = max(
        MIN_EPSILON,
        self.epsilon * EPSILON_DECAY,
    )

    round_number = (
        int(last_game_state.get("round", 0))
        if last_game_state is not None
        else 0
    )
    if round_number == 0 or round_number % SAVE_EVERY_ROUNDS == 0:
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
        e.KILLED_OPPONENT: 20.0,
        e.KILLED_SELF: -25.0,
        e.GOT_KILLED: -15.0,
        e.SURVIVED_ROUND: 3.0,
        e.WAITED: -0.2,
        e.INVALID_ACTION: -1.0,
        MOVED_TOWARD_COIN: 0.2,
        MOVED_NOT_TOWARD_COIN: -0.1,
        USEFUL_BOMB: 0.25,
        USELESS_BOMB: -0.5,
        MOVED_OUT_OF_DANGER: 1.0,
        MOVED_INTO_DANGER: -2.0,
        STAYED_IN_DANGER: -2.0,
        MOVED_TOWARD_SAFETY: 0.5,
        MOVED_INTO_IMMEDIATE_DANGER: -3.0,
        BOMB_WITHOUT_ESCAPE: -5.0,
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
        "q_network": self.q_network,
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
        f"Saved Double-DQN with {len(self.q_network)} real transitions "
        f"and {self.q_network.training_steps} training steps, "
        f"plus {len(self.world_model)} real world-model transitions, "
        f"with epsilon={self.epsilon:.4f}."
    )
