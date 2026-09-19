"""Small CPU-friendly Double-DQN for the compact symbolic policy state."""

from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
from sklearn.neural_network import MLPRegressor

from .world_model import N_ACTIONS, N_STATE_FEATURES, encode_state


@dataclass(frozen=True)
class QExperience:
    state: tuple
    action_index: int
    reward: float
    next_state: tuple | None
    done: bool
    next_valid_actions: tuple
    bootstrap_discount: float | None = None


class NeuralQFunction:
    """Incremental Double-DQN with replay and a periodically synced target."""

    def __init__(
        self,
        replay_capacity=30_000,
        batch_size=64,
        min_samples=128,
        hidden_size=64,
        learning_rate=0.0005,
        gamma=0.95,
        target_update_interval=250,
        target_clip=30.0,
        n_step=5,
        seed=11,
    ):
        self.replay_capacity = int(replay_capacity)
        self.batch_size = int(batch_size)
        self.min_samples = int(min_samples)
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.target_clip = float(target_clip)
        self.n_step = int(n_step)
        self.seed = int(seed)
        self.replay = []
        self.replay_position = 0
        self.rare_replay_capacity = 2_000
        self.positive_replay = []
        self.negative_replay = []
        self.pending = []
        self.total_observations = 0
        self.training_steps = 0
        self.last_loss = None
        self.rng = random.Random(seed)
        self.online_model = MLPRegressor(
            hidden_layer_sizes=(hidden_size, hidden_size),
            activation="tanh",
            solver="adam",
            learning_rate_init=learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=seed,
        )

        # ``partial_fit`` establishes sklearn's internal optimizer state. The
        # output layer is then zeroed so an untrained policy has no directional
        # bias and action ties remain random.
        dummy_x = np.zeros((1, N_STATE_FEATURES), dtype=np.float32)
        dummy_y = np.zeros((1, N_ACTIONS), dtype=np.float32)
        self.online_model.partial_fit(dummy_x, dummy_y)
        self.online_model.coefs_[-1].fill(0.0)
        self.online_model.intercepts_[-1].fill(0.0)
        self.target_model = deepcopy(self.online_model)

    def __len__(self):
        return len(self.replay)

    def ensure_rare_replay_state(self):
        """Upgrade older checkpoints and retain rare outcomes across stages."""

        if not hasattr(self, "rare_replay_capacity"):
            self.rare_replay_capacity = 2_000
        if not hasattr(self, "positive_replay"):
            self.positive_replay = [
                item for item in self.replay if item.reward >= 10.0
            ][-self.rare_replay_capacity:]
        if not hasattr(self, "negative_replay"):
            self.negative_replay = [
                item for item in self.replay if item.reward <= -10.0
            ][-self.rare_replay_capacity:]

    @staticmethod
    def _encode(states):
        return np.stack([encode_state(state) for state in states])

    def predict(self, state, target=False):
        model = self.target_model if target else self.online_model
        return np.asarray(
            model.predict(encode_state(state).reshape(1, -1))[0],
            dtype=np.float32,
        )

    def _append(self, experience):
        self.ensure_rare_replay_state()
        if len(self.replay) < self.replay_capacity:
            self.replay.append(experience)
        else:
            self.replay[self.replay_position] = experience
        self.replay_position = (
            self.replay_position + 1
        ) % self.replay_capacity
        if experience.reward >= 10.0:
            self.positive_replay.append(experience)
            if len(self.positive_replay) > self.rare_replay_capacity:
                del self.positive_replay[0]
        elif experience.reward <= -10.0:
            self.negative_replay.append(experience)
            if len(self.negative_replay) > self.rare_replay_capacity:
                del self.negative_replay[0]
        self.total_observations += 1

    def observe(
        self,
        state,
        action_index,
        reward,
        next_state,
        done,
        next_valid_actions=(),
    ):
        one_step = QExperience(
            state=tuple(state),
            action_index=int(action_index),
            reward=float(reward),
            next_state=(tuple(next_state) if next_state is not None else None),
            done=bool(done),
            next_valid_actions=tuple(
                int(action) for action in (next_valid_actions or ())
            ),
        )
        self.pending.append(one_step)
        if done:
            while self.pending:
                self._append(self._n_step_experience(len(self.pending)))
                del self.pending[0]
        elif len(self.pending) >= self.n_step:
            self._append(self._n_step_experience(self.n_step))
            del self.pending[0]
        return self.train_replay_step()

    def _n_step_experience(self, steps):
        """Aggregate delayed outcomes into a direct n-step training target."""

        sequence = self.pending[:steps]
        first = sequence[0]
        last = sequence[-1]
        discounted_reward = sum(
            (self.gamma ** offset) * item.reward
            for offset, item in enumerate(sequence)
        )
        return QExperience(
            state=first.state,
            action_index=first.action_index,
            reward=discounted_reward,
            next_state=last.next_state,
            done=last.done,
            next_valid_actions=last.next_valid_actions,
            bootstrap_discount=self.gamma ** len(sequence),
        )

    def _targets(self, batch):
        states = self._encode([item.state for item in batch])
        predictions = self.online_model.predict(states)
        targets = np.asarray(predictions, dtype=np.float32).copy()
        td_errors = []

        nonterminal = [
            (index, item)
            for index, item in enumerate(batch)
            if not item.done
            and item.next_state is not None
            and item.next_valid_actions
        ]
        online_next = {}
        target_next = {}
        if nonterminal:
            encoded_next = self._encode(
                [item.next_state for _index, item in nonterminal]
            )
            online_values = self.online_model.predict(encoded_next)
            target_values = self.target_model.predict(encoded_next)
            for row, (index, _item) in enumerate(nonterminal):
                online_next[index] = online_values[row]
                target_next[index] = target_values[row]

        for index, item in enumerate(batch):
            target = item.reward
            if index in online_next:
                # Double-DQN: online network selects, target network evaluates.
                next_action = max(
                    item.next_valid_actions,
                    key=lambda action: online_next[index][action],
                )
                discount = (
                    self.gamma
                    if item.bootstrap_discount is None
                    else item.bootstrap_discount
                )
                target += discount * target_next[index][next_action]
            target = float(np.clip(target, -self.target_clip, self.target_clip))
            old_value = float(predictions[index, item.action_index])
            targets[index, item.action_index] = target
            td_errors.append(abs(target - old_value))

        return states, targets, td_errors

    def _fit(self, batch):
        if not batch:
            return []
        states, targets, td_errors = self._targets(batch)
        self.online_model.partial_fit(states, targets)
        self.training_steps += 1
        self.last_loss = float(self.online_model.loss_)
        if self.training_steps % self.target_update_interval == 0:
            self.sync_target()
        return td_errors

    def train_replay_step(self):
        if len(self.replay) < self.min_samples:
            return []
        sample_size = min(self.batch_size, len(self.replay))
        batch = self._sample_replay_batch(sample_size)
        return self._fit(batch)

    def _sample_replay_batch(self, sample_size):
        """Keep bombs and rare outcome tails visible across curricula."""

        self.ensure_rare_replay_state()

        bomb_items = [
            item for item in self.replay if item.action_index == N_ACTIONS - 1
        ]
        bomb_count = sample_size // 4 if bomb_items else 0
        positive_count = sample_size // 8 if self.positive_replay else 0
        negative_count = sample_size // 8 if self.negative_replay else 0
        ordinary_count = (
            sample_size - bomb_count - positive_count - negative_count
        )
        ordinary = self.rng.sample(
            self.replay,
            min(ordinary_count, len(self.replay)),
        )
        bombs = (
            self.rng.choices(bomb_items, k=bomb_count)
            if bomb_count
            else []
        )
        positives = (
            self.rng.choices(self.positive_replay, k=positive_count)
            if positive_count
            else []
        )
        negatives = (
            self.rng.choices(self.negative_replay, k=negative_count)
            if negative_count
            else []
        )
        batch = ordinary + bombs + positives + negatives
        while len(batch) < sample_size:
            batch.append(self.rng.choice(self.replay))
        self.rng.shuffle(batch)
        return batch

    def train_imagined(self, experiences):
        """Learn from model transitions without polluting real replay."""

        batch = [
            QExperience(
                state=tuple(item.state),
                action_index=int(item.action_index),
                reward=float(item.reward),
                next_state=(
                    tuple(item.next_state)
                    if item.next_state is not None
                    else None
                ),
                done=bool(item.done),
                next_valid_actions=tuple(item.next_valid_actions or ()),
                bootstrap_discount=(
                    self.gamma
                    if item.bootstrap_discount is None
                    else float(item.bootstrap_discount)
                ),
            )
            for item in experiences
        ]
        return self._fit(batch)

    def sync_target(self):
        self.target_model = deepcopy(self.online_model)
