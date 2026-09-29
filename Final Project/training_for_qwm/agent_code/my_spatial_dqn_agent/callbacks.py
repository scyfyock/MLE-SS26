"""
callbacks.py -- Inference and action selection for my_spatial_dqn_agent.
"""

import json
import logging
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

from .model import (
    ACTIONS,
    ACTION_TO_IDX,
    DIRS,
    N_ACTIONS,
    F_IN_DANGER,
    F_SAFE_MOVE,
    F_SAFE_WAIT,
    F_TRAPS_SELF,
    F_COIN_DIR,
    F_COIN_PROX,
    F_CRATE_DIR,
    F_BOMBS_LEFT,
    F_BOMB_UTIL,
    F_OPP_DIR,
    F_OPP_PROX,
    agent_path,
    load_model,
    new_model,
    state_to_features,
    state_to_spatial_tensor,
    valid_action_mask,
)

DEFAULT_CONFIG = {
    'model_file': 'my-saved-model-spatial-dqn.pt',
    'lr': 1e-4,
    'gamma': 0.95,
    'tau': 0.005,
    'batch_size': 64,
    'buffer_size': 25000,
    'explore_unsafe': 0.1,
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    path = agent_path('config.json')
    if os.path.isfile(path):
        with open(path) as fh:
            cfg.update(json.load(fh))
    return cfg


def setup(self):
    """Load the trained model (play) or load/create it (training)."""
    self.logger = getattr(self, 'logger', None)
    if self.logger is None:
        self.logger = logging.getLogger('my_spatial_dqn_agent')
        self.logger.setLevel(logging.INFO)

    self.cfg = load_config()
    fname = (
        os.environ.get('MY_SPATIAL_DQN_MODEL_FILE')
        or os.environ.get('MY_DQN_MODEL_FILE')
        or (os.environ.get('MY_WM_MODEL_FILE') if 'qwm' not in str(os.environ.get('MY_WM_MODEL_FILE', '')) else None)
        or self.cfg['model_file']
    )
    run_dir = os.environ.get('MY_WM_RUN_DIR')
    if os.path.isfile(fname):
        load_path = os.path.abspath(fname)
        self.model_file = load_path
    elif run_dir:
        run_model_file = os.path.join(run_dir, fname)
        load_path = run_model_file if os.path.isfile(run_model_file) else agent_path(fname)
        if self.train:
            os.makedirs(run_dir, exist_ok=True)
            self.model_file = run_model_file
        else:
            self.model_file = load_path
    else:
        self.model_file = agent_path(fname)
        load_path = self.model_file

    self.coordinate_history = deque(maxlen=40)
    self._consecutive_stationary_steps = 0
    self._steps_without_progress = 0
    self._last_score = 0
    self._last_crates = None
    self.feature_cache = {}
    self._think_times = []
    self._last_round = None

    try:
        import parallel_train
        if parallel_train.ACTIVE_SHARED_MODEL is not None:
            self.model = parallel_train.ACTIVE_SHARED_MODEL
            self.grad_queue = parallel_train.ACTIVE_GRAD_QUEUE
            self.stats_queue = parallel_train.ACTIVE_STATS_QUEUE
            self.worker_id = parallel_train.ACTIVE_WORKER_ID
    except Exception:
        pass

    if not hasattr(self, 'model'):
        lr = float(os.environ.get('MY_WM_LR', self.cfg['lr']))
        if os.path.isfile(load_path):
            self.model = load_model(load_path, need_training=self.train)
            self.logger.info(f"Loaded spatial DQN model from {os.path.basename(load_path)}")
        else:
            self.model = new_model(lr=lr, gamma=self.cfg['gamma'], tau=self.cfg['tau'])
            self.logger.info(f"Created fresh spatial DQN model (lr={lr}, device={self.model.device})")

    if self.train and os.environ.get('MY_WM_LR'):
        self.model.set_lr(float(os.environ['MY_WM_LR']))

    if not hasattr(self, 'epsilon'):
        self.epsilon = 0.0
    self.explore_unsafe = float(os.environ.get('MY_WM_EXPLORE_UNSAFE', self.cfg['explore_unsafe']))


def detect_cycle(sequence, max_k=12):
    """
    Checks if a coordinate sequence ends with a repeating cycle of period k for any k in [2, max_k].
    Returns (is_cycle: bool, period: int, repetitions: int).
    """
    n = len(sequence)
    for k in range(2, min(max_k + 1, n // 2 + 1)):
        if sequence[-k:] == sequence[-2 * k : -k]:
            reps = 2
            while (reps + 1) * k <= n and sequence[-(reps + 1) * k : -reps * k] == sequence[-k:]:
                reps += 1
            return True, k, reps
    return False, 0, 0


def _is_safe_action(features, a_idx):
    """World-model verdict for one action (used to bias exploration)."""
    if a_idx < 4:
        return features[F_SAFE_MOVE + a_idx] > 0.5
    if ACTIONS[a_idx] == 'WAIT':
        return features[F_SAFE_WAIT] > 0.5
    return features[F_TRAPS_SELF] < 0.5 and features[F_SAFE_WAIT] > 0.5   # BOMB


def act(self, game_state: dict) -> str:
    t0 = time.perf_counter()
    if game_state is None:
        return 'WAIT'

    if game_state['step'] == 1 or game_state['round'] != self._last_round:
        self.coordinate_history.clear()
        self._last_round = game_state['round']
        self._consecutive_stationary_steps = 0
        self._steps_without_progress = 0
        self._last_score = game_state['self'][1]
        self._last_crates = int(np.count_nonzero(game_state['field'] == 1))

    spatial = state_to_spatial_tensor(game_state, self.coordinate_history)
    features = state_to_features(game_state, self.coordinate_history)
    mask = valid_action_mask(features)
    x, y = game_state['self'][3]

    if self.coordinate_history and self.coordinate_history[-1] == (x, y):
        self._consecutive_stationary_steps += 1
    else:
        self._consecutive_stationary_steps = 0

    current_score = game_state['self'][1]
    current_crates = int(np.count_nonzero(game_state['field'] == 1))
    has_targets = (
        current_crates > 0 or
        len(game_state.get('coins', [])) > 0 or
        len(game_state.get('others', [])) > 0
    )

    if not has_targets:
        self._steps_without_progress = 0
        self._consecutive_stationary_steps = 0
    elif current_score > self._last_score or (self._last_crates is not None and current_crates < self._last_crates):
        self._steps_without_progress = 0
        self.coordinate_history.clear()
    else:
        self._steps_without_progress += 1
    self._last_score = current_score
    self._last_crates = current_crates

    in_danger = features[F_IN_DANGER] > 0.5
    safe_moves = (features[F_SAFE_MOVE:F_SAFE_MOVE + 4] > 0.5) & mask[:4]
    no_active_bombs = len(game_state.get('bombs', [])) == 0

    is_self_play = (os.environ.get('MY_WM_SELF_PLAY') == '1')
    if not is_self_play and getattr(self, 'train', False) and getattr(self, 'epsilon', 0.0) > 0 and random.random() < self.epsilon:
        valid = [i for i in range(len(ACTIONS)) if mask[i]]
        if random.random() >= self.explore_unsafe:
            safe = [i for i in valid if _is_safe_action(features, i)]
            if safe:
                valid = safe

        # Filter out WAIT when safe with no bombs or when stagnating (only when targets exist)
        if has_targets and (no_active_bombs or self._consecutive_stationary_steps >= 1 or self._steps_without_progress >= 1) and len(valid) > 1:
            valid_no_wait = [i for i in valid if ACTIONS[i] != 'WAIT']
            if valid_no_wait:
                valid = valid_no_wait

        # Avoid loop continuation and direct reversals during random exploration
        if has_targets and not in_danger and self.coordinate_history and len(valid) > 1:
            prev_tile = self.coordinate_history[-1]
            non_looping = []
            for i in valid:
                if i < 4:
                    dx, dy = DIRS[i][1]
                    cand_pos = (x + dx, y + dy)
                    cand_traj = list(self.coordinate_history) + [(x, y), cand_pos]
                    is_cycle, _, _ = detect_cycle(cand_traj, max_k=12)
                    if not is_cycle and cand_pos != prev_tile:
                        non_looping.append(i)
                else:
                    non_looping.append(i)
            if non_looping:
                valid = non_looping

        action = ACTIONS[random.choice(valid)] if valid else 'WAIT'
    else:
        # Full-board Spatial Dueling DQN Q-values
        q = np.asarray(self.model.q_values(spatial, features), dtype=np.float64)
        q[~mask] = -np.inf

        # 1. Standing around / Idle WAIT penalty (only when there are targets to pursue):
        wait_idx = ACTION_TO_IDX['WAIT']
        if has_targets and mask[wait_idx] and not in_danger and safe_moves.any():
            if no_active_bombs:
                safe_move_min = float(np.min(q[:4][safe_moves]))
                q[wait_idx] = min(q[wait_idx], safe_move_min - 4.5)
            if self._consecutive_stationary_steps >= 1:
                q[wait_idx] -= self._consecutive_stationary_steps * 3.0
            wait_visits = sum(1 for p in self.coordinate_history if p == (x, y))
            if wait_visits >= 1:
                q[wait_idx] -= wait_visits * 2.0
            is_wait_cycle, _, _ = detect_cycle(list(self.coordinate_history) + [(x, y), (x, y)], max_k=12)
            if is_wait_cycle:
                q[wait_idx] -= 8.0

        # 2. Arbitrary-length loop & corridor patrol penalties (only when there are targets to pursue):
        if has_targets and not in_danger and safe_moves.any() and self.coordinate_history:
            prev_tile = self.coordinate_history[-1]
            stagnation_factor = 1.0 + 0.15 * self._steps_without_progress
            for i, (a_name, (dx, dy)) in enumerate(DIRS):
                if not mask[i] or features[F_SAFE_MOVE + i] <= 0.5:
                    continue
                nx, ny = x + dx, y + dy
                cand_traj = list(self.coordinate_history) + [(x, y), (nx, ny)]

                # A. Detect repeating cycles of any period k in [2, 12]
                is_cycle, cycle_k, reps = detect_cycle(cand_traj, max_k=12)
                if is_cycle:
                    q[i] -= (7.0 + 3.0 * (reps - 2)) * stagnation_factor

                # B. Direct reversal (stepping straight back to previous tile)
                if (nx, ny) == prev_tile:
                    q[i] -= 4.5 * stagnation_factor

                # C. Heatmap revisit count (repels pacing across corridors or clusters)
                visits = sum(1 for p in self.coordinate_history if p == (nx, ny))
                if visits >= 1:
                    q[i] -= (visits * 2.0 + max(0, visits - 2) * 2.0) * stagnation_factor

        # 3. Non-progress penalty & target motivation boost (only when there are targets to pursue):
        if has_targets and self._steps_without_progress >= 1:
            q[wait_idx] -= min(25.0, 2.0 + 0.8 * self._steps_without_progress)

            coin_boost = min(5.0, 1.0 + 0.25 * self._steps_without_progress)
            crate_boost = min(4.0, 0.8 + 0.2 * self._steps_without_progress)
            bomb_boost = min(5.0, 1.0 + 0.25 * self._steps_without_progress)

            if features[F_COIN_PROX] > 0:
                for i in range(4):
                    if features[F_COIN_DIR + i] > 0.5 and mask[i] and features[F_SAFE_MOVE + i] > 0.5:
                        q[i] += coin_boost
            elif features[F_BOMBS_LEFT] > 0.5:
                for i in range(4):
                    if features[F_CRATE_DIR + i] > 0.5 and mask[i] and features[F_SAFE_MOVE + i] > 0.5:
                        q[i] += crate_boost
                if features[F_BOMB_UTIL] > 0.0 and mask[ACTION_TO_IDX['BOMB']] and features[F_TRAPS_SELF] < 0.5:
                    q[ACTION_TO_IDX['BOMB']] += bomb_boost
            elif features[F_OPP_PROX] > 0:
                opp_boost = min(5.0, 1.0 + 0.25 * self._steps_without_progress)
                for i in range(4):
                    if features[F_OPP_DIR + i] > 0.5 and mask[i] and features[F_SAFE_MOVE + i] > 0.5:
                        q[i] += opp_boost

        best = int(np.argmax(q)) if np.isfinite(q).any() else ACTIONS.index('WAIT')
        action = ACTIONS[best]

    self.coordinate_history.append((x, y))
    self.last_spatial = spatial
    self.last_features = features
    self.last_mask = mask

    dt = time.perf_counter() - t0
    self._think_times.append(dt)
    if len(self._think_times) >= 200:
        arr = np.asarray(self._think_times) * 1000.0
        self.logger.info(f"Spatial DQN act() think time over last {len(arr)} steps: mean {arr.mean():.2f} ms, max {arr.max():.2f} ms")
        self._think_times = []
    self.logger.debug(f"step {game_state['step']}: pos=({x},{y}) action={action} ({dt * 1000:.1f} ms)")
    return action
