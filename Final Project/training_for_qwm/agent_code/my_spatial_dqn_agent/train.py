"""
train.py -- Training loop, reward shaping, experience replay, and checkpointing
for my_spatial_dqn_agent (Full-Board Spatial Dueling DQN).
"""

import csv
import json
import os
import random
import shutil
import time
from collections import deque
from typing import List, Optional

import numpy as np

import events as e
from .callbacks import detect_cycle
from .model import (
    ACTIONS,
    ACTION_TO_IDX,
    BOARD_H,
    BOARD_W,
    F_BOMB_UTIL,
    F_BOMBS_LEFT,
    F_COIN_PROX,
    F_IN_DANGER,
    F_OPP_PROX,
    F_SAFE_MOVE,
    F_SAFE_WAIT,
    F_TRAPS_SELF,
    INF,
    MOVE_ACTIONS,
    N_FEATURES,
    N_SPATIAL_CHANNELS,
    agent_path,
    augment_spatial_transition,
    bfs_direction,
    crate_adjacent_tiles,
    save_model,
    state_to_features,
    state_to_spatial_tensor,
)

# ----------------------------------------------------------------------------
# Rewards & Auxiliary Events
# ----------------------------------------------------------------------------
MOVED_TOWARD_COIN = 'MOVED_TOWARD_COIN'
MOVED_AWAY_COIN = 'MOVED_AWAY_COIN'
MOVED_TOWARD_CRATE = 'MOVED_TOWARD_CRATE'
MOVED_AWAY_CRATE = 'MOVED_AWAY_CRATE'
MOVED_TOWARD_OPPONENT = 'MOVED_TOWARD_OPPONENT'
MOVED_AWAY_OPPONENT = 'MOVED_AWAY_OPPONENT'
MOVED_TO_SAFETY = 'MOVED_TO_SAFETY'
STAYED_IN_DANGER = 'STAYED_IN_DANGER'
UNSAFE_ACTION = 'UNSAFE_ACTION'
USEFUL_BOMB = 'USEFUL_BOMB'
USELESS_BOMB = 'USELESS_BOMB'
SUICIDAL_BOMB = 'SUICIDAL_BOMB'
SPARSE_SUICIDE_PENALTY = 'SPARSE_SUICIDE_PENALTY'
LOOP_DETECTED = 'LOOP_DETECTED'
IDLE_WAIT = 'IDLE_WAIT'
STANDING_AROUND = 'STANDING_AROUND'
NO_PROGRESS = 'NO_PROGRESS'
STAGNATION = 'STAGNATION'

REWARD_TABLE = {
    # Environment events
    e.COIN_COLLECTED: 5.0,
    e.CRATE_DESTROYED: 0.3,
    e.COIN_FOUND: 0.3,
    e.KILLED_OPPONENT: 10.0,
    e.KILLED_SELF: -20.0,
    e.GOT_KILLED: -10.0,
    e.SURVIVED_ROUND: 5.0,
    e.INVALID_ACTION: -0.5,
    # Navigation shaping
    MOVED_TOWARD_COIN: 0.5,
    MOVED_AWAY_COIN: -0.6,
    MOVED_TOWARD_CRATE: 0.2,
    MOVED_AWAY_CRATE: -0.3,
    MOVED_TOWARD_OPPONENT: 0.4,
    MOVED_AWAY_OPPONENT: -0.5,
    # Survival & safety
    MOVED_TO_SAFETY: 1.0,
    STAYED_IN_DANGER: -1.0,
    UNSAFE_ACTION: -3.0,
    USEFUL_BOMB: 0.5,
    USELESS_BOMB: -1.5,
    SUICIDAL_BOMB: -6.0,
    SPARSE_SUICIDE_PENALTY: -5.0,
    # Idling & Anti-looping
    LOOP_DETECTED: -0.8,
    IDLE_WAIT: -0.2,
    STANDING_AROUND: -1.0,
    # Escalating non-progress penalties
    NO_PROGRESS: -0.2,
    STAGNATION: -0.4,
}

STATE_FILE = 'training_state-spatial-dqn.json'
STATS_FILE = 'training_stats-spatial-dqn.csv'
CSV_COLUMNS = [
    'round', 'stage', 'epsilon', 'score', 'coins', 'kills', 'crates', 'bombs', 'survived',
    'steps', 'invalid_acts', 'killed_self', 'got_killed', 'mean_reward', 'sum_reward',
    'mean_td_error', 'unsafe_actions', 'wall_time_s'
]


def _output_path(self, base):
    run_dir = os.environ.get('MY_WM_RUN_DIR')
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        return os.path.join(run_dir, base)
    return agent_path(base)


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, '') else default


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, '') else default


def is_better_round(cand: dict, best: Optional[dict]) -> bool:
    """
    Determines if candidate round is better than current_best based on:
    1. Filter first by highest reward (and score).
    2. Prefer rounds where the agent survived.
    3. If survived: tie-break by lowest step count (fastest completion).
    4. If neither survived (no agents with highest score survived):
       take the one with highest survival time (steps).
    """
    if best is None:
        return True

    c_score = float(cand.get('score', 0))
    b_score = float(best.get('score', 0))
    c_rew = round(float(cand.get('sum_reward', cand.get('reward', 0.0))), 2)
    b_rew = round(float(best.get('sum_reward', best.get('reward', 0.0))), 2)
    c_surv = int(cand.get('survived', 0)) == 1
    b_surv = int(best.get('survived', 0)) == 1
    c_steps = int(cand.get('steps', 0))
    b_steps = int(best.get('steps', 0))

    # 1. Both survived
    if c_surv and b_surv:
        if c_rew > b_rew + 0.05:
            return True
        elif b_rew > c_rew + 0.05:
            return False
        if c_score > b_score:
            return True
        elif b_score > c_score:
            return False
        # Tie breaker: lowest step count (while still having survived)
        return c_steps < b_steps

    # 2. Candidate survived, current best did not
    if c_surv and not b_surv:
        return c_score >= b_score or c_rew >= b_rew

    # 3. Candidate did not survive, current best did survive
    if not c_surv and b_surv:
        return c_score > b_score and c_rew > b_rew

    # 4. Neither candidate nor current best survived
    if not c_surv and not b_surv:
        if c_score > b_score and c_rew > b_rew:
            return True
        elif c_rew > b_rew + 0.05:
            return True
        elif b_rew > c_rew + 0.05:
            return False
        if c_score > b_score:
            return True
        elif b_score > c_score:
            return False
        # [if no agents with the highest score survived then take the one with the highest survival time]
        return c_steps > b_steps

    return False


def _save_stage_model(self, kind: str, info: dict):
    """
    Saves stage-specific model checkpoint in the run folder (and agent dir).
    kind is 'best' or 'last'.
    """
    stage = getattr(self, 'stage', 'default')
    stem = os.path.splitext(os.path.basename(self.model_file))[0]
    ext = os.path.splitext(self.model_file)[1] or '.pt'

    path_primary = _output_path(self, f"{kind}_model_{stage}{ext}")
    save_model(self.model, path_primary)

    info_path = _output_path(self, f"{kind}_model_{stage}.json")
    try:
        with open(info_path, 'w') as fh:
            json.dump(info, fh, indent=2)
    except Exception:
        pass

    self.logger.info(
        f"[{stage}] Saved {kind.upper()} model (round {info.get('round')}, "
        f"score={info.get('score')}, reward={info.get('sum_reward')}, "
        f"survived={info.get('survived')}, steps={info.get('steps')}) -> {os.path.basename(path_primary)}"
    )


# ----------------------------------------------------------------------------
# Setup Training
# ----------------------------------------------------------------------------
def setup_training(self):
    self.run_dir = os.environ.get('MY_WM_RUN_DIR')
    self.batch_size = _env_int('MY_WM_BATCH', 64)
    self.replay_buffer = deque(maxlen=_env_int('MY_WM_BUFFER', 25000))
    self.augment = _env_int('MY_WM_AUGMENT', 1) == 1
    self.save_every = _env_int('MY_WM_SAVE_EVERY', 25)
    self.n_rounds_target = _env_int('MY_WM_N_ROUNDS', 0)
    self.stage = os.environ.get('MY_WM_STAGE', 'unnamed')
    self.eps_end = _env_float('MY_WM_EPS_END', 0.05)

    self._transition_buffer = []
    try:
        import parallel_train
        if getattr(parallel_train, 'ACTIVE_TRANSITION_QUEUE', None) is not None:
            self.transition_queue = parallel_train.ACTIVE_TRANSITION_QUEUE
        if getattr(parallel_train, 'ACTIVE_GRAD_QUEUE', None) is not None:
            self.grad_queue = parallel_train.ACTIVE_GRAD_QUEUE
        if getattr(parallel_train, 'ACTIVE_STATS_QUEUE', None) is not None:
            self.stats_queue = parallel_train.ACTIVE_STATS_QUEUE
        if getattr(parallel_train, 'ACTIVE_WORKER_ID', None) is not None:
            self.worker_id = parallel_train.ACTIVE_WORKER_ID
    except Exception:
        pass

    state = {}
    state_path = _output_path(self, STATE_FILE)
    if os.path.isfile(state_path) and os.path.isfile(self.model_file):
        with open(state_path) as fh:
            state = json.load(fh)
    self.rounds_done = int(state.get('rounds_done', 0))
    self.rounds_this_run = 0
    self.epsilon = float(state.get('epsilon', 1.0))
    if os.environ.get('MY_WM_EPSILON') is not None:
        self.epsilon = float(os.environ['MY_WM_EPSILON'])
    if os.environ.get('MY_WM_EPS_END') is not None:
        self.eps_end = float(os.environ['MY_WM_EPS_END'])
    if os.environ.get('MY_WM_EPS_DECAY') is not None:
        self.eps_decay = float(os.environ['MY_WM_EPS_DECAY'])
    else:
        eps_rounds = _env_int('MY_WM_EPS_ROUNDS', 0)
        if eps_rounds > 0 and self.epsilon > self.eps_end:
            self.eps_decay = (self.eps_end / self.epsilon) ** (1.0 / eps_rounds)
        else:
            self.eps_decay = _env_float('MY_WM_EPS_DECAY', 0.9995)

    if os.environ.get('MY_WM_SELF_PLAY') == '1' or self.epsilon <= 0.0:
        self.epsilon = 0.0
        self.eps_end = 0.0
        self.eps_decay = 1.0

    _reset_round_stats(self)
    self.reward_history = deque(maxlen=100 * 400)
    self._t_round_start = time.time()
    self.plot_every = _env_int('MY_WM_PLOT_EVERY', 10)
    self.plot_interval_s = _env_float('MY_WM_PLOT_INTERVAL', 5.0)
    self._last_plot_time = time.time()
    stats_path = _output_path(self, STATS_FILE)
    if getattr(self, 'stats_queue', None) is None and not os.path.isfile(stats_path):
        with open(stats_path, 'w') as fh:
            fh.write(','.join(CSV_COLUMNS) + '\n')

    # Track best round per stage from historical stats if continuing
    self._stage_best_round = {}
    if os.path.isfile(stats_path):
        try:
            with open(stats_path, 'r') as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    st = r.get('stage', self.stage)
                    if is_better_round(r, self._stage_best_round.get(st)):
                        self._stage_best_round[st] = r
        except Exception:
            pass

    self.logger.info(f"Spatial DQN Training setup: stage={self.stage} rounds_done={self.rounds_done} "
                     f"epsilon={self.epsilon:.3f} decay={self.eps_decay:.5f} augment={self.augment} "
                     f"buffer={self.replay_buffer.maxlen} batch={self.batch_size}")


def _reset_round_stats(self):
    self._consecutive_stationary_steps = 0
    self._steps_without_progress = 0
    self._consecutive_loop_steps = 0
    self.round_stats = {
        'coins': 0, 'kills': 0, 'crates': 0, 'bombs': 0, 'invalid_acts': 0,
        'killed_self': 0, 'got_killed': 0, 'unsafe_actions': 0,
        'rewards': [], 'td_errors': []
    }
    self._t_round_start = time.time()


def _crate_distance(game_state):
    field = game_state['field']
    pos = game_state['self'][3]
    targets = crate_adjacent_tiles(field)
    obstacles = {b[0] for b in game_state.get('bombs', [])} | {o[3] for o in game_state.get('others', [])}
    _, d = bfs_direction(pos, field, targets, obstacles)
    return d


# ----------------------------------------------------------------------------
# Custom Events & Reward Shaping
# ----------------------------------------------------------------------------
def custom_events(self, old_state, action, new_state, old_f, new_f, events):
    """World-model based auxiliary events and penalties."""
    extra = []
    in_danger_old = old_f[F_IN_DANGER] > 0.5
    in_danger_new = new_f is not None and new_f[F_IN_DANGER] > 0.5
    moved = any(ev in events for ev in (e.MOVED_UP, e.MOVED_DOWN, e.MOVED_LEFT, e.MOVED_RIGHT))
    bombed = e.BOMB_DROPPED in events

    # --- Survival & Safety
    if action in MOVE_ACTIONS:
        chosen_safe = old_f[F_SAFE_MOVE + MOVE_ACTIONS.index(action)] > 0.5
    else:
        chosen_safe = old_f[F_SAFE_WAIT] > 0.5
    any_safe = (old_f[F_SAFE_MOVE:F_SAFE_MOVE + 4] > 0.5).any() or old_f[F_SAFE_WAIT] > 0.5

    if action == 'BOMB':
        if old_f[F_TRAPS_SELF] > 0.5:
            extra.append(SUICIDAL_BOMB)
    elif not chosen_safe and any_safe:
        extra.append(UNSAFE_ACTION)
    if in_danger_old and not in_danger_new and new_f is not None:
        extra.append(MOVED_TO_SAFETY)
    if in_danger_old and in_danger_new and not moved and not bombed:
        extra.append(STAYED_IN_DANGER)

    # --- Bomb utility
    if bombed:
        if old_f[F_TRAPS_SELF] < 0.5:
            extra.append(USEFUL_BOMB if old_f[F_BOMB_UTIL] > 0.0 else USELESS_BOMB)

    n_crates = int(np.count_nonzero(old_state['field'] == 1))
    n_coins = len(old_state.get('coins', []))
    n_opponents = len(old_state.get('others', []))
    n_objects = n_crates + n_coins + n_opponents
    has_targets = n_objects > 0

    # Scale suicide penalty when objects on the field are sparse
    # (so walking to a distant bot or crate is always preferable to suicide)
    if e.KILLED_SELF in events:
        extra_suicide_penalties = max(0, min(10, 10 - n_objects))
        if extra_suicide_penalties > 0:
            extra.extend([SPARSE_SUICIDE_PENALTY] * extra_suicide_penalties)

    # --- Navigation Potential (compare distances before/after)
    if new_f is not None and e.COIN_COLLECTED not in events and e.KILLED_OPPONENT not in events:
        if old_f[F_COIN_PROX] > 0 and new_f[F_COIN_PROX] > 0:
            d_old = 1.0 / old_f[F_COIN_PROX] - 1.0
            d_new = 1.0 / new_f[F_COIN_PROX] - 1.0
            if d_new < d_old - 0.5:
                extra.append(MOVED_TOWARD_COIN)
            elif d_new > d_old + 0.5:
                extra.append(MOVED_AWAY_COIN)
        elif old_f[F_COIN_PROX] == 0 and not in_danger_old and not bombed:
            if n_crates > 0 and old_f[F_BOMBS_LEFT] > 0.5:
                d_old = _crate_distance(old_state)
                d_new = _crate_distance(new_state)
                if d_old < INF and d_new < INF and d_old > 0:
                    if d_new < d_old:
                        extra.append(MOVED_TOWARD_CRATE)
                    elif d_new > d_old:
                        extra.append(MOVED_AWAY_CRATE)
            elif n_crates == 0 and n_opponents > 0 and old_f[F_OPP_PROX] > 0 and new_f[F_OPP_PROX] > 0:
                d_old = 1.0 / old_f[F_OPP_PROX] - 1.0
                d_new = 1.0 / new_f[F_OPP_PROX] - 1.0
                if d_new < d_old - 0.5:
                    extra.append(MOVED_TOWARD_OPPONENT)
                elif d_new > d_old + 0.5:
                    extra.append(MOVED_AWAY_OPPONENT)

    # --- Progress tracking (coin collected, crate destroyed, opponent killed, coin uncovered)
    progress_events = (e.COIN_COLLECTED, e.CRATE_DESTROYED, e.KILLED_OPPONENT, e.COIN_FOUND)
    if not has_targets:
        self._steps_without_progress = 0
        self._consecutive_stationary_steps = 0
    elif any(ev in events for ev in progress_events):
        self._steps_without_progress = 0
        if hasattr(self, 'coordinate_history'):
            self.coordinate_history.clear()
    else:
        self._steps_without_progress = getattr(self, '_steps_without_progress', 0) + 1

    # --- Lack of substantial progress penalty (exempt when actively moving to target)
    actively_moving_to_target = (
        MOVED_TOWARD_COIN in extra or
        MOVED_TOWARD_CRATE in extra or
        MOVED_TOWARD_OPPONENT in extra
    )
    if has_targets and not in_danger_old and not actively_moving_to_target:
        if self._steps_without_progress >= 1:
            num_penalties = min(8, 1 + (self._steps_without_progress - 1) // 3)
            extra.extend([NO_PROGRESS] * num_penalties)
            if self._steps_without_progress >= 10:
                extra.append(STAGNATION)
            if self._steps_without_progress >= 20:
                extra.append(STAGNATION)
            if self._steps_without_progress >= 35:
                extra.append(STAGNATION)

    # --- Idling / Standing around (only when there are targets to pursue)
    if not has_targets or moved:
        self._consecutive_stationary_steps = 0
    elif not bombed:
        self._consecutive_stationary_steps = getattr(self, '_consecutive_stationary_steps', 0) + 1
        if has_targets and self._consecutive_stationary_steps >= 3 and not in_danger_old:
            extra.append(STANDING_AROUND)

    something_to_do = old_f[F_COIN_PROX] > 0 or old_f[F_OPP_PROX] > 0 or (old_state['field'] == 1).any()
    if has_targets and not moved and not bombed and not in_danger_old and something_to_do:
        extra.append(IDLE_WAIT)

    # --- Arbitrary loop detection (periodic cycles k in [2, 12], corridor patrols, direct reversals; only when targets exist)
    is_loop = False
    loop_reps = 1
    if has_targets and moved and hasattr(self, 'coordinate_history') and len(self.coordinate_history) >= 2:
        new_pos = new_state['self'][3] if new_state is not None else None
        if new_pos is not None:
            traj = list(self.coordinate_history) + [new_pos]
            is_cycle, cycle_k, reps = detect_cycle(traj, max_k=12)
            if is_cycle:
                is_loop = True
                loop_reps = reps
            elif len(self.coordinate_history) >= 2 and new_pos == self.coordinate_history[-2]:
                is_loop = True
            elif sum(1 for p in self.coordinate_history if p == new_pos) >= 2:
                is_loop = True

    if is_loop and not in_danger_old and not in_danger_new:
        extra.append(LOOP_DETECTED)
        if loop_reps >= 2:
            extra.append(LOOP_DETECTED)
        self._consecutive_loop_steps = getattr(self, '_consecutive_loop_steps', 0) + 1
        if self._consecutive_loop_steps >= 3:
            extra.append(LOOP_DETECTED)
    else:
        self._consecutive_loop_steps = 0

    return extra


def reward_from_events(events):
    return float(sum(REWARD_TABLE.get(ev, 0.0) for ev in events))


def _store_and_learn(self, old_s, old_f, a_idx, r, new_s, new_f, done):
    if new_s is None:
        new_s = np.zeros((N_SPATIAL_CHANNELS, BOARD_W, BOARD_H), dtype=np.float32)
    if new_f is None:
        new_f = np.zeros(N_FEATURES, dtype=np.float32)

    tq = getattr(self, 'transition_queue', None)
    if tq is not None:
        # High-performance parallel mode: stream raw transitions to Central GPU Learner
        self._transition_buffer.append((old_s, old_f, int(a_idx), float(r), new_s, new_f, bool(done)))
        if len(self._transition_buffer) >= 16 or done:
            try:
                tq.put_nowait(self._transition_buffer)
            except Exception:
                try:
                    tq.put(self._transition_buffer, timeout=0.05)
                except Exception:
                    pass
            self._transition_buffer = []
        return

    if len(self.replay_buffer) >= self.batch_size:
        batch = random.sample(self.replay_buffer, self.batch_size)
        S, F, A, R, NS, NF, D = zip(*batch)
        grad_queue = getattr(self, 'grad_queue', None)
        if grad_queue is not None:
            td, grads = self.model.compute_gradients(
                np.asarray(S, dtype=np.float32),
                np.asarray(F, dtype=np.float32),
                np.asarray(A, dtype=np.int64),
                np.asarray(R, dtype=np.float32),
                np.asarray(NS, dtype=np.float32),
                np.asarray(NF, dtype=np.float32),
                np.asarray(D, dtype=bool),
            )
            grad_queue.put((getattr(self, 'worker_id', 0), td, grads))
        else:
            td = self.model.update_batch(
                np.asarray(S, dtype=np.float32),
                np.asarray(F, dtype=np.float32),
                np.asarray(A, dtype=np.int64),
                np.asarray(R, dtype=np.float32),
                np.asarray(NS, dtype=np.float32),
                np.asarray(NF, dtype=np.float32),
                np.asarray(D, dtype=bool),
            )
        self.round_stats['td_errors'].append(td)


def _count_events(self, events):
    rs = self.round_stats
    for ev in events:
        if ev == e.COIN_COLLECTED: rs['coins'] += 1
        elif ev == e.KILLED_OPPONENT: rs['kills'] += 1
        elif ev == e.CRATE_DESTROYED: rs['crates'] += 1
        elif ev == e.BOMB_DROPPED: rs['bombs'] += 1
        elif ev == e.INVALID_ACTION: rs['invalid_acts'] += 1
        elif ev == e.KILLED_SELF: rs['killed_self'] = 1
        elif ev == e.GOT_KILLED: rs['got_killed'] = 1
        elif ev == UNSAFE_ACTION: rs['unsafe_actions'] += 1


# ----------------------------------------------------------------------------
# Framework Training Callbacks
# ----------------------------------------------------------------------------
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action not in ACTION_TO_IDX:
        return
    old_s = state_to_spatial_tensor(old_game_state, self.coordinate_history)
    old_f = state_to_features(old_game_state, self.coordinate_history)

    if new_game_state is not None:
        new_s = state_to_spatial_tensor(new_game_state, self.coordinate_history)
        new_f = state_to_features(new_game_state, self.coordinate_history)
    else:
        new_s, new_f = None, None

    events = list(events) + custom_events(self, old_game_state, self_action, new_game_state, old_f, new_f, events)
    reward = reward_from_events(events)
    _count_events(self, events)
    self.round_stats['rewards'].append(reward)
    self.reward_history.append(reward)
    self.logger.debug(f"step {old_game_state['step']}: {self_action} -> reward {reward:+.2f} events={events}")

    _store_and_learn(self, old_s, old_f, ACTION_TO_IDX[self_action], reward, new_s, new_f, new_game_state is None)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if e.SURVIVED_ROUND in events:
        events = [e.SURVIVED_ROUND]
    if last_game_state is not None and last_action in ACTION_TO_IDX:
        s = state_to_spatial_tensor(last_game_state, self.coordinate_history)
        f = state_to_features(last_game_state, self.coordinate_history)
        if e.SURVIVED_ROUND not in events:
            events = list(events) + custom_events(self, last_game_state, last_action, None, f, None, events)
        reward = reward_from_events(events)
        _count_events(self, events)
        self.round_stats['rewards'].append(reward)
        self.reward_history.append(reward)
        _store_and_learn(self, s, f, ACTION_TO_IDX[last_action], reward, None, None, True)

    self.rounds_done += 1
    self.rounds_this_run += 1
    if os.environ.get('MY_WM_SELF_PLAY') == '1':
        self.epsilon = 0.0
    else:
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    rs = self.round_stats
    rewards = np.asarray(rs['rewards']) if rs['rewards'] else np.zeros(1)
    row = {
        'round': self.rounds_done, 'stage': self.stage, 'epsilon': round(self.epsilon, 4),
        'score': rs['coins'] + 5 * rs['kills'],
        'coins': rs['coins'], 'kills': rs['kills'], 'crates': rs['crates'], 'bombs': rs['bombs'],
        'survived': int(e.SURVIVED_ROUND in events),
        'steps': last_game_state['step'] if last_game_state else 0,
        'invalid_acts': rs['invalid_acts'], 'killed_self': rs['killed_self'], 'got_killed': rs['got_killed'],
        'mean_reward': round(float(rewards.mean()), 4), 'sum_reward': round(float(rewards.sum()), 2),
        'mean_td_error': round(float(np.mean(rs['td_errors'])), 4) if rs['td_errors'] else '',
        'unsafe_actions': rs['unsafe_actions'],
        'wall_time_s': round(time.time() - self._t_round_start, 2),
    }
    # Stage best model tracking
    cand_info = {
        'round': self.rounds_done,
        'stage': self.stage,
        'score': row['score'],
        'sum_reward': row['sum_reward'],
        'survived': row['survived'],
        'steps': row['steps'],
        'epsilon': row['epsilon'],
        'wall_time_s': row['wall_time_s'],
    }

    tq = getattr(self, 'transition_queue', None)
    if tq is not None and getattr(self, '_transition_buffer', None):
        try:
            tq.put_nowait(self._transition_buffer)
        except Exception:
            pass
        self._transition_buffer = []

    stats_queue = getattr(self, 'stats_queue', None)
    if stats_queue is not None:
        stats_queue.put({'worker_id': getattr(self, 'worker_id', 0), 'row': row, 'cand_info': cand_info})
    else:
        stats_path = _output_path(self, STATS_FILE)
        with open(stats_path, 'a') as fh:
            fh.write(','.join(str(row[c]) for c in CSV_COLUMNS) + '\n')

        cur_best = self._stage_best_round.get(self.stage)
        if is_better_round(cand_info, cur_best):
            self._stage_best_round[self.stage] = cand_info
            _save_stage_model(self, kind='best', info=cand_info)

        last = (self.n_rounds_target > 0 and self.rounds_this_run >= self.n_rounds_target)
        if self.rounds_done % self.save_every == 0 or self.rounds_this_run == 1 or last:
            save_model(self.model, self.model_file)
            _save_stage_model(self, kind='last', info=cand_info)
            if os.environ.get('MY_WM_SELF_PLAY') != '1':
                with open(_output_path(self, STATE_FILE), 'w') as fh:
                    json.dump({'rounds_done': self.rounds_done, 'epsilon': self.epsilon, 'stage': self.stage}, fh, indent=2)
            self.logger.info(f"Saved spatial DQN checkpoint to {self.model_file} (round {self.rounds_done})")

        # Live update training progress curves
        now = time.time()
        should_plot = (
            (self.rounds_this_run % self.plot_every == 0) or
            (now - self._last_plot_time >= self.plot_interval_s) or
            (self.rounds_done % self.save_every == 0) or
            (self.rounds_this_run == 1)
        )
        if should_plot:
            self._last_plot_time = now
            try:
                from .plot_training import plot_stats_file
            except ImportError:
                try:
                    from plot_training import plot_stats_file
                except ImportError:
                    plot_stats_file = None
            if plot_stats_file is not None:
                try:
                    curves_path = _output_path(self, 'training_curves-spatial-dqn.png')
                    plot_stats_file(stats_path, curves_path, window=25)
                    # Also save standard training_curves.png
                    canonical_curves = _output_path(self, 'training_curves.png')
                    if curves_path != canonical_curves and os.path.isfile(curves_path):
                        import shutil
                        shutil.copyfile(curves_path, canonical_curves)
                    if self.run_dir:
                        import shutil
                        shutil.copyfile(curves_path, agent_path('training_curves-spatial-dqn.png'))
                        shutil.copyfile(curves_path, agent_path('training_curves.png'))
                except BaseException as plot_err:
                    self.logger.warning(f"Plotting skipped due to error: {plot_err}")

    _reset_round_stats(self)
