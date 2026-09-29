"""
train.py -- Training loop for my_spatial_qwm_agent.

Supports:
  - Phased training: World model warmup (DQN backbone frozen) -> Joint learning
  - D4 Dihedral Symmetry Augmentation (8x sample efficiency)
  - Prioritized transition storage and batch optimization
  - Logging of Q-loss and World Model prediction errors (transition MSE and reward loss)
"""

from collections import deque
import csv
import json
import logging
import os
from pathlib import Path
import random
import time
from typing import List, Optional

import numpy as np
import torch

import events as e
from .callbacks import get_blast_map, find_escape_moves, detect_cycle
from .model import (
    ACTIONS,
    ACTION_TO_IDX,
    N_ACTIONS,
    agent_path,
    augment_spatial_transition,
    state_to_features,
    state_to_spatial_tensor,
    valid_action_mask,
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
)

# Custom shaping events and active penalties
DROPPED_BOMB_IN_DANGER = 'DROPPED_BOMB_IN_DANGER'
WAITED_IN_DANGER_WITH_ESCAPE = 'WAITED_IN_DANGER_WITH_ESCAPE'
STEPPED_INTO_BLAST = 'STEPPED_INTO_BLAST'
SUICIDAL_BOMB = 'SUICIDAL_BOMB'
USEFUL_BOMB = 'USEFUL_BOMB'
USELESS_BOMB = 'USELESS_BOMB'
MOVED_TO_SAFETY = 'MOVED_TO_SAFETY'
APPROACHED_SAFETY = 'APPROACHED_SAFETY'
STAYED_IN_DANGER = 'STAYED_IN_DANGER'
UNSAFE_ACTION = 'UNSAFE_ACTION'
LOOP_DETECTED = 'LOOP_DETECTED'
REVERSED_MOVE = 'REVERSED_MOVE'
SEVERE_LOOP_DETECTED = 'SEVERE_LOOP_DETECTED'
TILE_REVISITED = 'TILE_REVISITED'
IDLE_WAIT = 'IDLE_WAIT'
STANDING_AROUND = 'STANDING_AROUND'
COWARDICE_PENALTY = 'COWARDICE_PENALTY'
WON_ROUND = 'WON_ROUND'
LOST_ROUND = 'LOST_ROUND'
NO_PROGRESS = 'NO_PROGRESS'
STAGNATION = 'STAGNATION'
MOVED_TOWARD_COIN = 'MOVED_TOWARD_COIN'
MOVED_AWAY_COIN = 'MOVED_AWAY_COIN'
MOVED_TOWARD_CRATE = 'MOVED_TOWARD_CRATE'
MOVED_AWAY_CRATE = 'MOVED_AWAY_CRATE'
MOVED_TOWARD_OPPONENT = 'MOVED_TOWARD_OPPONENT'
MOVED_AWAY_OPPONENT = 'MOVED_AWAY_OPPONENT'
SPARSE_SUICIDE_PENALTY = 'SPARSE_SUICIDE_PENALTY'

REWARD_TABLE = {
    # Game engine events
    e.COIN_COLLECTED: 5.0,
    e.CRATE_DESTROYED: 0.5,
    e.COIN_FOUND: 0.3,
    e.KILLED_OPPONENT: 10.0,
    e.KILLED_SELF: -20.0,
    e.GOT_KILLED: -10.0,
    e.SURVIVED_ROUND: 5.0,
    e.INVALID_ACTION: -1.0,

    # Ultimate match outcome rewards (highest priority goal of the model)
    WON_ROUND: 40.0,
    LOST_ROUND: -5.0,

    # Active safety penalties (replaces hardcoded inference shield)
    DROPPED_BOMB_IN_DANGER: -10.0,
    WAITED_IN_DANGER_WITH_ESCAPE: -10.0,
    STEPPED_INTO_BLAST: -10.0,
    SUICIDAL_BOMB: -12.0,
    USELESS_BOMB: -2.0,
    STAYED_IN_DANGER: -3.0,
    UNSAFE_ACTION: -5.0,
    LOOP_DETECTED: -8.0,
    REVERSED_MOVE: -5.0,
    SEVERE_LOOP_DETECTED: -12.0,
    TILE_REVISITED: -3.0,
    IDLE_WAIT: -2.5,
    STANDING_AROUND: -3.5,
    COWARDICE_PENALTY: -5.0,
    NO_PROGRESS: -0.3,
    STAGNATION: -0.5,
    SPARSE_SUICIDE_PENALTY: -6.0,

    # Active tactical rewards
    MOVED_TO_SAFETY: 3.0,
    APPROACHED_SAFETY: 1.5,
    USEFUL_BOMB: 2.0,
    MOVED_TOWARD_COIN: 0.6,
    MOVED_AWAY_COIN: -0.6,
    MOVED_TOWARD_CRATE: 0.4,
    MOVED_AWAY_CRATE: -0.4,
    MOVED_TOWARD_OPPONENT: 0.5,
    MOVED_AWAY_OPPONENT: -0.5,
}

STATE_FILE = 'training_state-spatial-qwm.json'
STATS_FILE = 'training_stats-spatial-qwm.csv'
CSV_COLUMNS = [
    'round', 'stage', 'epsilon', 'score', 'coins', 'kills', 'crates', 'bombs', 'survived', 'won',
    'steps', 'invalid_acts', 'killed_self', 'got_killed', 'mean_reward', 'sum_reward',
    'mean_td_error', 'wm_loss', 'trans_loss', 'rew_loss', 'future_val_loss', 'legal_loss', 'dqn_frozen', 'wall_time_s'
]


def _output_path(self, base):
    run_dir = os.environ.get('MY_WM_RUN_DIR')
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        return os.path.join(run_dir, base)
    return agent_path(base)


def _count_round_events(self, events: List[str]):
    for ev in events:
        if ev == e.COIN_COLLECTED: self._round_coins += 1
        elif ev == e.KILLED_OPPONENT: self._round_kills += 1
        elif ev == e.CRATE_DESTROYED: self._round_crates += 1
        elif ev == e.BOMB_DROPPED: self._round_bombs += 1
        elif ev == e.INVALID_ACTION: self._round_invalid += 1
        elif ev == e.KILLED_SELF: self._round_killed_self = 1
        elif ev == e.GOT_KILLED: self._round_got_killed = 1


def setup_training(self):
    """Initializes replay buffer and training hyperparameter state."""
    cfg = getattr(self, 'cfg', {})
    self.batch_size = int(os.environ.get('MY_WM_BATCH_SIZE', cfg.get('batch_size', 64)))
    self.buffer_size = int(os.environ.get('MY_WM_BUFFER_SIZE', cfg.get('buffer_size', 25000)))
    self.transitions = deque(maxlen=self.buffer_size)

    self.eps_start = float(os.environ.get('MY_WM_EPS_START', 1.0))
    self.eps_end = float(os.environ.get('MY_WM_EPS_END', 0.05))
    self.eps_decay = float(os.environ.get('MY_WM_EPS_DECAY', 0.9995))
    self.epsilon = float(os.environ.get('MY_WM_EPSILON', self.eps_start))
    self.stage = os.environ.get('MY_WM_STAGE', 'unnamed')

    # Self-play zero discovery enforcement
    is_self_play = (os.environ.get('MY_WM_SELF_PLAY') == '1')
    if is_self_play:
        self.epsilon = 0.0
        self.eps_end = 0.0
        self.eps_decay = 1.0

    # Phased training: warmup steps for world model with frozen DQN
    self.wm_warmup_steps = int(os.environ.get('MY_WM_WARMUP_STEPS', cfg.get('wm_warmup_steps', 400)))
    self.train_step_count = 0

    self._round_rewards = []
    self._round_losses = []
    self._round_wm_losses = []
    self._round_trans_losses = []
    self._round_rew_losses = []
    self._round_future_val_losses = []
    self._round_legal_losses = []
    self._episode_history = []
    self.path_buffer = deque(maxlen=5000)
    self._round_start_time = time.time()
    self._total_rounds = 0

    self._round_coins = 0
    self._round_kills = 0
    self._round_crates = 0
    self._round_bombs = 0
    self._round_invalid = 0
    self._round_killed_self = 0
    self._round_got_killed = 0
    self._round_won = 0
    self._last_round_won = 0
    self._consecutive_stationary_steps = 0
    self._last_plot_time = 0.0

    self.transition_queue = None
    self.stats_queue = None
    self.worker_id = 0
    self._transition_buffer = []
    try:
        import parallel_train
        if getattr(parallel_train, 'ACTIVE_TRANSITION_QUEUE', None) is not None:
            self.transition_queue = parallel_train.ACTIVE_TRANSITION_QUEUE
        if getattr(parallel_train, 'ACTIVE_STATS_QUEUE', None) is not None:
            self.stats_queue = parallel_train.ACTIVE_STATS_QUEUE
        if getattr(parallel_train, 'ACTIVE_WORKER_ID', None) is not None:
            self.worker_id = parallel_train.ACTIVE_WORKER_ID
    except Exception:
        pass

    state_path = _output_path(self, STATE_FILE)
    if os.path.isfile(state_path) and os.path.isfile(self.model_file):
        try:
            with open(state_path) as fh:
                st = json.load(fh)
                self._total_rounds = int(st.get('rounds_done', 0))
        except Exception:
            pass


def custom_events(
    self,
    old_state: dict,
    action: str,
    new_state: Optional[dict],
    old_features: np.ndarray,
    new_features: Optional[np.ndarray],
    events: List[str]
) -> List[str]:
    """
    Computes auxiliary events and active penalties to train the model to avoid dangerous actions
    (dropping bombs in danger, waiting in danger, stepping into blast, suicidal bomb drops)
    and reward safe, purposeful actions (moving to safety, useful bombs, coin/crate approach).
    """
    extra = []
    if old_state is None or action is None:
        return extra

    arena = old_state['field']
    bombs = old_state.get('bombs', [])
    exp_map = old_state.get('explosion_map')
    self_info = old_state.get('self')
    if not self_info:
        return extra
    pos = self_info[3]
    others = tuple(o[3] for o in old_state.get('others', []))

    in_danger_old, old_escape_moves = find_escape_moves(arena, bombs, exp_map, pos, others)

    if new_state is not None and new_state.get('self') is not None:
        new_arena = new_state['field']
        new_bombs = new_state.get('bombs', [])
        new_exp_map = new_state.get('explosion_map')
        new_pos = new_state['self'][3]
        new_others = tuple(o[3] for o in new_state.get('others', []))
        in_danger_new, _ = find_escape_moves(new_arena, new_bombs, new_exp_map, new_pos, new_others)
    else:
        in_danger_new = False

    # 1. Dropping bomb behavior & penalties
    if action == 'BOMB' or e.BOMB_DROPPED in events:
        if in_danger_old:
            extra.append(DROPPED_BOMB_IN_DANGER)
        # Check if bomb placement is suicidal (traps self with no short escape)
        hypo_bombs = list(bombs) + [(pos, 4)]
        _, hypo_esc = find_escape_moves(arena, hypo_bombs, exp_map, pos, others)
        min_esc_d = min((d for _, d in hypo_esc), default=999)
        if not hypo_esc or min_esc_d > 2 or (old_features is not None and old_features[F_TRAPS_SELF] > 0.5):
            extra.append(SUICIDAL_BOMB)
        elif old_features is not None and old_features[F_BOMB_UTIL] > 0.0:
            extra.append(USEFUL_BOMB)
        else:
            extra.append(USELESS_BOMB)

    # 2. Waiting behavior in danger
    if action == 'WAIT':
        if in_danger_old and old_escape_moves:
            extra.append(WAITED_IN_DANGER_WITH_ESCAPE)
        elif not in_danger_old:
            n_crates = int(np.count_nonzero(arena == 1))
            n_coins = len(old_state.get('coins', []))
            n_opps = len(old_state.get('others', []))
            if (n_crates + n_coins + n_opps) > 0:
                extra.append(IDLE_WAIT)
                stat_steps = getattr(self, '_consecutive_stationary_steps', 0)
                if stat_steps >= 1:
                    extra.append(STANDING_AROUND)

    # 3. Stepping into blast from safe tile
    if not in_danger_old and action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        old_blast_map = get_blast_map(arena, bombs)
        dx, dy = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}[action]
        tx, ty = pos[0] + dx, pos[1] + dy
        if 0 <= tx < arena.shape[0] and 0 <= ty < arena.shape[1]:
            if old_blast_map[tx, ty] <= 1 or (exp_map is not None and exp_map[tx, ty] > 0):
                extra.append(STEPPED_INTO_BLAST)

    # 4. Escape / Fleeing behavior while in danger
    if in_danger_old:
        if not in_danger_new and new_state is not None and new_state.get('self') is not None:
            extra.append(MOVED_TO_SAFETY)
        elif action in ('UP', 'DOWN', 'LEFT', 'RIGHT') and old_escape_moves:
            a_idx = ACTION_TO_IDX[action]
            esc_a_indices = {a for a, _ in old_escape_moves}
            if a_idx in esc_a_indices:
                extra.append(APPROACHED_SAFETY)
            else:
                extra.append(STAYED_IN_DANGER)
        else:
            extra.append(STAYED_IN_DANGER)

    # 5. Cycle / Loop detection & Direct Reversal
    if new_state is not None and new_state.get('self') is not None:
        new_pos = new_state['self'][3]
    elif action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        dx, dy = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}[action]
        new_pos = (pos[0] + dx, pos[1] + dy)
    else:
        new_pos = pos

    if hasattr(self, 'coordinate_history') and len(self.coordinate_history) >= 2:
        # A. Direct Reversal (stepping straight back to previous tile when safe)
        prev_tile = self.coordinate_history[-2] if (len(self.coordinate_history) >= 2 and self.coordinate_history[-1] == pos) else self.coordinate_history[-1]
        if prev_tile is not None and new_pos == prev_tile and not in_danger_old and not in_danger_new:
            extra.append(REVERSED_MOVE)

        # B. Cycle detection of period k in [2, 12]
        cand_traj = list(self.coordinate_history)
        if not cand_traj or cand_traj[-1] != pos:
            cand_traj.append(pos)
        cand_traj.append(new_pos)

        is_cycle, cycle_k, reps = detect_cycle(cand_traj, max_k=12)
        if is_cycle:
            extra.append(LOOP_DETECTED)
            if reps >= 3:
                extra.append(SEVERE_LOOP_DETECTED)

        # C. Frequent revisit of recent tiles without danger
        visits = sum(1 for p in self.coordinate_history if p == new_pos)
        if visits >= 2 and not in_danger_old and not in_danger_new:
            extra.append(TILE_REVISITED)

    # 6. Navigation towards targets
    if old_features is not None and new_features is not None and e.COIN_COLLECTED not in events and e.KILLED_OPPONENT not in events:
        # Coin progress
        if old_features[F_COIN_PROX] > 0 and new_features[F_COIN_PROX] > 0:
            if new_features[F_COIN_PROX] > old_features[F_COIN_PROX]:
                extra.append(MOVED_TOWARD_COIN)
            elif new_features[F_COIN_PROX] < old_features[F_COIN_PROX]:
                extra.append(MOVED_AWAY_COIN)
        # Crate progress
        elif old_features[F_CRATE_DIR:F_CRATE_DIR + 4].any() and action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
            dir_idx = ACTION_TO_IDX[action]
            if old_features[F_CRATE_DIR + dir_idx] > 0.5:
                extra.append(MOVED_TOWARD_CRATE)
            else:
                extra.append(MOVED_AWAY_CRATE)
        # Opponent progress
        elif old_features[F_OPP_PROX] > 0 and new_features[F_OPP_PROX] > 0:
            if new_features[F_OPP_PROX] > old_features[F_OPP_PROX]:
                extra.append(MOVED_TOWARD_OPPONENT)
            elif new_features[F_OPP_PROX] < old_features[F_OPP_PROX]:
                extra.append(MOVED_AWAY_OPPONENT)

    return extra


def compute_reward(events_list: List[str]) -> float:
    """Sums rewards according to REWARD_TABLE."""
    return float(sum(REWARD_TABLE.get(ev, 0.0) for ev in events_list))


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Stores transitions, applies dihedral augmentation, and performs training step."""
    if old_game_state is None or self_action is None:
        return

    # Track consecutive stationary steps during training
    old_self = old_game_state.get('self')
    new_self = new_game_state.get('self') if new_game_state else None
    if old_self and new_self and old_self[3] == new_self[3]:
        self._consecutive_stationary_steps = getattr(self, '_consecutive_stationary_steps', 0) + 1
    else:
        self._consecutive_stationary_steps = 0

    old_spatial = state_to_spatial_tensor(old_game_state, self.coordinate_history)
    old_features = state_to_features(old_game_state, self.coordinate_history)

    action_idx = ACTION_TO_IDX[self_action]

    new_spatial = state_to_spatial_tensor(new_game_state, self.coordinate_history)
    new_features = state_to_features(new_game_state, self.coordinate_history)

    # Detect active penalties and rewards, then add to events
    extra = custom_events(self, old_game_state, self_action, new_game_state, old_features, new_features, events)
    events.extend(extra)
    _count_round_events(self, events)

    reward = compute_reward(events)
    done = float(new_game_state.get('self') is None or new_game_state['self'][1] is None) if new_game_state else 1.0

    self._round_rewards.append(reward)

    tq = getattr(self, 'transition_queue', None)
    if tq is not None:
        self._transition_buffer.append((old_spatial, old_features, int(action_idx), float(reward), new_spatial, new_features, bool(done)))
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

    # Local training mode: Augment transition with all 8 D4 dihedral symmetries
    aug_transitions = augment_spatial_transition(old_spatial, old_features, action_idx, new_spatial, new_features)
    for s_aug, f_aug, a_aug, ns_aug, nf_aug in aug_transitions:
        self.transitions.append((s_aug, f_aug, a_aug, reward, ns_aug, nf_aug, done))

    # Store unaugmented sequential steps for multi-step candidate path rollouts
    if hasattr(self, '_episode_history'):
        self._episode_history.append((old_spatial, old_features, action_idx, reward, new_spatial, new_features, done))
        h_horizon = max(8, min(int(os.environ.get("MY_WM_PATH_HORIZON", getattr(self, "path_horizon", 8))), 16))
        if len(self._episode_history) >= h_horizon + 1:
            w_steps = self._episode_history[-(h_horizon + 1):]
            if not any(st[6] for st in w_steps[:-1]):
                s_seq = [st[0] for st in w_steps]
                f_seq = [st[1] for st in w_steps]
                a_seq = [st[2] for st in w_steps[:h_horizon]]
                r_seq = [st[3] for st in w_steps[:h_horizon]]
                d_seq = [st[6] for st in w_steps[:h_horizon]]
                self.path_buffer.append((s_seq, f_seq, a_seq, r_seq, d_seq))

    # Optimization step
    if len(self.transitions) >= self.batch_size:
        batch = random.sample(self.transitions, self.batch_size)
        spatials = np.array([t[0] for t in batch], dtype=np.float32)
        features = np.array([t[1] for t in batch], dtype=np.float32)
        actions = np.array([t[2] for t in batch], dtype=np.int64)
        rewards = np.array([t[3] for t in batch], dtype=np.float32)
        next_spatials = np.array([t[4] for t in batch], dtype=np.float32)
        next_features = np.array([t[5] for t in batch], dtype=np.float32)
        dones = np.array([t[6] for t in batch], dtype=np.float32)

        # Update model
        step_metrics = self.model.update_batch(
            spatials, features, actions, rewards, next_spatials, next_features, dones
        )

        self._round_losses.append(step_metrics["loss"])
        self._round_wm_losses.append(step_metrics["wm_loss"])
        self._round_trans_losses.append(step_metrics["trans_loss"])
        self._round_rew_losses.append(step_metrics["rew_loss"])
        if "legal_loss" in step_metrics:
            self._round_legal_losses.append(float(step_metrics["legal_loss"]))

        # Multi-step path update for Latent World Model (future imagination with random horizons 1..8)
        if hasattr(self, 'path_buffer') and len(self.path_buffer) >= 8 and hasattr(self.model, 'update_multistep_paths'):
            p_batch = random.sample(self.path_buffer, min(16, len(self.path_buffer)))
            P_S, P_F, P_A, P_R, P_D = zip(*p_batch)
            p_metrics = self.model.update_multistep_paths(
                np.array(P_S, dtype=np.float32),
                np.array(P_F, dtype=np.float32),
                np.array(P_A, dtype=np.int64),
                np.array(P_R, dtype=np.float32),
                np.array(P_D, dtype=np.float32),
                random_horizon=True,
            )
            if p_metrics.get("future_val_loss") is not None:
                self._round_future_val_losses.append(float(p_metrics["future_val_loss"]))
            if p_metrics.get("legal_loss") is not None:
                self._round_legal_losses.append(float(p_metrics["legal_loss"]))

        self.train_step_count += 1
        # Phased unfreezing: after warmup steps, unfreeze DQN if not locked by environment
        if self.train_step_count >= self.wm_warmup_steps and self.model.is_dqn_frozen:
            if os.environ.get('MY_WM_FREEZE_DQN') != '1':
                self.model.unfreeze_dqn()
                self.logger.info(f"Step {self.train_step_count}: World model warmup completed. Unfreezing DQN backbone for joint learning.")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Records final state transition, logs stats, and updates epsilon."""
    # Determine match outcome (WON_ROUND vs LOST_ROUND)
    is_winner = (e.WON_ROUND in events or 'WON_ROUND' in events or WON_ROUND in events)
    if not is_winner and last_game_state is not None:
        self_info = last_game_state.get('self')
        if self_info is not None:
            self_score = self_info[1]
            others = last_game_state.get('others', [])
            other_scores = [o[1] for o in others if len(o) > 1]
            sole_surv = (len(others) == 0 and e.SURVIVED_ROUND in events)
            score_win = (self_score > 0 and (not other_scores or self_score >= max(other_scores)))
            if sole_surv or score_win:
                is_winner = True
    self._round_won = int(is_winner)
    self._last_round_won = int(is_winner)
    if is_winner:
        if WON_ROUND not in events:
            events.append(WON_ROUND)
    else:
        if LOST_ROUND not in events:
            events.append(LOST_ROUND)

    if last_game_state is not None and last_action is not None:
        old_spatial = state_to_spatial_tensor(last_game_state, self.coordinate_history)
        old_features = state_to_features(last_game_state, self.coordinate_history)
        action_idx = ACTION_TO_IDX[last_action]
        extra = custom_events(self, last_game_state, last_action, None, old_features, None, events)
        events.extend(extra)
        _count_round_events(self, events)

        # Penalize cowardice (surviving 400 steps without collecting coins, destroying crates, or kills)
        if e.SURVIVED_ROUND in events:
            round_crates = getattr(self, '_round_crates', 0)
            round_coins = getattr(self, '_round_coins', 0)
            round_kills = getattr(self, '_round_kills', 0)
            if round_crates == 0 and round_coins == 0 and round_kills == 0:
                events.remove(e.SURVIVED_ROUND)
                events.append(COWARDICE_PENALTY)

        reward = compute_reward(events)
        self._round_rewards.append(reward)
        terminal_spatial = np.zeros_like(old_spatial)
        terminal_features = np.zeros_like(old_features)
        self.transitions.append((
            old_spatial, old_features, action_idx, reward,
            terminal_spatial, terminal_features, 1.0
        ))
        tq = getattr(self, 'transition_queue', None)
        if tq is not None:
            self._transition_buffer.append((old_spatial, old_features, int(action_idx), float(reward), terminal_spatial, terminal_features, True))
            try:
                tq.put_nowait(self._transition_buffer)
            except Exception:
                try:
                    tq.put(self._transition_buffer, timeout=0.05)
                except Exception:
                    pass
            self._transition_buffer = []
    else:
        _count_round_events(self, events)

    self._total_rounds += 1
    # Epsilon decay
    is_self_play = (os.environ.get('MY_WM_SELF_PLAY') == '1')
    if not is_self_play:
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # Save model periodically
    if hasattr(self, 'model_file') and not hasattr(self, 'grad_queue'):
        self.model.save(self.model_file)

    # Compile round row
    round_score = self._round_coins + 5 * self._round_kills
    row = {
        'round': self._total_rounds,
        'stage': getattr(self, 'stage', 0),
        'epsilon': round(getattr(self, 'epsilon', 0.0), 4),
        'score': round_score,
        'coins': self._round_coins,
        'kills': self._round_kills,
        'crates': self._round_crates,
        'bombs': self._round_bombs,
        'survived': int(e.SURVIVED_ROUND in events),
        'won': getattr(self, '_round_won', int(is_winner)),
        'steps': last_game_state['step'] if last_game_state else len(self._round_rewards),
        'invalid_acts': self._round_invalid,
        'killed_self': self._round_killed_self,
        'got_killed': self._round_got_killed,
        'mean_reward': round(float(np.mean(self._round_rewards)), 4) if self._round_rewards else 0.0,
        'sum_reward': round(float(np.sum(self._round_rewards)), 2) if self._round_rewards else 0.0,
        'mean_td_error': round(float(np.mean(self._round_losses)), 4) if getattr(self, '_round_losses', None) else '',
        'wm_loss': round(float(np.mean(self._round_wm_losses)), 4) if getattr(self, '_round_wm_losses', None) else '',
        'trans_loss': round(float(np.mean(self._round_trans_losses)), 4) if getattr(self, '_round_trans_losses', None) else '',
        'rew_loss': round(float(np.mean(self._round_rew_losses)), 4) if getattr(self, '_round_rew_losses', None) else '',
        'future_val_loss': round(float(np.mean(self._round_future_val_losses)), 4) if getattr(self, '_round_future_val_losses', None) else '',
        'legal_loss': round(float(np.mean(self._round_legal_losses)), 4) if getattr(self, '_round_legal_losses', None) else '',
        'dqn_frozen': int(getattr(getattr(self, 'model', None), 'is_dqn_frozen', False)),
        'wall_time_s': round(time.time() - getattr(self, '_round_start_time', time.time()), 2),
    }

    if hasattr(self, '_episode_history'):
        self._episode_history.clear()
    if hasattr(self, '_round_future_val_losses'):
        self._round_future_val_losses.clear()
    if hasattr(self, '_round_legal_losses'):
        self._round_legal_losses.clear()

    cand_info = {
        'round': self._total_rounds,
        'stage': getattr(self, 'stage', 0),
        'score': round_score,
        'sum_reward': row['sum_reward'],
        'survived': row['survived'],
        'won': row['won'],
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

    sq = getattr(self, 'stats_queue', None)
    if sq is not None:
        sq.put({'worker_id': getattr(self, 'worker_id', 0), 'row': row, 'cand_info': cand_info})
        _reset_round_trackers(self)
        return

    # Write stats to CSV (single process local execution)
    stats_path = _output_path(self, STATS_FILE)
    write_header = not os.path.isfile(stats_path) or os.path.getsize(stats_path) == 0
    with open(stats_path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    agent_stats = agent_path(STATS_FILE)
    if stats_path != agent_stats:
        ag_header = not os.path.isfile(agent_stats) or os.path.getsize(agent_stats) == 0
        with open(agent_stats, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            if ag_header:
                writer.writeheader()
            writer.writerow(row)

    # Save state
    with open(_output_path(self, STATE_FILE), 'w') as fh:
        json.dump({'rounds_done': self._total_rounds, 'epsilon': self.epsilon, 'stage': getattr(self, 'stage', 0)}, fh, indent=2)

    # Live update curves
    now = time.time()
    last_p_time = getattr(self, '_last_plot_time', 0.0)
    if (self._total_rounds % 5 == 0) or (now - last_p_time >= 10.0) or (self._total_rounds <= 2):
        self._last_plot_time = now
        try:
            from .plot_training import plot_stats_file
        except ImportError:
            try:
                from plot_training import plot_stats_file
            except ImportError:
                plot_stats_file = None

        if plot_stats_file is not None:
            curves_path = _output_path(self, 'training_curves-spatial-qwm.png')
            plot_stats_file(stats_path, curves_path, window=25)
            ag_curves = agent_path('training_curves-spatial-qwm.png')
            if curves_path != ag_curves and os.path.isfile(curves_path):
                import shutil
                shutil.copyfile(curves_path, ag_curves)

    _reset_round_trackers(self)


def _reset_round_trackers(self):
    self._round_rewards = []
    self._round_losses = []
    self._round_wm_losses = []
    self._round_trans_losses = []
    self._round_rew_losses = []
    self._round_coins = 0
    self._round_kills = 0
    self._round_crates = 0
    self._round_bombs = 0
    self._round_invalid = 0
    self._round_killed_self = 0
    self._round_got_killed = 0
    self._round_won = 0
    self._round_start_time = time.time()

