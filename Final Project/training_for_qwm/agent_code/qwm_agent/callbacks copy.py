"""
callbacks.py -- Inference and action selection for my_spatial_qwm_agent.

Utilizes QWM Tree Search over actions powered by the Latent World Model.
"""

from collections import deque
import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

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

try:
    from .plan_overlay import (
        CandidatePlanTrace, PlannerOverlayTrace,
        publish_overlay_trace, clear_overlay_trace, walk_path
    )
except ImportError:
    try:
        from plan_overlay import (
            CandidatePlanTrace, PlannerOverlayTrace,
            publish_overlay_trace, clear_overlay_trace, walk_path
        )
    except ImportError:
        publish_overlay_trace = None
        clear_overlay_trace = None
        walk_path = None

DEFAULT_CONFIG = {
    'model_file': 'my-saved-model-spatial-qwm.pt',
    'initial_checkpoint': '../my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/best_model_s4_vs_rule_based.pt',
    'lr': 1e-4,
    'wm_lr': 3e-4,
    'gamma': 0.95,
    'tau': 0.005,
    'batch_size': 64,
    'buffer_size': 25000,
    'tree_search': True,
    'search_depth': 4,
    'beam_size': 12,
    'tree_discount': 0.12,
    'alpha_vq': 0.45,
    'explore_unsafe': 0.05,
    'freeze_dqn': False,
    'wm_warmup_steps': 400,
    'path_horizon': 4,
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    path = agent_path('config.json')
    if os.path.isfile(path):
        with open(path) as fh:
            cfg.update(json.load(fh))
    return cfg


def setup(self):
    """Load model (with partial loading support from base DQN checkpoint) and initialize agent."""
    self.logger = getattr(self, 'logger', None)
    if self.logger is None:
        self.logger = logging.getLogger('my_spatial_qwm_agent')
        self.logger.setLevel(logging.INFO)

    self.cfg = load_config()
    fname = os.environ.get('MY_QWM_MODEL_FILE') or os.environ.get('MY_WM_MODEL_FILE') or self.cfg['model_file']
    run_dir = os.environ.get('MY_QWM_RUN_DIR') or os.environ.get('MY_WM_RUN_DIR')

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
    self.bomb_history = deque(maxlen=8)
    self._consecutive_stationary_steps = 0
    self._steps_without_progress = 0
    self._last_score = 0
    self._last_crates = None
    self.feature_cache = {}
    self._think_times = []
    self._last_round = None

    # Tree search parameters
    ts_env = os.environ.get('MY_QWM_TREE_SEARCH') or os.environ.get('MY_WM_TREE_SEARCH')
    if ts_env is not None:
        self.tree_search = (ts_env.strip().lower() in ('1', 'true', 'yes'))
    else:
        self.tree_search = bool(self.cfg.get('tree_search', True))
    raw_depth = int(os.environ.get('MY_QWM_SEARCH_DEPTH') or os.environ.get('MY_WM_SEARCH_DEPTH', self.cfg.get('search_depth', 4)))
    self.search_depth = max(1, min(raw_depth, 16))
    raw_beam = int(os.environ.get('MY_QWM_BEAM_SIZE') or os.environ.get('MY_WM_BEAM_SIZE', self.cfg.get('beam_size', 12)))
    self.beam_size = max(6, min(raw_beam, 60))
    self.tree_discount = float(os.environ.get('MY_QWM_TREE_DISCOUNT') or os.environ.get('MY_WM_TREE_DISCOUNT', self.cfg.get('tree_discount', 0.12)))
    self.alpha_vq = float(os.environ.get('MY_QWM_ALPHA_VQ') or os.environ.get('MY_WM_ALPHA_VQ', self.cfg.get('alpha_vq', 0.45)))
    self.path_horizon = max(2, min(int(os.environ.get('MY_QWM_PATH_HORIZON') or os.environ.get('MY_WM_PATH_HORIZON', self.cfg.get('path_horizon', 4))), 16))

    # Parallel training model attachment
    try:
        import parallel_train
        if parallel_train.ACTIVE_SHARED_MODEL is not None:
            self.model = parallel_train.ACTIVE_SHARED_MODEL
            self.transition_queue = parallel_train.ACTIVE_TRANSITION_QUEUE
            self.stats_queue = parallel_train.ACTIVE_STATS_QUEUE
            self.worker_id = parallel_train.ACTIVE_WORKER_ID
    except Exception:
        pass

    if not hasattr(self, 'model'):
        lr = float(os.environ.get('MY_WM_LR', self.cfg['lr']))
        wm_lr = float(os.environ.get('MY_WM_WM_LR', self.cfg['wm_lr']))

        if os.path.isfile(load_path):
            self.model = load_model(load_path, need_training=getattr(self, 'train', False), partial=True)
            self.logger.info(f"Loaded Spatial QWM model from {os.path.basename(load_path)}")
        else:
            # Check for initial checkpoint to bootstrap DQN backbone
            init_ckpt = os.environ.get('MY_WM_INIT_CHECKPOINT') or self.cfg.get('initial_checkpoint')
            self.model = new_model(lr=lr, wm_lr=wm_lr, gamma=self.cfg['gamma'], tau=self.cfg['tau'])
            if init_ckpt:
                resolved_init = os.path.abspath(os.path.join(str(AGENT_DIR), init_ckpt)) if not os.path.isabs(init_ckpt) else init_ckpt
                if os.path.isfile(resolved_init):
                    self.model.load(resolved_init, need_training=False, partial=True)
                    self.logger.info(f"Bootstrapped DQN backbone from initial checkpoint: {os.path.basename(resolved_init)}")
                else:
                    self.logger.warning(f"Initial checkpoint not found at {resolved_init}, starting fresh.")

        # Phased freezing check
        if self.train:
            freeze_env = os.environ.get('MY_WM_FREEZE_DQN')
            if freeze_env is not None:
                should_freeze = (freeze_env == '1')
            else:
                should_freeze = bool(self.cfg.get('freeze_dqn', True))

            if should_freeze:
                self.model.freeze_dqn()
                self.logger.info("DQN backbone is FROZEN. Training only Latent World Model.")
            else:
                self.model.unfreeze_dqn()
                self.logger.info("DQN backbone is UNROZEN. Joint end-to-end training.")

    if not hasattr(self, 'epsilon'):
        self.epsilon = 0.0
    self.explore_unsafe = float(os.environ.get('MY_WM_EXPLORE_UNSAFE', self.cfg['explore_unsafe']))


def detect_cycle(sequence, max_k=12):
    """Detect repeating cycles in position history."""
    n = len(sequence)
    for k in range(2, min(max_k + 1, n // 2 + 1)):
        matches = 0
        for i in range(1, n // k + 1):
            if sequence[-i * k:] == sequence[-k:] * i:
                matches = i
            else:
                break
        if matches >= 2:
            return True, k, matches
    return False, 0, 0


def get_blast_map(arena: np.ndarray, bombs: list) -> np.ndarray:
    """
    Computes minimum countdown of any bomb threatening each tile.
    999 means tile is not threatened.
    Stone walls (-1) stop blasts. In BombeRLe, blasts penetrate through crates.
    """
    blast_map = np.full(arena.shape, 999, dtype=np.int32)
    for (bx, by), t in bombs:
        blast_map[bx, by] = min(blast_map[bx, by], t)
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            for step in range(1, 4):
                nx, ny = bx + dx * step, by + dy * step
                if not (0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]):
                    break
                if arena[nx, ny] == -1:
                    break
                blast_map[nx, ny] = min(blast_map[nx, ny], t)
    return blast_map


def is_hazard_nearby(pos: Tuple[int, int], arena: np.ndarray, bombs: list, explosion_map: Optional[np.ndarray], radius: int = 4) -> bool:
    """Returns True if there is an active bomb blast reaching near pos, or any active flame near pos."""
    x, y = pos
    if explosion_map is not None:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:
                    tx, ty = x + dx, y + dy
                    if 0 <= tx < arena.shape[0] and 0 <= ty < arena.shape[1]:
                        if explosion_map[tx, ty] > 0:
                            return True
    blast_map = get_blast_map(arena, bombs)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) <= radius:
                tx, ty = x + dx, y + dy
                if 0 <= tx < arena.shape[0] and 0 <= ty < arena.shape[1]:
                    if blast_map[tx, ty] < 999:
                        return True
    return False


def find_escape_moves(arena: np.ndarray, bombs: list, explosion_map: Optional[np.ndarray], pos: tuple, others: tuple = ()) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Finds shortest escape paths from pos to any permanently safe tile.
    Returns:
      in_danger (bool): True if current position pos is in danger
      escape_moves (list of (action_idx, dist)): list of first actions that reach safety and the steps needed
    """
    x, y = pos
    blast_map = get_blast_map(arena, bombs)
    in_danger = bool((blast_map[x, y] < 999) or (explosion_map is not None and explosion_map[x, y] > 0))
    if not in_danger:
        return False, []

    bomb_xys = {b[0] for b in bombs}
    other_xys = set(others)

    # Safe tiles: free, not in any blast zone, not burning, not occupied
    safe_tiles = set()
    for rx in range(arena.shape[0]):
        for ry in range(arena.shape[1]):
            if (arena[rx, ry] == 0 and blast_map[rx, ry] == 999 
                and (explosion_map is None or explosion_map[rx, ry] == 0)
                and (rx, ry) not in bomb_xys and (rx, ry) not in other_xys):
                safe_tiles.add((rx, ry))

    if not safe_tiles:
        return True, []

    q = deque()
    visited = {(x, y)}
    escape_routes = []

    # Step 1: explore immediate moves from (x, y)
    for a_idx, (a_name, (dx, dy)) in enumerate(DIRS):
        nx, ny = x + dx, y + dy
        if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
            if arena[nx, ny] == 0 and (nx, ny) not in bomb_xys and (nx, ny) not in other_xys:
                if (explosion_map is None or explosion_map[nx, ny] == 0) and blast_map[nx, ny] > 0:
                    visited.add((nx, ny))
                    if (nx, ny) in safe_tiles:
                        escape_routes.append((a_idx, 1))
                    else:
                        q.append((nx, ny, 1, a_idx))

    if escape_routes:
        return True, escape_routes

    # BFS for multi-step escapes
    min_dist = 999
    while q:
        cx, cy, d, first_a = q.popleft()
        if d >= min_dist or d >= 6:
            continue
        for _, (dx, dy) in DIRS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                if arena[nx, ny] == 0 and (nx, ny) not in bomb_xys and (nx, ny) not in other_xys and (nx, ny) not in visited:
                    if blast_map[nx, ny] > d:
                        visited.add((nx, ny))
                        if (nx, ny) in safe_tiles:
                            min_dist = d + 1
                            escape_routes.append((first_a, d + 1))
                        else:
                            q.append((nx, ny, d + 1, first_a))

    return True, escape_routes


def check_opponent_trapped_or_threatened(
    arena: np.ndarray,
    bomb_pos: Tuple[int, int],
    bombs: list,
    exp_map: Optional[np.ndarray],
    others_raw: list,
    self_pos: Tuple[int, int],
) -> Tuple[bool, bool, int]:
    """
    Evaluates if dropping a bomb at bomb_pos traps or severely restricts opponents.
    Returns (is_any_opp_trapped, is_any_opp_severely_restricted, num_threatened).
    """
    bx, by = bomb_pos
    hypo_bombs = list(bombs) + [((bx, by), 4)]

    in_blast_opps = []
    for other in others_raw:
        if len(other) < 4:
            continue
        ox, oy = other[3]
        if ox == bx and abs(oy - by) <= 3:
            step_y = 1 if oy > by else -1
            blocked = False
            for ty in range(by + step_y, oy, step_y):
                if arena[bx, ty] == -1:
                    blocked = True
                    break
            if not blocked:
                in_blast_opps.append((ox, oy))
        elif oy == by and abs(ox - bx) <= 3:
            step_x = 1 if ox > bx else -1
            blocked = False
            for tx in range(bx + step_x, ox, step_x):
                if arena[tx, by] == -1:
                    blocked = True
                    break
            if not blocked:
                in_blast_opps.append((ox, oy))

    if not in_blast_opps:
        return False, False, 0

    any_trapped = False
    any_restricted = False
    for ox, oy in in_blast_opps:
        # Check escape moves for this threatened opponent
        # Note: self_pos (where our bomb is) and other agents are obstacles
        opp_obstacles = tuple(o[3] for o in others_raw if len(o) >= 4 and o[3] != (ox, oy)) + (self_pos,)
        _, opp_esc = find_escape_moves(arena, hypo_bombs, exp_map, (ox, oy), opp_obstacles)
        if not opp_esc:
            any_trapped = True
            break
        else:
            min_d = min(d for _, d in opp_esc)
            # Truly restricted only if escape takes 3 or more steps (bomb timer is 4)
            if min_d >= 3:
                any_restricted = True

    return any_trapped, any_restricted, len(in_blast_opps)


def _is_safe_action(features, a_idx):
    """World-model verdict for one action (used to bias exploration and enforce safety)."""
    if a_idx < 4:
        return features[F_SAFE_MOVE + a_idx] > 0.5
    if ACTIONS[a_idx] == 'WAIT':
        return features[F_SAFE_WAIT] > 0.5
    # BOMB: only safe if it leaves an escape route, waiting this turn is safe, and it hits targets
    return features[F_TRAPS_SELF] < 0.5 and features[F_SAFE_WAIT] > 0.5 and features[F_BOMB_UTIL] > 0.0


def act(self, game_state: dict) -> str:
    """Action selection using QWM Tree Search over actions with safety masking."""
    t0 = time.perf_counter()
    if game_state is None:
        return 'WAIT'

    if game_state['step'] == 1 or game_state['round'] != self._last_round:
        self.coordinate_history.clear()
        if hasattr(self, 'bomb_history'):
            self.bomb_history.clear()
        self._last_round = game_state['round']
        self._consecutive_stationary_steps = 0
        self._steps_without_progress = 0
        self._last_score = game_state['self'][1]
        self._last_crates = int(np.count_nonzero(game_state['field'] == 1))

    arena = game_state['field']
    bombs = game_state.get('bombs', [])
    exp_map = game_state.get('explosion_map')
    raw_others = game_state.get('others', [])
    others = [o[3] for o in raw_others]
    others_with_bombs = [o[3] for o in raw_others if (len(o) > 2 and bool(o[2]))]
    others_without_bombs = [o[3] for o in raw_others if (len(o) > 2 and not bool(o[2]))]

    spatial = state_to_spatial_tensor(game_state, self.coordinate_history)
    features = state_to_features(game_state, self.coordinate_history)
    mask = valid_action_mask(features)
    x, y = game_state['self'][3]

    blast_map = get_blast_map(arena, bombs)
    in_danger, escape_moves = find_escape_moves(arena, bombs, exp_map, (x, y), others)

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

    hazard_nearby = is_hazard_nearby((x, y), arena, bombs, exp_map, radius=4)

    # Never idle wait when completely safe and no bombs/explosions threaten nearby
    if has_targets and not in_danger and not hazard_nearby and mask[:4].any():
        mask[ACTION_TO_IDX['WAIT']] = False
    elif has_targets and not in_danger and self._consecutive_stationary_steps >= 2 and mask[:4].any():
        # Even if hazards exist elsewhere, don't stay frozen stationary for 3+ turns if safe moves exist
        blast_map = get_blast_map(arena, bombs)
        has_safe_move = False
        for a_i, (a_name, (dx, dy)) in enumerate(DIRS):
            if mask[a_i]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                    if blast_map[nx, ny] == 999 and (exp_map is None or exp_map[nx, ny] == 0):
                        has_safe_move = True
                        break
        if has_safe_move:
            mask[ACTION_TO_IDX['WAIT']] = False

    is_self_play = (os.environ.get('MY_WM_SELF_PLAY') == '1')
    if not is_self_play and getattr(self, 'train', False) and getattr(self, 'epsilon', 0.0) > 0 and random.random() < self.epsilon:
        valid = [i for i in range(len(ACTIONS)) if mask[i]]
        if self.coordinate_history and has_targets and not in_danger:
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
        if clear_overlay_trace is not None:
            clear_overlay_trace()
    else:
        # QWM Tree Search or Direct Q-values
        plan_details = None
        actor_action = 'WAIT'
        if getattr(self, 'tree_search', True):
            q_res = self.model.tree_search_action_scores(
                spatial, features,
                depth=self.search_depth,
                beam_size=getattr(self, 'beam_size', 12),
                discount=self.tree_discount,
                alpha_vq=self.alpha_vq,
                valid_mask=mask,
                return_details=True
            )
            if isinstance(q_res, tuple):
                q_raw, plan_details = q_res
            else:
                q_raw = q_res
            q = np.asarray(q_raw, dtype=np.float64)
            if np.isfinite(q).any():
                actor_action = ACTIONS[int(np.argmax(q))]
        else:
            q = np.asarray(self.model.q_values(spatial, features), dtype=np.float64)
            if np.isfinite(q).any():
                actor_action = ACTIONS[int(np.argmax(q))]

        # Ground-truth physical engine legality mask (stone walls, arena bounds, 0 bombs left)
        q[~mask] = -np.inf

        # Active loop, direct reversal, and revisit penalties (when safe and targets exist)
        if has_targets and not in_danger and mask[:4].any() and self.coordinate_history:
            prev_tile = self.coordinate_history[-1]
            stagnation_factor = 1.0 + 0.15 * getattr(self, '_steps_without_progress', 0)
            for i, (a_name, (dx, dy)) in enumerate(DIRS):
                if not mask[i]:
                    continue
                nx, ny = x + dx, y + dy
                cand_traj = list(self.coordinate_history) + [(x, y), (nx, ny)]

                # A. Detect repeating cycles of any period k in [2, 12]
                is_cycle, cycle_k, reps = detect_cycle(cand_traj, max_k=12)
                if is_cycle:
                    q[i] -= (8.0 + 4.0 * (reps - 2)) * stagnation_factor

                # B. Direct reversal (stepping straight back to previous tile)
                if (nx, ny) == prev_tile:
                    q[i] -= 5.0 * stagnation_factor

                # C. Heatmap revisit count (repels pacing back and forth across corridors)
                visits = sum(1 for p in self.coordinate_history if p == (nx, ny))
                if visits >= 1:
                    q[i] -= (visits * 2.5 + max(0, visits - 2) * 3.0) * stagnation_factor

        if np.isfinite(q).any():
            best = int(np.argmax(q))
        else:
            best = ACTION_TO_IDX['WAIT']
        action = ACTIONS[best]

        # Publish overlay trace for visual Pygame GUI debugging
        if publish_overlay_trace is not None and plan_details is not None and walk_path is not None:
            sorted_indices = np.argsort(-q)
            candidate_traces = []
            for rank, idx in enumerate(sorted_indices):
                cand = plan_details[idx]
                p_acts = cand['actions']
                tiles, blocked, bomb_steps = walk_path(
                    (x, y), p_acts,
                    arena=game_state.get('field'),
                    bombs=game_state.get('bombs')
                )
                score_val = float(q[idx]) if np.isfinite(q[idx]) else -999.0
                c_trace = CandidatePlanTrace(
                    rank=rank,
                    actions=p_acts,
                    tiles=tiles,
                    total_score=score_val,
                    predicted_return=float(cand.get('r0', 0.0)),
                    value_bootstrap=float(cand.get('v1', 0.0)),
                    continuation_probability=1.0,
                    blocked_from=blocked,
                    bomb_steps=bomb_steps,
                    first_action=ACTIONS[idx],
                    first_action_legal=bool(mask[idx]),
                )
                candidate_traces.append(c_trace)

            trace = PlannerOverlayTrace(
                round_id=int(game_state.get('round', 0)),
                env_step=int(game_state.get('step', 0)),
                agent_position=(x, y),
                planner_elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                actor_action=actor_action,
                planner_action=action,
                planner_changed_action=(actor_action != action),
                fallback_used=False,
                candidates=candidate_traces,
            )
            publish_overlay_trace(trace)

    self.coordinate_history.append((x, y))
    if action == 'BOMB' and hasattr(self, 'bomb_history'):
        self.bomb_history.append((x, y))
    dt = time.perf_counter() - t0
    self._think_times.append(dt)
    return action
