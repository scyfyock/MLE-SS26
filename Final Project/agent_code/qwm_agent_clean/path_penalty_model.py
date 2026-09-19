"""Small Neural Path Penalty / Reward Model for my_spatial_qwm_agent.

Provides a very small auxiliary neural network (~5.8k parameters, ~23 KB)
using dynActivation that predicts post-hoc path/action adjustments
(anti-looping, reversal suppression, corridor revisit deterrence,
active idle wait penalties, trace penalties, and suicide trap prevention).

Trained in two phases:
  Phase 1: Supervised regression to imitate Optuna-tuned heuristic penalty values.
  Phase 2: Environment RL fine-tuning (residual policy/value learning) with frozen backbone.
"""

import atexit
from collections import deque
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .model import (
    ACTIONS,
    ACTION_TO_IDX,
    DIRS,
    N_ACTIONS,
    state_to_features,
    valid_action_mask,
    dynActivation,
    F_SAFE_MOVE,
    F_SAFE_WAIT,
    F_IN_DANGER,
    F_TRAPS_SELF,
    F_COIN_DIR,
    F_COIN_PROX,
    F_CRATE_DIR,
    F_BOMBS_LEFT,
    F_BOMB_UTIL,
    F_OPP_DIR,
    F_OPP_PROX,
    F_BLOCKED,
)
try:
    from .plan_overlay import walk_path
except ImportError:
    try:
        from plan_overlay import walk_path
    except ImportError:
        walk_path = None

MOVE_ACTIONS = ACTIONS[:4]
F_NBR_VISITED = 29
F_BIAS = 0

FEATURE_DIM = 56
OPTUNA_WAIT_PENALTY = 1.25
OPTUNA_LOOP_PENALTY = 1.5


class SmallPathRewardModel(nn.Module):
    """A lightweight residual MLP (~5.8k parameters, ~23 KB) that predicts path adjustments using dynActivation."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            dynActivation(),
            nn.Linear(hidden_dim, 32),
            dynActivation(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: tensor of shape (batch_size, FEATURE_DIM) or (batch_size, N_ACTIONS, FEATURE_DIM)
        returns: tensor of shape (batch_size, 1) or (batch_size, N_ACTIONS)
        """
        if x.dim() == 3:
            b, n, d = x.shape
            out = self.net(x.reshape(-1, d)).reshape(b, n)
            return out
        return self.net(x)

    def predict_deltas(self, path_features_np: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        """
        Inference convenience method.
        path_features_np: numpy array of shape (6, FEATURE_DIM)
        returns: numpy array of shape (6,) containing predicted score adjustments.
        """
        self.eval()
        dev = device or next(self.parameters()).device
        with torch.no_grad():
            tensor = torch.as_tensor(path_features_np, dtype=torch.float32, device=dev)
            if tensor.dim() == 2:
                deltas = self.net(tensor).squeeze(-1).cpu().numpy()
            elif tensor.dim() == 1:
                deltas = self.net(tensor.unsqueeze(0)).squeeze().cpu().numpy()
            else:
                deltas = self.forward(tensor).cpu().numpy()
        return deltas


def detect_cycle_helper(sequence: List[Tuple[int, int]], max_k: int = 12) -> Tuple[bool, int, int]:
    """Detect repeating cycles in position history."""
    n = len(sequence)
    for period in range(2, min(max_k + 1, n // 2 + 1)):
        matches = 0
        for repeat_count in range(1, n // period + 1):
            if sequence[-repeat_count * period:] == sequence[-period:] * repeat_count:
                matches = repeat_count
            else:
                break
        if matches >= 2:
            return True, period, matches
    return False, 0, 0


def get_blast_map_helper(arena: np.ndarray, bombs: list) -> np.ndarray:
    """Compute the minimum countdown of any bomb threatening each tile."""
    blast_map = np.full(arena.shape, 999, dtype=np.int32)
    for (bomb_x, bomb_y), timer in bombs:
        blast_map[bomb_x, bomb_y] = min(blast_map[bomb_x, bomb_y], timer)
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            for step in range(1, 4):
                tile_x, tile_y = bomb_x + dx * step, bomb_y + dy * step
                if not (0 <= tile_x < arena.shape[0] and 0 <= tile_y < arena.shape[1]):
                    break
                if arena[tile_x, tile_y] == -1:
                    break
                blast_map[tile_x, tile_y] = min(blast_map[tile_x, tile_y], timer)
    return blast_map


def find_escape_moves_helper(
    arena: np.ndarray,
    bombs: list,
    explosion_map: Optional[np.ndarray],
    pos: Tuple[int, int],
    others: tuple = (),
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Finds shortest escape paths from pos to any safe tile."""
    x, y = pos
    blast_map = get_blast_map_helper(arena, bombs)
    in_danger = bool((blast_map[x, y] < 999) or (explosion_map is not None and explosion_map[x, y] > 0))
    if not in_danger:
        return False, []

    bomb_positions = {bomb[0] for bomb in bombs}
    other_positions = set(others)
    safe_tiles = {
        (tile_x, tile_y)
        for tile_x in range(arena.shape[0])
        for tile_y in range(arena.shape[1])
        if arena[tile_x, tile_y] == 0
        and blast_map[tile_x, tile_y] == 999
        and (explosion_map is None or explosion_map[tile_x, tile_y] == 0)
        and (tile_x, tile_y) not in bomb_positions
        and (tile_x, tile_y) not in other_positions
    }
    if not safe_tiles:
        return True, []

    queue = deque()
    visited = {(x, y)}
    escape_routes = []
    for action_index, (_, (dx, dy)) in enumerate(DIRS):
        next_x, next_y = x + dx, y + dy
        if 0 <= next_x < arena.shape[0] and 0 <= next_y < arena.shape[1]:
            if arena[next_x, next_y] == 0 and (next_x, next_y) not in bomb_positions and (next_x, next_y) not in other_positions:
                if (explosion_map is None or explosion_map[next_x, next_y] == 0) and blast_map[next_x, next_y] > 0:
                    visited.add((next_x, next_y))
                    if (next_x, next_y) in safe_tiles:
                        escape_routes.append((action_index, 1))
                    else:
                        queue.append((next_x, next_y, 1, action_index))
    if escape_routes:
        return True, escape_routes

    minimum_distance = 999
    while queue:
        current_x, current_y, distance, first_action = queue.popleft()
        if distance >= minimum_distance or distance >= 6:
            continue
        for _, (dx, dy) in DIRS:
            next_x, next_y = current_x + dx, current_y + dy
            if 0 <= next_x < arena.shape[0] and 0 <= next_y < arena.shape[1]:
                if arena[next_x, next_y] == 0 and (next_x, next_y) not in bomb_positions and (next_x, next_y) not in other_positions and (next_x, next_y) not in visited:
                    if blast_map[next_x, next_y] > distance:
                        visited.add((next_x, next_y))
                        if (next_x, next_y) in safe_tiles:
                            minimum_distance = distance + 1
                            escape_routes.append((first_action, distance + 1))
                        else:
                            queue.append((next_x, next_y, distance + 1, first_action))
    return True, escape_routes


def is_hazard_nearby_helper(
    pos: Tuple[int, int],
    arena: np.ndarray,
    bombs: list,
    explosion_map: Optional[np.ndarray] = None,
    radius: int = 4,
) -> bool:
    """Check if blast or explosion is within radius of pos."""
    x, y = pos
    if explosion_map is not None:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                tile_x, tile_y = x + dx, y + dy
                if abs(dx) + abs(dy) <= 2 and 0 <= tile_x < arena.shape[0] and 0 <= tile_y < arena.shape[1]:
                    if explosion_map[tile_x, tile_y] > 0:
                        return True

    blast_map = get_blast_map_helper(arena, bombs)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            tile_x, tile_y = x + dx, y + dy
            if abs(dx) + abs(dy) <= radius and 0 <= tile_x < arena.shape[0] and 0 <= tile_y < arena.shape[1]:
                if blast_map[tile_x, tile_y] < 999:
                    return True
    return False


def extract_path_candidate_features(
    game_state: dict,
    plan_details: Optional[List[dict]],
    coordinate_history: deque,
    consecutive_stationary_steps: int = 0,
    steps_without_progress: int = 0,
    base_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extracts a (6, FEATURE_DIM) feature array, one row for each of the 6 root candidate actions/paths.
    """
    arena = game_state['field']
    bombs = game_state.get('bombs', [])
    exp_map = game_state.get('explosion_map')
    raw_others = game_state.get('others', [])
    others = [o[3] for o in raw_others]
    x, y = game_state['self'][3]

    raw_feats = state_to_features(game_state, coordinate_history)
    valid_mask = valid_action_mask(raw_feats)

    in_danger, _ = find_escape_moves_helper(arena, bombs, exp_map, (x, y), tuple(others))
    hazard_nearby = is_hazard_nearby_helper((x, y), arena, bombs, exp_map, radius=4)

    current_crates = int(np.count_nonzero(arena == 1))
    has_targets = bool(current_crates > 0 or len(game_state.get('coins', [])) > 0 or len(raw_others) > 0)

    blast_map = get_blast_map_helper(arena, bombs)
    has_safe_move = any(
        valid_mask[a_i] and (0 <= x + dx < arena.shape[0] and 0 <= y + dy < arena.shape[1])
        and blast_map[x + dx, y + dy] == 999 and (exp_map is None or exp_map[x + dx, y + dy] == 0)
        for a_i, (_, (dx, dy)) in enumerate(DIRS)
    )

    hypo_bombs = list(bombs) + [((x, y), 4)]
    _, bomb_escapes = find_escape_moves_helper(arena, hypo_bombs, exp_map, (x, y), tuple(others))
    bomb_traps_self = not bool(bomb_escapes)

    prev_tile = coordinate_history[-1] if coordinate_history else None
    stagnation_factor = 1.0 + 0.15 * steps_without_progress

    # Selected compact subset of state features (26 dims)
    state_sub = np.array([
        raw_feats[F_SAFE_MOVE], raw_feats[F_SAFE_MOVE + 1], raw_feats[F_SAFE_MOVE + 2], raw_feats[F_SAFE_MOVE + 3],
        raw_feats[F_SAFE_WAIT], raw_feats[F_IN_DANGER], raw_feats[F_TRAPS_SELF], raw_feats[F_BOMBS_LEFT],
        raw_feats[F_BOMB_UTIL],
        raw_feats[F_COIN_DIR], raw_feats[F_COIN_DIR + 1], raw_feats[F_COIN_DIR + 2], raw_feats[F_COIN_DIR + 3],
        raw_feats[F_COIN_PROX],
        raw_feats[F_CRATE_DIR], raw_feats[F_CRATE_DIR + 1], raw_feats[F_CRATE_DIR + 2], raw_feats[F_CRATE_DIR + 3],
        raw_feats[F_OPP_DIR], raw_feats[F_OPP_DIR + 1], raw_feats[F_OPP_DIR + 2], raw_feats[F_OPP_DIR + 3],
        raw_feats[F_OPP_PROX],
        raw_feats[F_BLOCKED],
        raw_feats[F_NBR_VISITED],
        raw_feats[F_BIAS],
    ], dtype=np.float32)

    path_data = []
    if plan_details is not None:
        for idx, cand in enumerate(plan_details):
            tiles, blocked, bomb_steps = walk_path(
                (x, y), cand['actions'], arena=arena, bombs=bombs)
            path_data.append((cand['actions'], tiles, blocked, bomb_steps))
    else:
        for idx, act in enumerate(ACTIONS):
            tiles, blocked, bomb_steps = walk_path(
                (x, y), [act], arena=arena, bombs=bombs)
            path_data.append(([act], tiles, blocked, bomb_steps))

    feature_matrix = np.zeros((N_ACTIONS, FEATURE_DIM), dtype=np.float32)

    for k in range(N_ACTIONS):
        acts, tiles, blocked, b_steps = path_data[k]
        target_pos = tiles[1] if len(tiles) > 1 else (x, y)

        # 1. Root action one-hot (6 dims)
        one_hot = np.zeros(6, dtype=np.float32)
        one_hot[k] = 1.0

        # 2. Path Trajectory Properties (8 dims)
        n_tiles = max(1, len(tiles))
        n_acts = max(1, len(acts))
        wait_ratio = sum(a == 'WAIT' for a in acts) / n_acts
        bomb_ratio = sum(a == 'BOMB' for a in acts) / n_acts
        repeat_ratio = max(0, len(tiles) - len(set(tiles))) / n_tiles
        is_blocked = 1.0 if blocked is not None else 0.0
        blocked_ratio = (blocked / n_acts) if blocked is not None else 0.0
        dx_disp = (tiles[-1][0] - x) / 10.0
        dy_disp = (tiles[-1][1] - y) / 10.0
        path_props = np.array([
            len(tiles) / 8.0, wait_ratio, bomb_ratio, repeat_ratio,
            is_blocked, blocked_ratio, dx_disp, dy_disp
        ], dtype=np.float32)

        # 3. History & Temporal Context (8 dims)
        is_reversal = 1.0 if (prev_tile is not None and target_pos == prev_tile and k < 4) else 0.0
        revisits = sum(1 for p in coordinate_history if p == target_pos) if k < 4 else 0
        revisit_norm = min(revisits / 5.0, 2.0)

        cand_traj = list(coordinate_history) + [(x, y), target_pos]
        is_cycle, _, cycle_reps = detect_cycle_helper(cand_traj, max_k=12) if k < 4 else (False, 0, 0)
        cycle_flag = 1.0 if is_cycle else 0.0
        cycle_reps_norm = min(cycle_reps / 5.0, 2.0)

        stat_norm = min(consecutive_stationary_steps / 5.0, 2.0)
        prog_norm = min(steps_without_progress / 10.0, 3.0)
        is_wait = 1.0 if k == ACTION_TO_IDX['WAIT'] else 0.0

        hist_props = np.array([
            is_reversal, revisit_norm, cycle_flag, cycle_reps_norm,
            stat_norm, prog_norm, stagnation_factor / 3.0, is_wait
        ], dtype=np.float32)

        # 4. Local Physical & Tactical Context (7 dims)
        danger_flag = 1.0 if in_danger else 0.0
        hazard_flag = 1.0 if hazard_nearby else 0.0
        targets_flag = 1.0 if has_targets else 0.0
        safe_move_flag = 1.0 if has_safe_move else 0.0
        trap_flag = 1.0 if bomb_traps_self else 0.0
        is_bomb = 1.0 if k == ACTION_TO_IDX['BOMB'] else 0.0
        legal_flag = 1.0 if valid_mask[k] else 0.0

        tactical_props = np.array([
            danger_flag, hazard_flag, targets_flag, safe_move_flag,
            trap_flag, is_bomb, legal_flag
        ], dtype=np.float32)

        # 5. Base score feature (1 dim)
        base_score_val = (base_scores[k] / 10.0) if (base_scores is not None and np.isfinite(base_scores[k])) else 0.0
        score_feat = np.array([base_score_val], dtype=np.float32)

        # Concatenate: 6 + 8 + 8 + 7 + 1 + 26 = 56 dims
        feat_vec = np.concatenate([one_hot, path_props, hist_props, tactical_props, score_feat, state_sub])
        feature_matrix[k] = feat_vec

    return feature_matrix


def compute_heuristic_penalty_targets(
    game_state: dict,
    plan_details: Optional[List[dict]],
    coordinate_history: deque,
    consecutive_stationary_steps: int = 0,
    steps_without_progress: int = 0,
    wait_penalty: float = OPTUNA_WAIT_PENALTY,
    loop_penalty: float = OPTUNA_LOOP_PENALTY,
) -> np.ndarray:
    """
    Computes the exact ground-truth heuristic penalty targets (clipped to [-40.0, 0.0])
    using the Optuna-tuned coefficients (wait_penalty=1.25, loop_penalty=1.5)
    found in my_spatial_qwm_agent/optuna_study.db.
    """
    arena = game_state['field']
    bombs = game_state.get('bombs', [])
    exp_map = game_state.get('explosion_map')
    raw_others = game_state.get('others', [])
    others = [o[3] for o in raw_others]
    x, y = game_state['self'][3]

    raw_feats = state_to_features(game_state, coordinate_history)
    valid_mask = valid_action_mask(raw_feats)

    in_danger, _ = find_escape_moves_helper(arena, bombs, exp_map, (x, y), tuple(others))
    hazard_nearby = is_hazard_nearby_helper((x, y), arena, bombs, exp_map, radius=4)

    current_crates = int(np.count_nonzero(arena == 1))
    has_targets = bool(current_crates > 0 or len(game_state.get('coins', [])) > 0 or len(raw_others) > 0)

    blast_map = get_blast_map_helper(arena, bombs)
    has_safe_move = any(
        valid_mask[a_i] and (0 <= x + dx < arena.shape[0] and 0 <= y + dy < arena.shape[1])
        and blast_map[x + dx, y + dy] == 999 and (exp_map is None or exp_map[x + dx, y + dy] == 0)
        for a_i, (_, (dx, dy)) in enumerate(DIRS)
    )

    hypo_bombs = list(bombs) + [((x, y), 4)]
    _, bomb_escapes = find_escape_moves_helper(arena, hypo_bombs, exp_map, (x, y), tuple(others))
    bomb_traps_self = not bool(bomb_escapes)

    prev_tile = coordinate_history[-1] if coordinate_history else None
    stagnation_factor = 1.0 + 0.15 * steps_without_progress

    targets = np.zeros(N_ACTIONS, dtype=np.float32)

    # 1. Idle wait penalty (Optuna / safety heuristic)
    if has_targets and not in_danger and not hazard_nearby and has_safe_move:
        targets[ACTION_TO_IDX['WAIT']] -= 15.0
    elif has_targets and not in_danger and consecutive_stationary_steps >= 2 and has_safe_move:
        targets[ACTION_TO_IDX['WAIT']] -= 10.0

    # 2. Suicide bomb penalty
    if bomb_traps_self:
        targets[ACTION_TO_IDX['BOMB']] -= 20.0

    # 3. Movement anti-oscillation penalties
    if has_targets and not in_danger and coordinate_history:
        for i, (a_name, (dx, dy)) in enumerate(DIRS):
            nx, ny = x + dx, y + dy
            cand_traj = list(coordinate_history) + [(x, y), (nx, ny)]

            # A. Repeating cycle penalty
            is_cycle, _, reps = detect_cycle_helper(cand_traj, max_k=12)
            if is_cycle:
                targets[i] -= (8.0 + 4.0 * (reps - 2)) * stagnation_factor

            # B. Direct reversal penalty
            if prev_tile is not None and (nx, ny) == prev_tile:
                targets[i] -= 5.0 * stagnation_factor

            # C. Heatmap revisit count penalty
            visits = sum(1 for p in coordinate_history if p == (nx, ny))
            if visits >= 1:
                targets[i] -= (visits * 2.5 + max(0, visits - 2) * 3.0) * stagnation_factor

    # 4. Optuna rollout trace penalties (predicted_wait_penalty & predicted_loop_penalty)
    if plan_details is not None:
        for idx, cand in enumerate(plan_details):
            tiles, blocked, _ = walk_path(
                (x, y), cand['actions'], arena=arena, bombs=bombs)
            wait_count = sum(a == 'WAIT' for a in cand['actions'])
            repeated_steps = max(0, len(tiles) - len(set(tiles)))
            targets[idx] -= (wait_count * wait_penalty + repeated_steps * loop_penalty)

    targets = np.clip(targets, -40.0, 0.0)
    return targets


def save_small_model(model: SmallPathRewardModel, filepath: str):
    """Save small model weights to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': model.input_dim,
    }, filepath)


def load_small_model(filepath: str, device: str = 'cpu') -> SmallPathRewardModel:
    """Load small model weights from disk."""
    checkpoint = torch.load(filepath, map_location=device)
    input_dim = checkpoint.get('input_dim', FEATURE_DIM)
    model = SmallPathRewardModel(input_dim=input_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


_DATA_BUFFER_X: List[np.ndarray] = []
_DATA_BUFFER_Y: List[np.ndarray] = []
_ACTIVE_DATA_FILE: Optional[str] = None


def record_sample(feats: np.ndarray, targets: np.ndarray):
    """feats: (6, FEATURE_DIM), targets: (6,)"""
    _DATA_BUFFER_X.append(feats.astype(np.float32))
    _DATA_BUFFER_Y.append(targets.astype(np.float32))


def flush_samples_to_file(filepath: Optional[str] = None):
    """Flushes buffered (x, y) samples to a compressed npz file."""
    global _DATA_BUFFER_X, _DATA_BUFFER_Y, _ACTIVE_DATA_FILE
    target_path = filepath or _ACTIVE_DATA_FILE
    if not target_path or not _DATA_BUFFER_X:
        return

    new_x = np.stack(_DATA_BUFFER_X, axis=0)  # (N, 6, FEATURE_DIM)
    new_y = np.stack(_DATA_BUFFER_Y, axis=0)  # (N, 6)

    if os.path.isfile(target_path):
        try:
            with np.load(target_path) as data:
                old_x = data['x']
                old_y = data['y']
            new_x = np.concatenate([old_x, new_x], axis=0)
            new_y = np.concatenate([old_y, new_y], axis=0)
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    np.savez_compressed(target_path, x=new_x, y=new_y)
    _DATA_BUFFER_X.clear()
    _DATA_BUFFER_Y.clear()


def set_active_data_file(filepath: str):
    global _ACTIVE_DATA_FILE
    _ACTIVE_DATA_FILE = filepath


atexit.register(flush_samples_to_file)
