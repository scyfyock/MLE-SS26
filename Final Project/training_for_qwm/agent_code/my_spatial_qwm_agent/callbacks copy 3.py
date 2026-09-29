"""Training and inference callbacks for the spatial QWM agent."""

from collections import deque
import json
import logging
import os
import time
from typing import List, Optional, Tuple

import numpy as np

from .model import (
    ACTIONS,
    DIRS,
    agent_path,
    load_model,
    new_model,
    state_to_features,
    state_to_spatial_tensor,
    valid_action_mask,
)
from .path_penalty_model import (
    SmallPathRewardModel,
    load_small_model,
    extract_path_candidate_features,
    compute_heuristic_penalty_targets,
    record_sample,
    flush_samples_to_file,
    set_active_data_file,
)

try:
    from .plan_overlay import (
        CandidatePlanTrace, PlannerOverlayTrace,
        publish_overlay_trace, clear_overlay_trace, walk_path,
    )
except ImportError:
    try:
        from plan_overlay import (
            CandidatePlanTrace, PlannerOverlayTrace,
            publish_overlay_trace, clear_overlay_trace, walk_path,
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
    'freeze_dqn': False,
    'wm_warmup_steps': 400,
    'predicted_wait_penalty': 1.0,
    'predicted_loop_penalty': 2.0,
}


def load_config():
    config = dict(DEFAULT_CONFIG)
    path = agent_path('config.json')
    if os.path.isfile(path):
        with open(path) as config_file:
            config.update(json.load(config_file))
    return config


def setup(self):
    """Load or create the model and initialize callback state."""
    self.logger = getattr(self, 'logger', None) or logging.getLogger('my_spatial_qwm_agent')
    self.cfg = load_config()

    model_name = os.environ.get('MY_QWM_MODEL_FILE') or os.environ.get('MY_WM_MODEL_FILE') or self.cfg['model_file']
    run_dir = os.environ.get('MY_QWM_RUN_DIR') or os.environ.get('MY_WM_RUN_DIR')
    if os.path.isfile(model_name):
        load_path = os.path.abspath(model_name)
        self.model_file = load_path
    elif run_dir:
        run_model_file = os.path.join(run_dir, model_name)
        load_path = run_model_file if os.path.isfile(run_model_file) else agent_path(model_name)
        if getattr(self, 'train', False):
            os.makedirs(run_dir, exist_ok=True)
            self.model_file = run_model_file
        else:
            self.model_file = load_path
    else:
        load_path = agent_path(model_name)
        self.model_file = load_path

    self.coordinate_history = deque(maxlen=40)
    self.bomb_history = deque(maxlen=8)
    self._last_round = None
    self._think_times = []

    tree_search_env = os.environ.get('MY_QWM_TREE_SEARCH') or os.environ.get('MY_WM_TREE_SEARCH')
    self.tree_search = (
        tree_search_env.strip().lower() in ('1', 'true', 'yes')
        if tree_search_env is not None
        else bool(self.cfg.get('tree_search', True))
    )
    self.search_depth = max(1, min(int(os.environ.get(
        'MY_QWM_SEARCH_DEPTH', os.environ.get('MY_WM_SEARCH_DEPTH', self.cfg.get('search_depth', 4)))), 16))
    self.beam_size = max(6, min(int(os.environ.get(
        'MY_QWM_BEAM_SIZE', os.environ.get('MY_WM_BEAM_SIZE', self.cfg.get('beam_size', 12)))), 60))
    self.tree_discount = float(os.environ.get(
        'MY_QWM_TREE_DISCOUNT', os.environ.get('MY_WM_TREE_DISCOUNT', self.cfg.get('tree_discount', 0.12))))
    self.alpha_vq = float(os.environ.get(
        'MY_QWM_ALPHA_VQ', os.environ.get('MY_WM_ALPHA_VQ', self.cfg.get('alpha_vq', 0.45))))
    self.predicted_wait_penalty = float(os.environ.get(
        'MY_QWM_WAIT_PENALTY', self.cfg.get('predicted_wait_penalty', 1.0)))
    self.predicted_loop_penalty = float(os.environ.get(
        'MY_QWM_LOOP_PENALTY', self.cfg.get('predicted_loop_penalty', 2.0)))

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
        if os.path.isfile(load_path):
            self.model = load_model(load_path, need_training=getattr(self, 'train', False), partial=True)
            self.logger.info('Loaded Spatial QWM model from %s', os.path.basename(load_path))
        else:
            self.model = new_model(
                lr=float(os.environ.get('MY_WM_LR', self.cfg['lr'])),
                wm_lr=float(os.environ.get('MY_WM_WM_LR', self.cfg['wm_lr'])),
                gamma=self.cfg['gamma'],
                tau=self.cfg['tau'],
            )
            initial_checkpoint = os.environ.get('MY_WM_INIT_CHECKPOINT') or self.cfg.get('initial_checkpoint')
            if initial_checkpoint:
                initial_path = initial_checkpoint if os.path.isabs(initial_checkpoint) else os.path.abspath(
                    os.path.join(os.path.dirname(__file__), initial_checkpoint))
                if os.path.isfile(initial_path):
                    self.model.load(initial_path, need_training=False, partial=True)

        if getattr(self, 'train', False):
            freeze_env = os.environ.get('MY_WM_FREEZE_DQN')
            should_freeze = (freeze_env == '1') if freeze_env is not None else bool(self.cfg.get('freeze_dqn', True))
            if should_freeze:
                self.model.freeze_dqn()
            else:
                self.model.unfreeze_dqn()

    if not hasattr(self, 'epsilon'):
        self.epsilon = 0.0

    # Small Auxiliary Path Reward Model configuration
    use_small_model_env = os.environ.get('MY_QWM_USE_SMALL_MODEL')
    if use_small_model_env is not None:
        self.use_small_model = use_small_model_env.strip().lower() in ('1', 'true', 'yes')
    else:
        self.use_small_model = bool(self.cfg.get('use_small_model', False))

    small_model_file = os.environ.get('MY_QWM_SMALL_MODEL_FILE') or self.cfg.get('small_model_file', 'small_model.pt')
    small_model_path = small_model_file if os.path.isabs(small_model_file) else agent_path(small_model_file)

    self.small_model = None
    if self.use_small_model:
        if os.path.isfile(small_model_path):
            self.small_model = load_small_model(small_model_path)
            self.logger.info('Loaded small path reward model from %s', os.path.basename(small_model_path))
        else:
            self.logger.warning('Small model enabled but checkpoint not found at %s. Falling back to base scores.', small_model_path)

    # Data collection flag for Phase 1 supervised training
    collect_env = os.environ.get('MY_QWM_COLLECT_PATH_DATA')
    self.collect_path_data = (collect_env is not None and collect_env.strip().lower() in ('1', 'true', 'yes'))
    collect_file = os.environ.get('MY_QWM_COLLECT_PATH_DATA_FILE')
    if collect_file:
        set_active_data_file(collect_file)

    self._last_score = 0
    self._last_crates = None
    self._steps_without_progress = 0
    self._consecutive_stationary_steps = 0


def detect_cycle(sequence, max_k=12):
    """Detect repeating cycles for training reward shaping."""
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


def get_blast_map(arena: np.ndarray, bombs: list) -> np.ndarray:
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


def find_escape_moves(
    arena: np.ndarray,
    bombs: list,
    explosion_map: Optional[np.ndarray],
    pos: tuple,
    others: tuple = (),
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Find escape routes for the training reward shaper."""
    x, y = pos
    blast_map = get_blast_map(arena, bombs)
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


def is_hazard_nearby(
    pos: Tuple[int, int],
    arena: np.ndarray,
    bombs: list,
    explosion_map: Optional[np.ndarray],
    radius: int = 4,
) -> bool:
    """Check nearby hazards for training reward shaping."""
    x, y = pos
    if explosion_map is not None:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                tile_x, tile_y = x + dx, y + dy
                if abs(dx) + abs(dy) <= 2 and 0 <= tile_x < arena.shape[0] and 0 <= tile_y < arena.shape[1]:
                    if explosion_map[tile_x, tile_y] > 0:
                        return True

    blast_map = get_blast_map(arena, bombs)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            tile_x, tile_y = x + dx, y + dy
            if abs(dx) + abs(dy) <= radius and 0 <= tile_x < arena.shape[0] and 0 <= tile_y < arena.shape[1]:
                if blast_map[tile_x, tile_y] < 999:
                    return True
    return False


def check_opponent_trapped_or_threatened(
    arena: np.ndarray,
    bomb_pos: Tuple[int, int],
    bombs: list,
    exp_map: Optional[np.ndarray],
    others_raw: list,
    self_pos: Tuple[int, int],
) -> Tuple[bool, bool, int]:
    """Evaluate bomb threats for training reward shaping."""
    bomb_x, bomb_y = bomb_pos
    hypothetical_bombs = list(bombs) + [((bomb_x, bomb_y), 4)]
    threatened = []

    for other in others_raw:
        if len(other) < 4:
            continue
        other_x, other_y = other[3]
        if other_x == bomb_x and abs(other_y - bomb_y) <= 3:
            direction = 1 if other_y > bomb_y else -1
            blocked = any(arena[bomb_x, tile_y] == -1 for tile_y in range(bomb_y + direction, other_y, direction))
        elif other_y == bomb_y and abs(other_x - bomb_x) <= 3:
            direction = 1 if other_x > bomb_x else -1
            blocked = any(arena[tile_x, bomb_y] == -1 for tile_x in range(bomb_x + direction, other_x, direction))
        else:
            continue
        if not blocked:
            threatened.append((other_x, other_y))

    trapped = False
    restricted = False
    for other_x, other_y in threatened:
        obstacles = tuple(
            other[3] for other in others_raw
            if len(other) >= 4 and other[3] != (other_x, other_y)
        ) + (self_pos,)
        _, escape_routes = find_escape_moves(
            arena, hypothetical_bombs, exp_map, (other_x, other_y), obstacles)
        if not escape_routes:
            trapped = True
            break
        if min(distance for _, distance in escape_routes) >= 3:
            restricted = True
    return trapped, restricted, len(threatened)


def _trace_penalty(actions, tiles, wait_penalty, loop_penalty):
    wait_count = sum(action == 'WAIT' for action in actions)
    repeated_steps = max(0, len(tiles) - len(set(tiles)))
    return wait_count * wait_penalty + repeated_steps * loop_penalty


def act(self, game_state: dict) -> str:
    """Select from model scores, engine legality, and predicted trace penalties."""
    start_time = time.perf_counter()
    if game_state is None:
        return 'WAIT'

    x, y = game_state['self'][3]

    if game_state['step'] == 1 or game_state['round'] != self._last_round:
        self.coordinate_history.clear()
        self._last_round = game_state['round']
        self._steps_without_progress = 0
        self._consecutive_stationary_steps = 0
        if clear_overlay_trace is not None:
            clear_overlay_trace()
    elif self.coordinate_history and self.coordinate_history[-1] == (x, y):
        self._consecutive_stationary_steps += 1
    else:
        self._consecutive_stationary_steps = 0

    current_score = int(game_state['self'][1])
    current_crates = int(np.count_nonzero(game_state['field'] == 1))
    if current_score > self._last_score or (self._last_crates is not None and current_crates < self._last_crates):
        self._steps_without_progress = 0
    else:
        self._steps_without_progress += 1
    self._last_score = current_score
    self._last_crates = current_crates

    spatial = state_to_spatial_tensor(game_state, self.coordinate_history)
    features = state_to_features(game_state, self.coordinate_history)
    valid_mask = valid_action_mask(features)
    plan_details = None

    if self.tree_search:
        scores, plan_details = self.model.tree_search_action_scores(
            spatial, features,
            depth=self.search_depth,
            beam_size=self.beam_size,
            discount=self.tree_discount,
            alpha_vq=self.alpha_vq,
            valid_mask=valid_mask,
            return_details=True,
        )
    else:
        scores = self.model.q_values(spatial, features)

    scores = np.asarray(scores, dtype=np.float64)
    scores[~valid_mask] = -np.inf

    path_data = None
    if plan_details is not None and walk_path is not None:
        path_data = []
        for index, candidate in enumerate(plan_details):
            tiles, blocked, bomb_steps = walk_path(
                (x, y), candidate['actions'],
                arena=game_state.get('field'), bombs=game_state.get('bombs'))
            path_data.append((tiles, blocked, bomb_steps))

    # Phase 1 Data Collection Hook
    if getattr(self, 'collect_path_data', False):
        c_feats = extract_path_candidate_features(
            game_state, plan_details, self.coordinate_history,
            consecutive_stationary_steps=self._consecutive_stationary_steps,
            steps_without_progress=self._steps_without_progress,
            base_scores=scores,
        )
        c_targets = compute_heuristic_penalty_targets(
            game_state, plan_details, self.coordinate_history,
            consecutive_stationary_steps=self._consecutive_stationary_steps,
            steps_without_progress=self._steps_without_progress,
            wait_penalty=self.predicted_wait_penalty,
            loop_penalty=self.predicted_loop_penalty,
        )
        record_sample(c_feats, c_targets)
        if game_state.get('step', 0) % 50 == 0:
            flush_samples_to_file()

    # Score adjustment: Small Neural Model (dynActivation) OR Heuristic rules
    if self.use_small_model and self.small_model is not None:
        path_feats = extract_path_candidate_features(
            game_state, plan_details, self.coordinate_history,
            consecutive_stationary_steps=self._consecutive_stationary_steps,
            steps_without_progress=self._steps_without_progress,
            base_scores=scores,
        )
        predicted_deltas = self.small_model.predict_deltas(path_feats)
        base_scores_arr = scores.copy()
        scores = scores + predicted_deltas
        self._last_step_data = {
            'features': path_feats,
            'base_scores': base_scores_arr,
            'scores': scores.copy(),
            'valid_mask': valid_mask.copy(),
        }
    else:
        # Fallback to heuristic trace penalties when small model is not active
        if plan_details is not None and path_data is not None:
            for index, candidate in enumerate(plan_details):
                tiles, blocked, bomb_steps = path_data[index]
                scores[index] -= _trace_penalty(
                    candidate['actions'], tiles,
                    self.predicted_wait_penalty, self.predicted_loop_penalty)

    candidate_traces = []
    if plan_details is not None:
        if path_data is None and walk_path is not None:
            path_data = []
            for index, candidate in enumerate(plan_details):
                tiles, blocked, bomb_steps = walk_path(
                    (x, y), candidate['actions'],
                    arena=game_state.get('field'), bombs=game_state.get('bombs'))
                path_data.append((tiles, blocked, bomb_steps))

        for rank, index in enumerate(np.argsort(-scores)):
            candidate = plan_details[index]
            tiles, blocked, bomb_steps = path_data[index] if path_data else ([(x, y)], None, [])
            candidate_traces.append(CandidatePlanTrace(
                rank=rank,
                actions=candidate['actions'],
                tiles=tiles,
                total_score=float(scores[index]),
                predicted_return=float(candidate.get('r0', 0.0)),
                value_bootstrap=float(candidate.get('v1', 0.0)),
                blocked_from=blocked,
                bomb_steps=bomb_steps,
                first_action=ACTIONS[index],
                first_action_legal=bool(valid_mask[index]),
            ))

    valid_indices = np.flatnonzero(valid_mask)
    temperature = float(getattr(self, 'exploration_temperature', 0.0))
    if temperature > 0.0 and np.isfinite(scores).any():
        finite_scores = scores.copy()
        finite_scores[~valid_mask] = -1e9
        shift_scores = finite_scores - np.max(finite_scores[valid_mask])
        exp_scores = np.exp(np.clip(shift_scores / temperature, -50.0, 50.0))
        exp_scores[~valid_mask] = 0.0
        sum_exp = np.sum(exp_scores)
        if sum_exp > 0:
            probs = exp_scores / sum_exp
            action_index = int(np.random.choice(len(ACTIONS), p=probs))
        else:
            action_index = int(np.argmax(scores))
    else:
        action_index = int(np.argmax(scores)) if np.isfinite(scores).any() else int(valid_indices[0])
    action = ACTIONS[action_index]

    if hasattr(self, '_last_step_data') and self._last_step_data is not None:
        self._last_step_data['action'] = action
        self._last_step_data['action_index'] = action_index

    if plan_details is not None and publish_overlay_trace is not None:
        publish_overlay_trace(PlannerOverlayTrace(
            round_id=int(game_state.get('round', 0)),
            env_step=int(game_state.get('step', 0)),
            agent_position=(x, y),
            planner_elapsed_ms=(time.perf_counter() - start_time) * 1000.0,
            actor_action=action,
            planner_action=action,
            planner_changed_action=False,
            fallback_used=False,
            candidates=candidate_traces,
        ))

    self.coordinate_history.append((x, y))
    if action == 'BOMB' and hasattr(self, 'bomb_history'):
        self.bomb_history.append((x, y))
    self._think_times.append(time.perf_counter() - start_time)
    return action
