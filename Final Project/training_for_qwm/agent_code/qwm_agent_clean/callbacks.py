"""Inference callbacks for the trained spatial QWM agent (Clean Architecture).

Uses ONLY:
  1. Main Model: SpatialDuelingDQN + LatentWorldModel (Tree Search or Q-values)
  2. Small Model: SmallPathRewardModel (~5.8k parameters, dynActivation)
  3. Game Physics Legality Mask: valid_action_mask(features)

Zero hardcoded policies, zero heuristic BFS escape overrides, zero manual ban filters.
"""

from collections import deque
import json
import logging
import os
import time
from typing import Optional, Tuple

import numpy as np

from .model import (
    ACTIONS,
    ACTION_TO_IDX,
    DIRS,
    agent_path,
    load_model,
    state_to_features,
    state_to_spatial_tensor,
    valid_action_mask,
)
from .path_penalty_model import (
    SmallPathRewardModel,
    load_small_model,
    extract_path_candidate_features,
)

try:
    from .plan_overlay import (
        CandidatePlanTrace,
        PlannerOverlayTrace,
        publish_overlay_trace,
        clear_overlay_trace,
        walk_path,
    )
except (ImportError, ValueError):
    try:
        from plan_overlay import (
            CandidatePlanTrace,
            PlannerOverlayTrace,
            publish_overlay_trace,
            clear_overlay_trace,
            walk_path,
        )
    except (ImportError, ValueError):
        CandidatePlanTrace = None
        PlannerOverlayTrace = None
        publish_overlay_trace = None
        clear_overlay_trace = lambda: None
        walk_path = None


DEFAULT_CONFIG = {
    'model_file': 'my-saved-model-spatial-qwm.pt',
    'tree_search': True,
    'search_depth': 5,
    'beam_size': 18,
    'tree_discount': 0.08,
    'alpha_vq': 0.16,
    'use_small_model': True,
    'small_model_file': 'small_model.pt',
}


def load_config():
    config = dict(DEFAULT_CONFIG)
    config_path = agent_path('config.json')
    if os.path.isfile(config_path):
        with open(config_path) as config_file:
            config.update(json.load(config_file))
    return config


def setup(self):
    """Load the trained QWM checkpoint and small path reward model."""
    self.logger = getattr(self, 'logger', None) or logging.getLogger('qwm_agent_clean')
    self.cfg = load_config()

    # 1. Main QWM Model (Spatial Dueling DQN + Latent World Model)
    model_name = os.environ.get('MY_QWM_MODEL_FILE', self.cfg['model_file'])
    model_path = os.path.abspath(model_name) if os.path.isfile(model_name) else agent_path(model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained QWM checkpoint not found: {model_path}. "
            "Set MY_QWM_MODEL_FILE to a trained checkpoint."
        )

    self.model = load_model(model_path, need_training=False, partial=False)
    self.model_file = model_path
    self.logger.info("Loaded trained Spatial QWM model from %s", os.path.basename(model_path))

    self.coordinate_history = deque(maxlen=40)
    self._last_round = None
    self._consecutive_stationary_steps = 0
    self._steps_without_progress = 0
    self._last_score = 0
    self._last_crates = None

    tree_search_env = os.environ.get('MY_QWM_TREE_SEARCH')
    self.tree_search = (
        (tree_search_env.lower() in ('1', 'true', 'yes'))
        if tree_search_env is not None
        else bool(self.cfg.get('tree_search', True))
    )
    self.search_depth = max(1, min(int(os.environ.get('MY_QWM_SEARCH_DEPTH', self.cfg.get('search_depth', 5))), 16))
    self.beam_size = max(6, min(int(os.environ.get('MY_QWM_BEAM_SIZE', self.cfg.get('beam_size', 18))), 60))
    self.tree_discount = float(os.environ.get('MY_QWM_TREE_DISCOUNT', self.cfg.get('tree_discount', 0.08)))
    self.alpha_vq = float(os.environ.get('MY_QWM_ALPHA_VQ', self.cfg.get('alpha_vq', 0.16)))

    # 2. Small Neural Path Reward Model (~5.8k parameters, dynActivation)
    use_sm_env = os.environ.get('MY_QWM_USE_SMALL_MODEL')
    self.use_small_model = (
        (use_sm_env.lower() in ('1', 'true', 'yes'))
        if use_sm_env is not None
        else bool(self.cfg.get('use_small_model', True))
    )
    sm_file = os.environ.get('MY_QWM_SMALL_MODEL_FILE', self.cfg.get('small_model_file', 'small_model.pt'))
    sm_path = os.path.abspath(sm_file) if os.path.isfile(sm_file) else agent_path(sm_file)
    if self.use_small_model and os.path.isfile(sm_path):
        self.small_model = load_small_model(sm_path)
        self.logger.info("Loaded Small Path Reward Model (dynActivation) from %s", os.path.basename(sm_path))
    else:
        self.small_model = None
        if self.use_small_model:
            self.logger.warning("Small model file not found at %s.", sm_path)

    self.small_model_weight = float(
        os.environ.get('MY_QWM_SMALL_MODEL_WEIGHT', self.cfg.get('small_model_weight', 0.25))
    )


def act(self, game_state: dict) -> str:
    """
    Select an action using purely neural model predictions and physical rules validity.
    
    1. Main Model computes Q-values / Tree Search values over legal moves.
    2. Small Model predicts learned trajectory adjustments (anti-looping, reversal, wait penalties).
    3. Action is chosen as argmax(scores + predicted_deltas) over physically legal actions.
    """
    start_time = time.perf_counter()
    if game_state is None:
        return 'WAIT'

    # Round reset & coordinate history management
    if game_state['step'] == 1 or game_state['round'] != self._last_round:
        self.coordinate_history.clear()
        self._last_round = game_state['round']
        self._consecutive_stationary_steps = 0
        self._steps_without_progress = 0
        self._last_score = game_state['self'][1]
        self._last_crates = int(np.count_nonzero(game_state['field'] == 1))
        if clear_overlay_trace is not None:
            clear_overlay_trace()

    x, y = game_state['self'][3]

    if self.coordinate_history and self.coordinate_history[-1] == (x, y):
        self._consecutive_stationary_steps += 1
    else:
        self._consecutive_stationary_steps = 0

    current_score = game_state['self'][1]
    current_crates = int(np.count_nonzero(game_state['field'] == 1))
    if current_score > self._last_score or (self._last_crates is not None and current_crates < self._last_crates):
        self._steps_without_progress = 0
    else:
        self._steps_without_progress += 1
    self._last_score = current_score
    self._last_crates = current_crates

    # Extract state representation and physical legality mask
    spatial = state_to_spatial_tensor(game_state, self.coordinate_history)
    features = state_to_features(game_state, self.coordinate_history)
    valid_mask = valid_action_mask(features)

    # 1. Main Model: QWM Tree Search Lookahead or Direct Q-values
    if self.tree_search:
        scores, plan_details = self.model.tree_search_action_scores(
            spatial,
            features,
            depth=self.search_depth,
            beam_size=self.beam_size,
            discount=self.tree_discount,
            alpha_vq=self.alpha_vq,
            valid_mask=valid_mask,
            return_details=True,
        )
    else:
        scores = self.model.q_values(spatial, features)
        plan_details = None

    scores = np.asarray(scores, dtype=np.float64)
    scores[~valid_mask] = -np.inf

    # 2. Small Model: Neural Path Reward / Penalty Prediction
    if self.use_small_model and self.small_model is not None:
        path_feats = extract_path_candidate_features(
            game_state,
            plan_details,
            self.coordinate_history,
            consecutive_stationary_steps=self._consecutive_stationary_steps,
            steps_without_progress=self._steps_without_progress,
            base_scores=scores,
        )
        predicted_deltas = self.small_model.predict_deltas(path_feats)
        scores = scores + self.small_model_weight * predicted_deltas

    # 3. Action Selection: Argmax over physically valid actions
    valid_indices = np.flatnonzero(valid_mask)
    action_index = int(np.argmax(scores)) if np.isfinite(scores).any() else int(valid_indices[0])
    action = ACTIONS[action_index]

    # Publish plan overlay trace for optional GUI debugging visualization
    if plan_details is not None and publish_overlay_trace is not None and CandidatePlanTrace is not None:
        candidate_traces = []
        path_data = []
        if walk_path is not None:
            for index, candidate in enumerate(plan_details):
                tiles, blocked, bomb_steps = walk_path(
                    (x, y),
                    candidate['actions'],
                    arena=game_state.get('field'),
                    bombs=game_state.get('bombs'),
                )
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
    return action
