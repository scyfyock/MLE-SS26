"""Inference callbacks for the trained spatial QWM agent."""

from collections import deque
import json
import logging
import os

import numpy as np

from .model import (
    ACTIONS,
    agent_path,
    load_model,
    state_to_features,
    state_to_spatial_tensor,
    valid_action_mask,
)

try:
    from agent_code.my_spatial_qwm_agent.plan_overlay import (
        CandidatePlanTrace,
        PlannerOverlayTrace,
        publish_overlay_trace,
        clear_overlay_trace,
        walk_path,
    )
except ImportError:
    from .plan_overlay import (
        CandidatePlanTrace,
        PlannerOverlayTrace,
        publish_overlay_trace,
        clear_overlay_trace,
        walk_path,
    )


DEFAULT_CONFIG = {
    'model_file': 'my-saved-model-spatial-qwm.pt',
    'tree_search': True,
    'search_depth': 4,
    'beam_size': 12,
    'tree_discount': 0.12,
    'alpha_vq': 0.45,
}


def load_config():
    config = dict(DEFAULT_CONFIG)
    config_path = agent_path('config.json')
    if os.path.isfile(config_path):
        with open(config_path) as config_file:
            config.update(json.load(config_file))
    return config


def setup(self):
    """Load the trained QWM checkpoint and initialize inference state."""
    self.logger = getattr(self, 'logger', None) or logging.getLogger('qwm_agent')
    self.cfg = load_config()

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
    tree_search_env = os.environ.get('MY_QWM_TREE_SEARCH')
    self.tree_search = (
        (tree_search_env.lower() in ('1', 'true', 'yes'))
        if tree_search_env is not None
        else bool(self.cfg.get('tree_search', True))
    )
    self.search_depth = max(1, min(int(os.environ.get('MY_QWM_SEARCH_DEPTH', self.cfg.get('search_depth', 4))), 16))
    self.beam_size = max(6, min(int(os.environ.get('MY_QWM_BEAM_SIZE', self.cfg.get('beam_size', 12))), 60))
    self.tree_discount = float(os.environ.get('MY_QWM_TREE_DISCOUNT', self.cfg.get('tree_discount', 0.12)))
    self.alpha_vq = float(os.environ.get('MY_QWM_ALPHA_VQ', self.cfg.get('alpha_vq', 0.45)))


def act(self, game_state: dict) -> str:
    """Select an action using only the loaded QWM model and engine legality."""
    if game_state is None:
        return 'WAIT'

    if game_state['step'] == 1 or game_state['round'] != self._last_round:
        self.coordinate_history.clear()
        self._last_round = game_state['round']
        clear_overlay_trace()

    spatial = state_to_spatial_tensor(game_state, self.coordinate_history)
    features = state_to_features(game_state, self.coordinate_history)
    valid_mask = valid_action_mask(features)

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
    valid_indices = np.flatnonzero(valid_mask)
    action_index = int(np.argmax(scores)) if np.isfinite(scores).any() else int(valid_indices[0])
    action = ACTIONS[action_index]

    if plan_details is not None:
        x, y = game_state['self'][3]
        candidate_traces = []
        for rank, index in enumerate(np.argsort(-scores)):
            candidate = plan_details[index]
            tiles, blocked, bomb_steps = walk_path(
                (x, y),
                candidate['actions'],
                arena=game_state.get('field'),
                bombs=game_state.get('bombs'),
            )
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
            planner_elapsed_ms=0.0,
            actor_action=action,
            planner_action=action,
            planner_changed_action=False,
            fallback_used=False,
            candidates=candidate_traces,
        ))

    self.coordinate_history.append(game_state['self'][3])
    return action
