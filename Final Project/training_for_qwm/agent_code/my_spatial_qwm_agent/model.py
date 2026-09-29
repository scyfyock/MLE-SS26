"""
model.py -- Spatial QWM Architecture (Q-Learning with Latent World Models).

Combines:
  1. Spatial Dueling DQN: CNN + 33-dim physics features -> 256-dim Fusion layer -> Dueling Value/Advantage heads.
  2. DynActivation: Learnable activation function F.mish(x)*(alpha - beta) + beta*x.
  3. Latent World Model: Small predictive neural network predicting the next 256-dim fused state
     z_{t+1} = z_t + Delta_psi(z_t, a_t) and step reward r_psi(z_t, a_t).
  4. QWM Tree Search: Test-time lookahead planning over actions as formulated in Dong et al. (Stanford, 2026).
  5. Flexible Model Loading: Partial weight loading from existing spatial DQN checkpoints.
"""

from collections import deque
import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AGENT_DIR = Path(__file__).resolve().parent

# Re-use deterministic spatial & feature representations
try:
    from agent_code.my_spatial_dqn_agent.model import (
        ACTIONS,
        ACTION_TO_IDX,
        DIRS,
        N_ACTIONS,
        BOARD_W,
        BOARD_H,
        N_SPATIAL_CHANNELS,
        N_FEATURES,
        state_to_spatial_tensor,
        state_to_features,
        valid_action_mask,
        rotate_spatial,
        mirror_spatial,
        augment_spatial_transition,
        get_device,
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
        F_BLOCKED,
    )
except ImportError:
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
    DIRS = ACTIONS[:4]
    N_ACTIONS = len(ACTIONS)
    BOARD_W = 17
    BOARD_H = 17
    N_SPATIAL_CHANNELS = 10
    N_FEATURES = 33
    F_BOMBS_LEFT = 17
    F_BLOCKED = 23

    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def agent_path(*parts) -> str:
    return str(AGENT_DIR.joinpath(*parts))


# ============================================================================
# 1. DynActivation (Adaptive Non-Linear Activation)
# ============================================================================
class DynActivation(nn.Module):
    """
    Custom learnable activation function:
    forward(x) = F.mish(x) * (alpha - beta) + beta * x
    Initializes with alpha=1.0, beta=0.0 (exact Mish activation).
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(x) * (self.alpha - self.beta) + self.beta * x


# Aliases for convenience
dynActivation = DynActivation
dynActvation = DynActivation


# ============================================================================
# 2. Flexible Partial Weight Loader Helper
# ============================================================================
def load_matching_weights(model: nn.Module, state_dict: dict, partial: bool = True) -> Tuple[List[str], List[str]]:
    """
    Loads matching parameters from state_dict into model.
    When partial=True, all keys present in state_dict with matching shapes are loaded,
    and all other model parameters remain at their freshly initialized values.
    Returns (loaded_keys, uninitialized_keys).
    """
    model_dict = model.state_dict()
    matched = {}
    loaded_keys = []
    uninitialized_keys = []

    for k, v in model_dict.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            matched[k] = state_dict[k]
            loaded_keys.append(k)
        else:
            matched[k] = v
            uninitialized_keys.append(k)

    if partial:
        model.load_state_dict(matched)
    else:
        model.load_state_dict(state_dict)

    return loaded_keys, uninitialized_keys


# ============================================================================
# 3. Spatial Dueling DQN with Fused Representation Extraction
# ============================================================================
class SpatialDuelingDQN(nn.Module):
    """
    Dueling Q-Network combining a 10-channel 17x17 CNN with 33-dim physics features.
    Provides encode_fused() to produce the 256-dim latent state for the World Model.
    """
    def __init__(
        self,
        in_channels: int = N_SPATIAL_CHANNELS,
        n_features: int = N_FEATURES,
        n_actions: int = N_ACTIONS,
        fused_dim: int = 256,
    ):
        super().__init__()
        self.fused_dim = fused_dim

        # Conv backbone
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
        )

        # 32 * 17 * 17 = 9248 -> 256
        self.spatial_proj = nn.Sequential(
            nn.Linear(32 * BOARD_W * BOARD_H, 256),
            nn.ReLU(),
        )

        # Fusion layer: spatial embedding (256) + deterministic features (33) -> 256
        self.fusion = nn.Sequential(
            nn.Linear(256 + n_features, fused_dim),
            nn.ReLU(),
        )

        # Dueling streams
        self.val_stream = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def encode_fused(self, spatial: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Encodes raw inputs into the 256-dim fused latent vector."""
        b = spatial.size(0)
        c = self.conv(spatial).view(b, -1)
        s_embed = self.spatial_proj(c)
        fused = self.fusion(torch.cat([s_embed, features], dim=1))
        return fused

    def forward_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        """Computes Q(z, a) from the fused latent vector z."""
        val = self.val_stream(fused)
        adv = self.adv_stream(fused)
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q

    def forward(self, spatial: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Full forward pass from raw inputs to Q(s, a)."""
        fused = self.encode_fused(spatial, features)
        return self.forward_from_fused(fused)


# ============================================================================
# 4. Latent World Model (Transition & Reward Predictor)
# ============================================================================
class LatentWorldModel(nn.Module):
    """
    Action-conditioned latent dynamics model predicting next fused representation
    and step reward:
      z_{t+1} = z_t + Delta_psi(z_t, a_t)
      r_{t}   = r_psi(z_t, a_t)
    """
    def __init__(self, fused_dim: int = 256, n_actions: int = N_ACTIONS, hidden_dim: int = 256):
        super().__init__()
        self.fused_dim = fused_dim
        self.n_actions = n_actions

        # Action embedding: one-hot or linear projection
        in_dim = fused_dim + n_actions

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            DynActivation(),
            nn.Linear(hidden_dim, hidden_dim),
            DynActivation(),
        )

        # Residual next-state head
        self.trans_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            DynActivation(),
            nn.Linear(hidden_dim, fused_dim),
        )

        # Predicting heads:
        # Head 1: Specific turn reward head (predicts immediate reward for that turn)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            DynActivation(),
            nn.Linear(64, 1),
        )

        # Head 2: Long-term value head (predicts cumulative discounted value of move/sequence)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            DynActivation(),
            nn.Linear(128, 1),
        )

        # Head 3: Future legal moves head (predicts valid action logits for next state)
        self.legal_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            DynActivation(),
            nn.Linear(128, n_actions),
        )

    def _prepare_input(self, z: torch.Tensor, action: Any) -> torch.Tensor:
        """Combines fused representation z with action one-hot."""
        if isinstance(action, int):
            a_idx = torch.tensor([action], device=z.device, dtype=torch.int64)
        elif isinstance(action, np.ndarray):
            a_idx = torch.as_tensor(action, device=z.device, dtype=torch.int64)
        elif isinstance(action, torch.Tensor):
            a_idx = action.to(device=z.device, dtype=torch.int64)
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        if a_idx.dim() == 0:
            a_idx = a_idx.unsqueeze(0)
        if a_idx.dim() == 2 and a_idx.size(1) == 1:
            a_idx = a_idx.squeeze(1)

        a_one_hot = F.one_hot(a_idx, num_classes=self.n_actions).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Match batch sizes if needed (e.g. 1 state evaluated against multiple actions)
        if z.size(0) == 1 and a_one_hot.size(0) > 1:
            z = z.expand(a_one_hot.size(0), -1)

        return torch.cat([z, a_one_hot], dim=-1)

    def forward(self, z: torch.Tensor, action: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns (next_z, pred_reward, pred_value, pred_legal_logits):
          - next_z: Predicted next fused representation (for chaining across horizons)
          - pred_reward: Specific turn reward for this step
          - pred_value: Long-term value estimate of the move and sequence
          - pred_legal_logits: Predicted action legality logits for next state (6-dim)
        """
        x = self._prepare_input(z, action)
        h = self.trunk(x)
        delta_z = self.trans_head(h)
        # Residual update: z_{t+1} = z_t + Delta_z
        next_z = (z if z.size(0) == delta_z.size(0) else z.expand(delta_z.size(0), -1)) + delta_z
        pred_rew = self.reward_head(h)
        pred_val = self.value_head(h)
        pred_legal = self.legal_head(next_z)
        return next_z, pred_rew, pred_val, pred_legal

    @torch.no_grad()
    def predict_next(self, z: torch.Tensor, action: Any) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
        """Single-step inference returning (next_z, float_reward, float_value, pred_legal_logits)."""
        next_z, pred_rew, pred_val, pred_legal = self.forward(z, action)
        return next_z, float(pred_rew.squeeze().item()), float(pred_val.squeeze().item()), pred_legal

    def predict_legal(self, z: torch.Tensor) -> torch.Tensor:
        """Predicts 6-action legality logits for any latent state representation z."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.legal_head(z)


def get_action_legality_targets(features: torch.Tensor) -> torch.Tensor:
    """
    Extracts ground truth (B, 6) binary action legality float tensor directly from features:
      - Actions 0..3 (UP, RIGHT, DOWN, LEFT): 1.0 if not blocked (< 0.5), else 0.0
      - Action 4 (WAIT): 1.0 (always legal)
      - Action 5 (BOMB): 1.0 if bombs_left > 0.5, else 0.0
    """
    if features.dim() == 1:
        features = features.unsqueeze(0)
    B = features.size(0)
    legal = torch.ones((B, N_ACTIONS), device=features.device, dtype=torch.float32)
    legal[:, :4] = (features[:, F_BLOCKED:F_BLOCKED + 4] < 0.5).float()
    legal[:, ACTION_TO_IDX['BOMB']] = (features[:, F_BOMBS_LEFT] > 0.5).float()
    return legal


def amp_autocast(device_type: str = "cuda", enabled: bool = True):
    """Clean forward-compatible AMP autocast context manager."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(device_type: str = "cuda"):
    """Forward-compatible AMP GradScaler creation."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type)
    return torch.cuda.amp.GradScaler()


# ============================================================================
# 5. Spatial QWM Agent (Q-Learning + Latent World Model + Tree Search)
# ============================================================================
class SpatialQWMAgent:
    """
    Complete QWM Agent implementing:
      - Double Dueling DQN policy & target networks
      - Latent World Model with residual dynamics
      - Phased freezing of DQN backbone
      - QWM test-time Tree Search over actions
    """
    def __init__(
        self,
        lr: float = 1e-4,
        wm_lr: float = 3e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        device: Optional[torch.device] = None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.wm_lr = wm_lr
        self.device = device or get_device()
        self.kind = "spatial_qwm"

        self.policy_net = SpatialDuelingDQN().to(self.device)
        self.target_net = SpatialDuelingDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.world_model = LatentWorldModel().to(self.device)

        self.dqn_optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.lr, weight_decay=1e-4)
        self.wm_optimizer = torch.optim.AdamW(self.world_model.parameters(), lr=self.wm_lr, weight_decay=1e-4)

        self._dqn_frozen = False

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            self.scaler_wm = make_grad_scaler("cuda")
            self.scaler_dqn = make_grad_scaler("cuda")
            self.scaler_wm_path = make_grad_scaler("cuda")
        else:
            self.scaler_wm = None
            self.scaler_dqn = None
            self.scaler_wm_path = None
            if self.device.type == "cpu":
                torch.set_num_threads(1)

    # -------------------------------------------------------------------------
    # Freezing & Unfreezing
    # -------------------------------------------------------------------------
    def freeze_dqn(self):
        """Freeze DQN backbone parameters to protect existing Q-policy during WM warmup."""
        self._dqn_frozen = True
        for p in self.policy_net.parameters():
            p.requires_grad = False
        self.policy_net.eval()

    def unfreeze_dqn(self):
        """Unfreeze DQN backbone for joint end-to-end learning."""
        self._dqn_frozen = False
        for p in self.policy_net.parameters():
            p.requires_grad = True

    @property
    def is_dqn_frozen(self) -> bool:
        return self._dqn_frozen

    def set_lr(self, lr: float):
        self.lr = lr
        for param_group in self.dqn_optimizer.param_groups:
            param_group['lr'] = lr

    def set_wm_lr(self, wm_lr: float):
        self.wm_lr = wm_lr
        for param_group in self.wm_optimizer.param_groups:
            param_group['lr'] = wm_lr

    # -------------------------------------------------------------------------
    # Inference: Direct Q-values and QWM Tree Search
    # -------------------------------------------------------------------------
    def q_values(self, spatial: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Standard Q-value evaluation without tree search."""
        self.policy_net.eval()
        with torch.no_grad():
            s = torch.as_tensor(spatial, dtype=torch.float32, device=self.device)
            f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            if s.dim() == 3: s = s.unsqueeze(0)
            if f.dim() == 1: f = f.unsqueeze(0)
            q = self.policy_net(s, f)
            return q.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def tree_search_action_scores(
        self,
        spatial: np.ndarray,
        features: np.ndarray,
        depth: int = 8,
        beam_size: int = 12,
        discount: float = 0.1,
        alpha_vq: float = 0.5,
        valid_mask: Optional[np.ndarray] = None,
        return_details: bool = False,
    ) -> Any:
        """
        Deep QWM Tree Search with Top-K Beam Pruning (Dong et al., Stanford 2026, Section 4).
        Performs multi-step imagined rollouts up to depth D (e.g. 1 to 8 steps) in the latent
        space of the World Model M_psi. Prunes candidate branches to the top-K beams per root
        action to guarantee full action coverage without exponential latency explosion (<5ms).

        Backward value aggregation follows Eq. (8) & (9) of Dong et al. (2026):
          V_r(d | z_d) = r_d + lambda * V(d+1 | z_{d+1})
          V(d | z_d)   = alpha_vq * V_Q(d | z_d) + (1 - alpha_vq) * V_r(d | z_d)
          Q_ts(s0, a0) = 0.5 * (Q_phi(s0, a0) + r_psi(s0, a0) + lambda * V(1 | z1(a0)))

        If return_details=True, returns (scores, plan_details) including full multi-step
        action trajectory sequences (length D) for Pygame board & HUD overlay rendering.
        """
        self.policy_net.eval()
        self.world_model.eval()

        s = torch.as_tensor(spatial, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if s.dim() == 3: s = s.unsqueeze(0)
        if f.dim() == 1: f = f.unsqueeze(0)

        # 1. Encode root state into 256-dim fused latent representation
        z0 = self.policy_net.encode_fused(s, f)  # (1, 256)
        q0 = self.policy_net.forward_from_fused(z0).squeeze(0)  # (6,)

        depth = max(1, min(int(depth), 16))
        beam_size = max(6, int(beam_size))

        # 2. Depth 1 expansion: evaluate all 6 root actions
        a0 = torch.arange(N_ACTIONS, device=self.device, dtype=torch.int64)
        z0_rep = z0.expand(N_ACTIONS, -1)
        z1, r0, v0, legal0 = self.world_model(z0_rep, a0)  # (6, 256), (6, 1), (6, 1), (6, 6)
        r0_val = torch.nan_to_num(r0.squeeze(-1).clamp(-10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0)
        v0_val = torch.nan_to_num(v0.squeeze(-1).clamp(-50.0, 50.0), nan=0.0, posinf=50.0, neginf=-50.0)

        q1 = self.policy_net.forward_from_fused(z1)  # (6, 6)
        legal0_mask = (legal0 >= 0.0)
        legal0_mask[:, ACTION_TO_IDX['WAIT']] = True
        q1_masked = torch.where(legal0_mask, q1, torch.tensor(-1e9, device=self.device))
        v_q1 = q1_masked.max(dim=-1)[0]  # (6,)
        v1_comb = alpha_vq * v_q1 + (1.0 - alpha_vq) * v0_val

        if depth == 1:
            v1 = v1_comb
            q_ts = 0.5 * (q0 + (r0_val + discount * v1))
            if valid_mask is not None:
                vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
                q_ts = torch.where(vm, q_ts, torch.tensor(-1e9, device=self.device))
            scores = torch.nan_to_num(q_ts, nan=-1e9).cpu().numpy()
            if not return_details:
                return scores
            details = [{
                'a0': i,
                'a0_name': ACTIONS[i],
                'a1': None,
                'a1_name': None,
                'actions': [ACTIONS[i]],
                'q_ts': float(scores[i]),
                'q_root': float(q0[i].item()),
                'r0': float(r0_val[i].item()),
                'v1': float(v1[i].item()),
                'v2': float(v_q1[i].item()),
                'is_valid': bool(valid_mask[i]) if valid_mask is not None else True,
            } for i in range(N_ACTIONS)]
            return scores, details

        # For depth >= 2: beam search tracking top-K branches per root action
        K = max(1, beam_size // N_ACTIONS)

        curr_z = z1  # (P, 256), P=6 at depth 1
        curr_legal = legal0  # (P, 6): predicted action legalities from curr_z
        curr_root_idx = torch.arange(N_ACTIONS, device=self.device)  # (P,)
        curr_action_seqs = [[i] for i in range(N_ACTIONS)]
        curr_cum_r = r0_val.clone()
        curr_r_history = [r0_val]
        curr_vq_history = [v1_comb]

        best_action_seqs = None
        for d in range(1, depth):
            P = curr_z.size(0)
            z_exp = curr_z.repeat_interleave(N_ACTIONS, dim=0)  # (P * 6, 256)
            a_exp = torch.arange(N_ACTIONS, device=self.device).repeat(P)  # (P * 6,)

            # Check legality of candidate action a_exp from curr_z
            cand_legality = curr_legal.reshape(-1)  # (P * 6,)
            cand_legal_mask = (cand_legality >= 0.0) | (a_exp == ACTION_TO_IDX['WAIT'])

            next_z, next_r, next_v, next_legal = self.world_model(z_exp, a_exp)
            next_r_val = torch.nan_to_num(next_r.squeeze(-1).clamp(-10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0)
            next_v_val = torch.nan_to_num(next_v.squeeze(-1).clamp(-50.0, 50.0), nan=0.0, posinf=50.0, neginf=-50.0)

            next_q = self.policy_net.forward_from_fused(next_z)  # (P * 6, 6)
            next_legal_mask = (next_legal >= 0.0)
            next_legal_mask[:, ACTION_TO_IDX['WAIT']] = True
            next_q_masked = torch.where(next_legal_mask, next_q, torch.tensor(-1e9, device=self.device))
            next_vq = next_q_masked.max(dim=-1)[0]  # (P * 6,)
            next_comb_val = alpha_vq * next_vq + (1.0 - alpha_vq) * next_v_val

            cand_root_idx = curr_root_idx.repeat_interleave(N_ACTIONS)
            cand_action_seqs = [curr_action_seqs[p] + [a_idx] for p in range(P) for a_idx in range(N_ACTIONS)]
            cand_cum_r = curr_cum_r.repeat_interleave(N_ACTIONS) + (discount ** d) * next_r_val

            # Prospective cumulative branch score Sigma (Dong et al. 2026, Eq. 10-12):
            branch_score = cand_cum_r + (discount ** (d + 1)) * next_comb_val
            # Strictly penalize illegal candidate moves so beam search NEVER expands or selects them!
            branch_score = torch.where(cand_legal_mask, branch_score, torch.tensor(-1e9, device=self.device))

            if d < depth - 1:
                survivor_indices = []
                for root_i in range(N_ACTIONS):
                    mask_i = (cand_root_idx == root_i).nonzero(as_tuple=True)[0]
                    if len(mask_i) > 0:
                        scores_i = branch_score[mask_i]
                        legal_locs = (scores_i > -1e8).nonzero(as_tuple=True)[0]
                        if len(legal_locs) > 0:
                            topk_local = legal_locs[scores_i[legal_locs].topk(min(K, len(legal_locs)))[1]]
                        else:
                            topk_local = scores_i.topk(min(K, len(scores_i)))[1]
                        survivor_indices.extend(mask_i[topk_local].tolist())

                surv_idx = torch.tensor(survivor_indices, device=self.device, dtype=torch.int64)
                curr_z = next_z[surv_idx]
                curr_legal = next_legal[surv_idx]
                curr_root_idx = cand_root_idx[surv_idx]
                curr_action_seqs = [cand_action_seqs[idx] for idx in survivor_indices]
                curr_cum_r = cand_cum_r[surv_idx]

                parent_indices = (surv_idx // N_ACTIONS).cpu()
                new_r_hist = [hist[parent_indices] for hist in curr_r_history]
                new_r_hist.append(next_r_val[surv_idx])
                curr_r_history = new_r_hist

                new_vq_hist = [hist[parent_indices] for hist in curr_vq_history]
                new_vq_hist.append(next_comb_val[surv_idx])
                curr_vq_history = new_vq_hist
            else:
                # Leaf depth reached: pick best leaf branch per root action
                best_leaf_idx_per_root = []
                for root_i in range(N_ACTIONS):
                    mask_i = (cand_root_idx == root_i).nonzero(as_tuple=True)[0]
                    if len(mask_i) > 0:
                        scores_i = branch_score[mask_i]
                        legal_locs = (scores_i > -1e8).nonzero(as_tuple=True)[0]
                        if len(legal_locs) > 0:
                            best_idx = mask_i[legal_locs[scores_i[legal_locs].argmax()].item()]
                        else:
                            best_idx = mask_i[scores_i.argmax()].item()
                        best_leaf_idx_per_root.append(best_idx)
                    else:
                        best_leaf_idx_per_root.append(0)

                best_leaf_idx = torch.tensor(best_leaf_idx_per_root, device=self.device, dtype=torch.int64)
                best_action_seqs = [cand_action_seqs[idx] for idx in best_leaf_idx_per_root]

                parent_indices = (best_leaf_idx // N_ACTIONS).cpu()
                final_r_hist = [hist[parent_indices] for hist in curr_r_history] + [next_r_val[best_leaf_idx]]
                final_vq_hist = [hist[parent_indices] for hist in curr_vq_history] + [next_comb_val[best_leaf_idx]]

                # Backward value aggregation from depth D down to depth 1 (Eq. 8)
                V_curr = final_vq_hist[-1]  # Leaf values at depth D
                for step in range(depth - 1, 0, -1):
                    r_step = final_r_hist[step]
                    vq_step = final_vq_hist[step - 1]
                    v_r = r_step + discount * V_curr
                    V_curr = alpha_vq * vq_step + (1.0 - alpha_vq) * v_r

                v1 = V_curr

        # Root action score: Eq. (9)
        q_ts = 0.5 * (q0 + (r0_val + discount * v1))

        if valid_mask is not None:
            vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
            q_ts = torch.where(vm, q_ts, torch.tensor(-1e9, device=self.device))

        scores = torch.nan_to_num(q_ts, nan=-1e9).cpu().numpy()

        if not return_details:
            return scores

        plan_details = []
        for i in range(N_ACTIONS):
            p_seq = best_action_seqs[i] if best_action_seqs is not None else [i]
            acts = [ACTIONS[a] for a in p_seq]
            a1_idx = p_seq[1] if len(p_seq) > 1 else None
            plan_details.append({
                'a0': i,
                'a0_name': ACTIONS[i],
                'a1': a1_idx,
                'a1_name': ACTIONS[a1_idx] if a1_idx is not None else None,
                'actions': acts,
                'q_ts': float(scores[i]),
                'q_root': float(q0[i].item()),
                'r0': float(r0_val[i].item()),
                'v1': float(v1[i].item()),
                'is_valid': bool(valid_mask[i]) if valid_mask is not None else True,
            })
        return scores, plan_details

    # -------------------------------------------------------------------------
    # Training Updates
    # -------------------------------------------------------------------------
    def update_world_model_batch(
        self,
        spatial: np.ndarray,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_spatial: np.ndarray,
        next_features: np.ndarray,
        val_loss_weight: float = 0.5,
        legal_loss_weight: float = 0.5,
    ) -> Tuple[float, float, float, float, float]:
        """
        Trains the Latent World Model with transition, turn reward, long-term value, and legal action heads:
          - next-fused state MSE loss: ||hat{z}_{t+1} - target(z_{t+1})||^2
          - specific turn reward SmoothL1 loss: SmoothL1(hat{r}_t, r_t)
          - long-term sequence value SmoothL1 loss: SmoothL1(hat{V}_t, V^*(s_{t+1}))
          - future action legality BCE loss: BCEWithLogits(hat{ell}_{t+1}, legal_mask(s_{t+1}))
        Returns (total_wm_loss, trans_loss, rew_loss, val_loss, legal_loss).
        """
        self.world_model.train()
        self.policy_net.eval()

        with torch.no_grad():
            s = torch.as_tensor(spatial, dtype=torch.float32, device=self.device)
            f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            ns = torch.as_tensor(next_spatial, dtype=torch.float32, device=self.device)
            nf = torch.as_tensor(next_features, dtype=torch.float32, device=self.device)
            # Current and target next fused states
            z = self.policy_net.encode_fused(s, f).detach()
            target_next_z = self.target_net.encode_fused(ns, nf).detach()
            target_val = self.target_net.forward_from_fused(target_next_z).max(dim=-1, keepdim=True)[0]
            target_legal = get_action_legality_targets(nf)
            target_legal_curr = get_action_legality_targets(f)

        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)

        pred_next_z, pred_r, pred_v, pred_legal = self.world_model(z, a)

        loss_trans = F.mse_loss(pred_next_z, target_next_z)
        loss_rew = F.smooth_l1_loss(pred_r, r)
        loss_val = F.smooth_l1_loss(pred_v, target_val)
        loss_legal = F.binary_cross_entropy_with_logits(pred_legal, target_legal)
        loss_legal_root = F.binary_cross_entropy_with_logits(self.world_model.legal_head(z), target_legal_curr)

        total_loss = loss_trans + loss_rew + val_loss_weight * loss_val + legal_loss_weight * (loss_legal + 0.5 * loss_legal_root)

        self.wm_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
        self.wm_optimizer.step()

        return (
            float(total_loss.item()),
            float(loss_trans.item()),
            float(loss_rew.item()),
            float(loss_val.item()),
            float(loss_legal.item()),
        )

    def update_dqn_batch(
        self,
        spatial: np.ndarray,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_spatial: np.ndarray,
        next_features: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Standard Double Dueling DQN update."""
        if self._dqn_frozen:
            return 0.0

        self.policy_net.train()
        s = torch.as_tensor(spatial, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(next_spatial, dtype=torch.float32, device=self.device)
        nf = torch.as_tensor(next_features, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.policy_net(s, f).gather(1, a)

        with torch.no_grad():
            next_actions = self.policy_net(ns, nf).argmax(dim=1, keepdim=True)
            q_target_next = self.target_net(ns, nf).gather(1, next_actions)
            q_target = r + self.gamma * (1.0 - d) * q_target_next

        loss = F.smooth_l1_loss(q_pred, q_target)
        self.dqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.dqn_optimizer.step()

        # Polyak soft update of target network
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

        return float(loss.item())

    def update_batch(
        self,
        spatial: np.ndarray,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_spatial: np.ndarray,
        next_features: np.ndarray,
        dones: np.ndarray,
    ) -> Dict[str, float]:
        """Performs joint or phased update."""
        wm_loss, trans_loss, rew_loss, val_loss, legal_loss = self.update_world_model_batch(
            spatial, features, actions, rewards, next_spatial, next_features
        )
        dqn_loss = self.update_dqn_batch(
            spatial, features, actions, rewards, next_spatial, next_features, dones
        )
        return {
            "loss": dqn_loss,
            "wm_loss": wm_loss,
            "trans_loss": trans_loss,
            "rew_loss": rew_loss,
            "val_loss": val_loss,
            "legal_loss": legal_loss,
        }

    def update_batch_cuda(
        self,
        spatial: torch.Tensor,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_spatial: torch.Tensor,
        next_features: torch.Tensor,
        dones: torch.Tensor,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """High-performance CUDA batch update for QWM using AMP FP16 and Tensor Cores."""
        use_amp = (self.device.type == "cuda")

        # 1. World Model update
        self.world_model.train()
        with torch.no_grad():
            z_curr = self.policy_net.encode_fused(spatial, features)
            z_next_target = self.policy_net.encode_fused(next_spatial, next_features)
            target_next_q = self.target_net.forward_from_fused(z_next_target)
            target_val = target_next_q.max(dim=-1, keepdim=True)[0]
            target_legal = get_action_legality_targets(next_features)
            target_legal_curr = get_action_legality_targets(features)

        with amp_autocast("cuda", enabled=use_amp):
            z_next_pred, r_pred, v_pred, legal_pred = self.world_model.forward(z_curr, actions)
            trans_loss = F.mse_loss(z_next_pred, z_next_target)
            rew_loss = F.smooth_l1_loss(r_pred, rewards)
            val_loss = F.smooth_l1_loss(v_pred, target_val)
            loss_legal = F.binary_cross_entropy_with_logits(legal_pred, target_legal)
            loss_legal_root = F.binary_cross_entropy_with_logits(self.world_model.legal_head(z_curr), target_legal_curr)
            wm_loss = trans_loss + rew_loss + 0.5 * val_loss + 0.5 * (loss_legal + 0.5 * loss_legal_root)

        self.wm_optimizer.zero_grad()
        if use_amp and getattr(self, "scaler_wm", None) is None:
            self.scaler_wm = make_grad_scaler("cuda")
        wm_scaler = getattr(self, "scaler_wm", None) or scaler
        if wm_scaler is not None and use_amp:
            wm_scaler.scale(wm_loss).backward()
            wm_scaler.unscale_(self.wm_optimizer)
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            wm_scaler.step(self.wm_optimizer)
            wm_scaler.update()
        else:
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.wm_optimizer.step()

        # 2. DQN update (if not frozen)
        dqn_loss_val = 0.0
        if not self._dqn_frozen:
            self.policy_net.train()
            with amp_autocast("cuda", enabled=use_amp):
                q_pred = self.policy_net(spatial, features).gather(1, actions)
                with torch.no_grad():
                    next_actions = self.policy_net(next_spatial, next_features).argmax(dim=1, keepdim=True)
                    q_target_next = self.target_net(next_spatial, next_features).gather(1, next_actions)
                    q_target = rewards + self.gamma * (1.0 - dones) * q_target_next
                dqn_loss = F.smooth_l1_loss(q_pred, q_target)

            self.dqn_optimizer.zero_grad()
            if use_amp and getattr(self, "scaler_dqn", None) is None:
                self.scaler_dqn = make_grad_scaler("cuda")
            dqn_scaler = getattr(self, "scaler_dqn", None) or scaler
            if dqn_scaler is not None and use_amp:
                dqn_scaler.scale(dqn_loss).backward()
                dqn_scaler.unscale_(self.dqn_optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                dqn_scaler.step(self.dqn_optimizer)
                dqn_scaler.update()
            else:
                dqn_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                self.dqn_optimizer.step()

            with torch.no_grad():
                for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

            dqn_loss_val = float(dqn_loss.item())

        return {
            "loss": dqn_loss_val,
            "wm_loss": float(wm_loss.item()),
            "trans_loss": float(trans_loss.item()),
            "rew_loss": float(rew_loss.item()),
            "val_loss": float(val_loss.item()),
            "legal_loss": float(loss_legal.item()),
        }

    def update_multistep_paths(
        self,
        spatials: np.ndarray,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        val_loss_weight: float = 0.5,
        random_horizon: bool = False,
        horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Trains the Latent World Model on multi-step candidate trajectories (full predicted paths).
        Supports future imagination learning across random horizons (1 through 8) or fixed H.
        Unrolls M_psi autoregressively across horizon H:
          z_0 -> (hat{z}_1, hat{r}_0) -> (hat{z}_2, hat{r}_1) -> ... -> (hat{z}_H, hat{r}_{H-1})
        and penalizes:
          1. Multi-step latent transition error: ||hat{z}_{h+1} - z^*_{h+1}||^2
          2. Multi-step reward prediction error: |hat{r}_h - r_h|
          3. Future value prediction error: |hat{V}_{h+1} - V^*_{h+1}| where hat{V} = max_a Q(hat{z})
        Returns overall losses and per-horizon error metrics.
        """
        self.world_model.train()
        self.policy_net.eval()

        s_t = torch.as_tensor(spatials, dtype=torch.float32, device=self.device)   # (B, H+1, C, W, H_b)
        f_t = torch.as_tensor(features, dtype=torch.float32, device=self.device)   # (B, H+1, F)
        a_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)      # (B, H)
        r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)    # (B, H)
        d_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)      # (B, H)

        B, full_H = a_t.size(0), a_t.size(1)
        if horizon is not None:
            H = min(full_H, max(1, horizon))
        elif random_horizon and full_H > 1:
            H = random.randint(1, min(8, full_H))
        else:
            H = full_H

        if H < full_H:
            s_t = s_t[:, :H + 1]
            f_t = f_t[:, :H + 1]
            a_t = a_t[:, :H]
            r_t = r_t[:, :H]
            d_t = d_t[:, :H]

        with torch.no_grad():
            z_curr = self.policy_net.encode_fused(s_t[:, 0], f_t[:, 0]).detach()

        total_trans_loss = torch.tensor(0.0, device=self.device)
        total_rew_loss = torch.tensor(0.0, device=self.device)
        total_val_loss = torch.tensor(0.0, device=self.device)
        total_legal_loss = torch.tensor(0.0, device=self.device)

        horizon_mae = []
        horizon_rmse = []
        pred_vals_mean = []
        actual_vals_mean = []

        for h in range(H):
            act_h = a_t[:, h]
            rew_h = r_t[:, h].unsqueeze(1)
            discount = self.gamma ** h

            z_next_pred, r_pred, v_pred, legal_pred = self.world_model(z_curr, act_h)

            with torch.no_grad():
                z_next_target = self.target_net.encode_fused(s_t[:, h + 1], f_t[:, h + 1]).detach()
                target_q = self.target_net.forward_from_fused(z_next_target)
                target_val = target_q.max(dim=-1, keepdim=True)[0]
                target_legal = get_action_legality_targets(f_t[:, h + 1])

            step_trans = F.mse_loss(z_next_pred, z_next_target)
            step_rew = F.smooth_l1_loss(r_pred, rew_h)
            step_val = F.smooth_l1_loss(v_pred, target_val)
            step_legal = F.binary_cross_entropy_with_logits(legal_pred, target_legal)

            total_trans_loss = total_trans_loss + discount * step_trans
            total_rew_loss = total_rew_loss + discount * step_rew
            total_val_loss = total_val_loss + discount * step_val
            total_legal_loss = total_legal_loss + discount * step_legal

            with torch.no_grad():
                diff = (v_pred - target_val).squeeze(-1)
                horizon_mae.append(float(diff.abs().mean().item()))
                horizon_rmse.append(float(torch.sqrt((diff ** 2).mean()).item()))
                pred_vals_mean.append(float(v_pred.mean().item()))
                actual_vals_mean.append(float(target_val.mean().item()))

            # Autoregressive forward connection for multi-step path rollouts
            z_curr = z_next_pred

        total_loss = total_trans_loss + total_rew_loss + val_loss_weight * total_val_loss + 0.5 * total_legal_loss

        self.wm_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
        self.wm_optimizer.step()

        return {
            "wm_path_loss": float(total_loss.item()),
            "trans_loss": float(total_trans_loss.item()),
            "rew_loss": float(total_rew_loss.item()),
            "future_val_loss": float(total_val_loss.item()),
            "legal_loss": float(total_legal_loss.item()),
            "horizon": H,
            "horizon_mae": horizon_mae,
            "horizon_rmse": horizon_rmse,
            "pred_vals_mean": pred_vals_mean,
            "actual_vals_mean": actual_vals_mean,
        }

    def update_multistep_paths_cuda(
        self,
        spatials: torch.Tensor,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        val_loss_weight: float = 0.5,
        random_horizon: bool = False,
        horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        """CUDA-accelerated AMP multi-step path training on full predicted trajectories.
        Supports future imagination learning across random horizons (1 through 8) or fixed H.
        """
        use_amp = (self.device.type == "cuda")
        self.world_model.train()
        self.policy_net.eval()

        B, full_H = actions.size(0), actions.size(1)
        if horizon is not None:
            H = min(full_H, max(1, horizon))
        elif random_horizon and full_H > 1:
            H = random.randint(1, min(8, full_H))
        else:
            H = full_H

        if H < full_H:
            spatials = spatials[:, :H + 1]
            features = features[:, :H + 1]
            actions = actions[:, :H]
            rewards = rewards[:, :H]
            dones = dones[:, :H]

        with torch.no_grad():
            z_curr = self.policy_net.encode_fused(spatials[:, 0], features[:, 0]).detach()

        total_trans_loss = torch.tensor(0.0, device=self.device)
        total_rew_loss = torch.tensor(0.0, device=self.device)
        total_val_loss = torch.tensor(0.0, device=self.device)
        total_legal_loss = torch.tensor(0.0, device=self.device)

        horizon_mae = []
        horizon_rmse = []
        pred_vals_mean = []
        actual_vals_mean = []

        with amp_autocast("cuda", enabled=use_amp):
            for h in range(H):
                act_h = actions[:, h]
                rew_h = rewards[:, h].unsqueeze(1)
                discount = self.gamma ** h

                z_next_pred, r_pred, v_pred, legal_pred = self.world_model(z_curr, act_h)

                with torch.no_grad():
                    z_next_target = self.target_net.encode_fused(spatials[:, h + 1], features[:, h + 1]).detach()
                    target_q = self.target_net.forward_from_fused(z_next_target)
                    target_val = target_q.max(dim=-1, keepdim=True)[0]
                    target_legal = get_action_legality_targets(features[:, h + 1])

                step_trans = F.mse_loss(z_next_pred, z_next_target)
                step_rew = F.smooth_l1_loss(r_pred, rew_h)
                step_val = F.smooth_l1_loss(v_pred, target_val)
                step_legal = F.binary_cross_entropy_with_logits(legal_pred, target_legal)

                total_trans_loss = total_trans_loss + discount * step_trans
                total_rew_loss = total_rew_loss + discount * step_rew
                total_val_loss = total_val_loss + discount * step_val
                total_legal_loss = total_legal_loss + discount * step_legal

                with torch.no_grad():
                    diff = (v_pred - target_val).squeeze(-1)
                    horizon_mae.append(float(diff.abs().mean().item()))
                    horizon_rmse.append(float(torch.sqrt((diff ** 2).mean()).item()))
                    pred_vals_mean.append(float(v_pred.mean().item()))
                    actual_vals_mean.append(float(target_val.mean().item()))

                z_curr = z_next_pred

            total_loss = total_trans_loss + total_rew_loss + val_loss_weight * total_val_loss + 0.5 * total_legal_loss

        self.wm_optimizer.zero_grad()
        if use_amp and getattr(self, "scaler_wm_path", None) is None:
            self.scaler_wm_path = make_grad_scaler("cuda")
        wm_scaler = getattr(self, "scaler_wm_path", None) or getattr(self, "scaler_wm", None) or scaler
        if wm_scaler is not None and use_amp:
            wm_scaler.scale(total_loss).backward()
            wm_scaler.unscale_(self.wm_optimizer)
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            wm_scaler.step(self.wm_optimizer)
            wm_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.wm_optimizer.step()

        return {
            "wm_path_loss": float(total_loss.item()),
            "trans_loss": float(total_trans_loss.item()),
            "rew_loss": float(total_rew_loss.item()),
            "future_val_loss": float(total_val_loss.item()),
            "legal_loss": float(total_legal_loss.item()),
            "horizon": H,
            "horizon_mae": horizon_mae,
            "horizon_rmse": horizon_rmse,
            "pred_vals_mean": pred_vals_mean,
            "actual_vals_mean": actual_vals_mean,
        }

    @torch.no_grad()
    def evaluate_future_values(
        self,
        spatials: np.ndarray,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluates multi-step predicted values against actual target values across horizons."""
        self.world_model.eval()
        self.policy_net.eval()

        s_t = torch.as_tensor(spatials, dtype=torch.float32, device=self.device)
        f_t = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

        B, H = a_t.size(0), a_t.size(1)
        z_curr = self.policy_net.encode_fused(s_t[:, 0], f_t[:, 0])

        horizon_data = []
        for h in range(H):
            act_h = a_t[:, h]
            z_next_pred, r_pred, v_pred, _ = self.world_model(z_curr, act_h)

            z_next_target = self.target_net.encode_fused(s_t[:, h + 1], f_t[:, h + 1])
            target_val = self.target_net.forward_from_fused(z_next_target).max(dim=-1)[0]
            pred_val = v_pred.squeeze(-1)

            diff = (pred_val - target_val).cpu().numpy()
            horizon_data.append({
                "horizon": h + 1,
                "pred_mean": float(pred_val.mean().item()),
                "pred_std": float(pred_val.std().item()),
                "actual_mean": float(target_val.mean().item()),
                "actual_std": float(target_val.std().item()),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff ** 2))),
                "sample_preds": pred_val[:12].cpu().numpy().tolist(),
                "sample_actuals": target_val[:12].cpu().numpy().tolist(),
            })
            z_curr = z_next_pred

        return {"horizons": horizon_data}

    def sync_to_cpu_shared(self, cpu_shared_model):
        """In-place weight transfer from CUDA policy, target, and world model nets into CPU shared memory."""
        with torch.no_grad():
            for p_gpu, p_cpu in zip(self.policy_net.parameters(), cpu_shared_model.policy_net.parameters()):
                p_cpu.data.copy_(p_gpu.data.cpu())
            for p_gpu, p_cpu in zip(self.target_net.parameters(), cpu_shared_model.target_net.parameters()):
                p_cpu.data.copy_(p_gpu.data.cpu())
            for p_gpu, p_cpu in zip(self.world_model.parameters(), cpu_shared_model.world_model.parameters()):
                p_cpu.data.copy_(p_gpu.data.cpu())

    # -------------------------------------------------------------------------
    # Saving & Loading
    # -------------------------------------------------------------------------
    def save(self, filepath: str):
        out_dir = os.path.dirname(os.path.abspath(filepath))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            'kind': self.kind,
            'lr': self.lr,
            'wm_lr': self.wm_lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'world_model_state_dict': self.world_model.state_dict(),
            'dqn_optimizer_state_dict': self.dqn_optimizer.state_dict(),
            'wm_optimizer_state_dict': self.wm_optimizer.state_dict(),
            'dqn_frozen': self._dqn_frozen,
        }
        tmp_file = f"{filepath}.tmp.{os.getpid()}"
        try:
            torch.save(payload, tmp_file)
            os.replace(tmp_file, filepath)
        except Exception:
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)
            torch.save(payload, filepath)

    def load(self, filepath: str, need_training: bool = False, partial: bool = True) -> Tuple[List[str], List[str]]:
        checkpoint = None
        for attempt in range(5):
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                break
            except Exception as ex:
                if attempt == 4:
                    raise ex
                import time
                time.sleep(0.05)

        loaded_keys = []
        uninit_keys = []

        # 1. Load policy and target nets
        pol_dict = checkpoint.get('policy_state_dict', checkpoint)
        tgt_dict = checkpoint.get('target_state_dict', pol_dict)
        l_pol, u_pol = load_matching_weights(self.policy_net, pol_dict, partial=partial)
        load_matching_weights(self.target_net, tgt_dict, partial=partial)
        loaded_keys.extend([f"policy.{k}" for k in l_pol])
        uninit_keys.extend([f"policy.{k}" for k in u_pol])

        # 2. Load world model if present in checkpoint
        if 'world_model_state_dict' in checkpoint:
            l_wm, u_wm = load_matching_weights(self.world_model, checkpoint['world_model_state_dict'], partial=partial)
            loaded_keys.extend([f"wm.{k}" for k in l_wm])
            uninit_keys.extend([f"wm.{k}" for k in u_wm])
        else:
            uninit_keys.extend([f"wm.{k}" for k in self.world_model.state_dict().keys()])

        # 3. Load optimizers if not partial
        if need_training and not partial:
            if 'dqn_optimizer_state_dict' in checkpoint:
                try: self.dqn_optimizer.load_state_dict(checkpoint['dqn_optimizer_state_dict'])
                except Exception: pass
            if 'wm_optimizer_state_dict' in checkpoint:
                try: self.wm_optimizer.load_state_dict(checkpoint['wm_optimizer_state_dict'])
                except Exception: pass

        if 'dqn_frozen' in checkpoint:
            self._dqn_frozen = bool(checkpoint['dqn_frozen'])
            if self._dqn_frozen:
                self.freeze_dqn()
            else:
                self.unfreeze_dqn()

        return loaded_keys, uninit_keys


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------
def new_model(
    lr: float = 1e-4,
    wm_lr: float = 3e-4,
    gamma: float = 0.95,
    tau: float = 0.005,
    device: Optional[torch.device] = None
) -> SpatialQWMAgent:
    return SpatialQWMAgent(lr=lr, wm_lr=wm_lr, gamma=gamma, tau=tau, device=device)


def load_model(
    filepath: str,
    need_training: bool = False,
    partial: bool = True,
    device: Optional[torch.device] = None
) -> SpatialQWMAgent:
    checkpoint = torch.load(filepath, map_location=device or get_device())
    lr = float(checkpoint.get('lr', 1e-4))
    wm_lr = float(checkpoint.get('wm_lr', 3e-4))
    gamma = float(checkpoint.get('gamma', 0.95))
    tau = float(checkpoint.get('tau', 0.005))
    agent = SpatialQWMAgent(lr=lr, wm_lr=wm_lr, gamma=gamma, tau=tau, device=device)
    agent.load(filepath, need_training=need_training, partial=partial)
    return agent
