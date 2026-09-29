"""
model.py -- Full-Board Spatial Dueling DQN Agent.

Architecture:
  - Input A (Spatial): 10-channel 17x17 grid representing the entire board:
      0: Stone walls (-1)
      1: Destructible crates (1)
      2: Free tiles (0)
      3: Active bombs & projected blast hazard corridors
      4: Active explosions (explosion_map)
      5: Coins
      6: Self position
      7: Opponent positions
      8: Recent tile trail / heatmap from coordinate_history (anti-looping)
      9: Full-board BFS distance gradient to nearest coin / crate target
  - Input B (Deterministic): 33-dim physics/safety features from my_wm_agent.
  - Torso: 4-layer Conv2D spatial backbone + Linear projection (256 dims).
  - Fusion: Concat(Spatial embed [256], Features [33]) -> Linear(289 -> 256).
  - Head: Dueling DQN (Value stream V(s) in R^1, Advantage stream A(s, a) in R^6).
  - Training: Double DQN with soft target network updates (tau=0.005) and
    dihedral D4 symmetry augmentation.
"""

import copy
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_code.my_wm_agent.model import (
    ACTIONS,
    ACTION_TO_IDX,
    DIRS,
    DIR_VECS,
    MOVE_ACTIONS,
    N_ACTIONS,
    N_FEATURES,
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
    F_VISITED,
    F_NBR_VISITED,
    INF,
    rotate_features,
    mirror_features,
    state_to_features as wm_state_to_features,
    valid_action_mask as wm_valid_action_mask,
    crate_adjacent_tiles,
    get_blast_zones,
    bfs_direction,
)

AGENT_DIR = Path(__file__).resolve().parent
N_SPATIAL_CHANNELS = 10
BOARD_W = 17
BOARD_H = 17
DIR_VECS_LIST = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT


def get_device(explicit_device: Optional[str] = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    env_dev = "cuda"#os.environ.get("MY_WM_DEVICE")
    if env_dev:
        return torch.device(env_dev)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def agent_path(*parts) -> str:
    return str(AGENT_DIR.joinpath(*parts))


# Custom activation function: dynActivation

class dynActivation(nn.Module):
    """
    Custom activation function that dynamically adjusts its behavior based on the input.
    This is a placeholder for a more complex activation function that could be used in the model.
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Learnable parameter to scale the input
        self.beta = nn.Parameter(torch.tensor(0.0))  # Learnable parameter to scale the input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return F.mish(x) * (self.alpha - self.beta) + self.beta * x


# ============================================================================
# 1. Full-Board Spatial Tensor Extractor
# ============================================================================
def state_to_spatial_tensor(
    game_state: Optional[dict],
    coordinate_history: Optional[deque] = None
) -> np.ndarray:
    """
    Extracts a (10, 17, 17) float32 tensor representing the whole board state.
    """
    spatial = np.zeros((N_SPATIAL_CHANNELS, BOARD_W, BOARD_H), dtype=np.float32)
    if game_state is None or 'field' not in game_state:
        return spatial

    field = game_state['field']
    W, H = field.shape
    bombs = game_state.get('bombs', [])
    explosion_map = game_state.get('explosion_map')
    coins = game_state.get('coins', [])
    others = game_state.get('others', [])
    self_info = game_state.get('self')

    # Channel 0: Stone walls (-1)
    spatial[0] = (field == -1).astype(np.float32)

    # Channel 1: Crates (1)
    spatial[1] = (field == 1).astype(np.float32)

    # Channel 2: Free tiles (0)
    spatial[2] = (field == 0).astype(np.float32)

    # Channel 3: Active bombs & projected blast corridors
    # Timer is 4..0. Normalized intensity (5 - t) / 5.0
    for (bx, by), timer in bombs:
        if 0 <= bx < W and 0 <= by < H:
            danger_val = (5.0 - float(timer)) / 5.0
            spatial[3, bx, by] = max(spatial[3, bx, by], danger_val)
            spatial[2, bx, by] = 0.0  # blocked
            for dx, dy in DIR_VECS_LIST:
                for dist in range(1, 4):
                    nx, ny = bx + dx * dist, by + dy * dist
                    if not (0 <= nx < W and 0 <= ny < H):
                        break
                    if field[nx, ny] == -1:
                        break
                    spatial[3, nx, ny] = max(spatial[3, nx, ny], danger_val)
                    if field[nx, ny] == 1:
                        break

    # Channel 4: Active explosions
    if explosion_map is not None:
        spatial[4] = np.clip(explosion_map.astype(np.float32) / 2.0, 0.0, 1.0)

    # Channel 5: Coins
    for cx, cy in coins:
        if 0 <= cx < W and 0 <= cy < H:
            spatial[5, cx, cy] = 1.0

    # Channel 6: Self position
    self_x, self_y = (1, 1)
    if self_info is not None and len(self_info) >= 4:
        _, _, _, (self_x, self_y) = self_info
        if 0 <= self_x < W and 0 <= self_y < H:
            spatial[6, self_x, self_y] = 1.0

    # Channel 7: Opponents
    for other in others:
        if other is not None and len(other) >= 4:
            ox, oy = other[3]
            if 0 <= ox < W and 0 <= oy < H:
                spatial[7, ox, oy] = 1.0
                spatial[2, ox, oy] = 0.0  # blocked

    # Channel 8: History Heatmap (recent visited tiles)
    if coordinate_history:
        for p in coordinate_history:
            if isinstance(p, (tuple, list)) and len(p) >= 2:
                px, py = p[0], p[1]
                if 0 <= px < W and 0 <= py < H:
                    spatial[8, px, py] = min(1.0, spatial[8, px, py] + 0.25)

    # Channel 9: Full-board BFS distance gradient to nearest goal
    # Provides explicit reachability slope across the whole maze
    passable = (field == 0)
    for (bx, by), _ in bombs:
        if 0 <= bx < W and 0 <= by < H:
            passable[bx, by] = False
    if explosion_map is not None:
        passable &= (explosion_map == 0)
    for other in others:
        if other is not None and len(other) >= 4:
            ox, oy = other[3]
            if 0 <= ox < W and 0 <= oy < H:
                passable[ox, oy] = False

    # Target set: coins if present, else crate-adjacent tiles
    targets = set(coins) if len(coins) > 0 else set(crate_adjacent_tiles(field))
    if targets and (0 <= self_x < W and 0 <= self_y < H):
        dist_map = np.full((W, H), -1, dtype=np.int32)
        q = deque([(self_x, self_y)])
        dist_map[self_x, self_y] = 0
        while q:
            cx, cy = q.popleft()
            d = dist_map[cx, cy]
            for dx, dy in DIR_VECS_LIST:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H and dist_map[nx, ny] < 0 and passable[nx, ny]:
                    dist_map[nx, ny] = d + 1
                    q.append((nx, ny))
        # Where targets are reached, write normalized gradient 1 / (1 + d)
        for tx, ty in targets:
            if 0 <= tx < W and 0 <= ty < H and dist_map[tx, ty] >= 0:
                spatial[9, tx, ty] = 1.0 / (1.0 + float(dist_map[tx, ty]))

    return spatial


def state_to_features(game_state, coordinate_history=None):
    """33-dim deterministic physics/world-model feature vector."""
    return wm_state_to_features(game_state, coordinate_history)


def valid_action_mask(features):
    """Boolean mask of length 6 derived from features (blocked moves, bombs_left)."""
    return wm_valid_action_mask(features)


# ============================================================================
# 2. Dueling Spatial DQN Architecture
# ============================================================================
class SpatialDuelingDQN(nn.Module):
    """
    Dueling Q-Network combining a 10-channel 17x17 spatial CNN with a 33-dim
    physics feature vector.
    """
    def __init__(self, in_channels: int = N_SPATIAL_CHANNELS, n_features: int = N_FEATURES, n_actions: int = N_ACTIONS):
        super().__init__()
        # Conv backbone preserving spatial resolution with padding
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
        # 32 * 17 * 17 = 9248
        self.spatial_proj = nn.Sequential(
            nn.Linear(32 * BOARD_W * BOARD_H, 256),
            nn.ReLU(),
        )
        # Fusion of spatial embedding (256) + deterministic features (33)
        self.fusion = nn.Sequential(
            nn.Linear(256 + n_features, 256),
            nn.ReLU(),
        )
        # Dueling streams
        self.val_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, spatial: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        b = spatial.size(0)
        c = self.conv(spatial).view(b, -1)
        s_embed = self.spatial_proj(c)
        fused = self.fusion(torch.cat([s_embed, features], dim=1))
        val = self.val_stream(fused)
        adv = self.adv_stream(fused)
        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
        q = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q


def amp_autocast(device_type: str = "cuda", enabled: bool = True):
    """Clean forward-compatible AMP autocast context manager."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


# ============================================================================
# 3. SpatialDQNAgent Class
# ============================================================================
class SpatialDQNAgent:
    """
    Double Dueling DQN Agent with Polyak soft target updates and Huber loss.
    """
    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        device: Optional[torch.device] = None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.device = device or get_device()
        self.kind = "spatial_dqn"

        self.policy_net = SpatialDuelingDQN().to(self.device)
        self.target_net = SpatialDuelingDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.lr, weight_decay=1e-4)

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        elif self.device.type == "cpu":
            torch.set_num_threads(1)

    def set_lr(self, lr: float):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def q_values(self, spatial: np.ndarray, features: np.ndarray) -> np.ndarray:
        self.policy_net.eval()
        with torch.no_grad():
            s = torch.as_tensor(spatial, dtype=torch.float32, device=self.device)
            f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            if s.dim() == 3:
                s = s.unsqueeze(0)
            if f.dim() == 1:
                f = f.unsqueeze(0)
            q = self.policy_net(s, f)
            if q.size(0) == 1:
                return q.squeeze(0).cpu().numpy()
            return q.cpu().numpy()

    def update_batch(
        self,
        spatial: np.ndarray,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_spatial: np.ndarray,
        next_features: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        self.policy_net.train()
        s = torch.as_tensor(spatial, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(next_spatial, dtype=torch.float32, device=self.device)
        nf = torch.as_tensor(next_features, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q estimates: Q_policy(s, a)
        q_pred = self.policy_net(s, f).gather(1, a)

        # Double DQN target computation:
        # a* = argmax_a Q_policy(s', a)
        # target = r + gamma * (1 - d) * Q_target(s', a*)
        with torch.no_grad():
            next_actions = self.policy_net(ns, nf).argmax(dim=1, keepdim=True)
            q_target_next = self.target_net(ns, nf).gather(1, next_actions)
            q_target = r + self.gamma * (1.0 - d) * q_target_next

        loss = F.smooth_l1_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Polyak soft update of target network
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

        return float(loss.item())

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
    ) -> float:
        """High-performance CUDA batch update using AMP FP16 and Tensor Cores."""
        self.policy_net.train()
        use_amp = (self.device.type == "cuda")

        with amp_autocast("cuda", enabled=use_amp):
            q_pred = self.policy_net(spatial, features).gather(1, actions)
            with torch.no_grad():
                next_actions = self.policy_net(next_spatial, next_features).argmax(dim=1, keepdim=True)
                q_target_next = self.target_net(next_spatial, next_features).gather(1, next_actions)
                q_target = rewards + self.gamma * (1.0 - dones) * q_target_next
            loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Polyak soft update of target network
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

        return float(loss.item())

    def sync_to_cpu_shared(self, cpu_shared_model):
        """In-place weight transfer from CUDA policy and target nets into CPU shared memory."""
        with torch.no_grad():
            for p_gpu, p_cpu in zip(self.policy_net.parameters(), cpu_shared_model.policy_net.parameters()):
                p_cpu.data.copy_(p_gpu.data.cpu())
            for p_gpu, p_cpu in zip(self.target_net.parameters(), cpu_shared_model.target_net.parameters()):
                p_cpu.data.copy_(p_gpu.data.cpu())

    def compute_gradients(
        self,
        spatial: np.ndarray,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_spatial: np.ndarray,
        next_features: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[float, List[torch.Tensor]]:
        """Compute loss and return detached parameter gradients for policy_net without optimizer step."""
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
        self.policy_net.zero_grad()
        loss.backward()
        grads = [p.grad.detach().clone() for p in self.policy_net.parameters() if p.grad is not None]
        self.policy_net.zero_grad()
        return float(loss.item()), grads

    def apply_accumulated_gradients(self, accumulated_grads: List[torch.Tensor], scale: float = 1.0):
        """Apply accumulated gradients to policy_net, step optimizer, and soft-update target_net."""
        self.optimizer.zero_grad()
        for p, g in zip(self.policy_net.parameters(), accumulated_grads):
            p.grad = g.to(self.device) * scale
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Polyak soft update of target network
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            'kind': self.kind,
            'lr': self.lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(payload, filepath)

    def load(self, filepath: str, need_training: bool = False):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint.get('target_state_dict', checkpoint['policy_state_dict']))
        if need_training and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass


def new_model(lr: float = 1e-4, gamma: float = 0.95, tau: float = 0.005, device: Optional[torch.device] = None) -> SpatialDQNAgent:
    return SpatialDQNAgent(lr=lr, gamma=gamma, tau=tau, device=device)


def load_model(filepath: str, need_training: bool = False, device: Optional[torch.device] = None) -> SpatialDQNAgent:
    checkpoint = torch.load(filepath, map_location=device or get_device())
    lr = float(checkpoint.get('lr', 1e-4))
    gamma = float(checkpoint.get('gamma', 0.95))
    tau = float(checkpoint.get('tau', 0.005))
    agent = SpatialDQNAgent(lr=lr, gamma=gamma, tau=tau, device=device)
    agent.load(filepath, need_training=need_training)
    return agent


def save_model(agent: SpatialDQNAgent, filepath: str):
    agent.save(filepath)


# ============================================================================
# 4. Dihedral D4 Symmetry Augmentation for Spatial Transitions
# ============================================================================
# Action permutation for rotation 90 deg clockwise: UP->RIGHT, RIGHT->DOWN, DOWN->LEFT, LEFT->UP
ROT_ACTION_MAP = {0: 1, 1: 2, 2: 3, 3: 0, 4: 4, 5: 5}
# Action permutation for horizontal flip (mirror): UP->UP, DOWN->DOWN, RIGHT<->LEFT
MIRROR_ACTION_MAP = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5}


def rotate_spatial(spatial: np.ndarray, k: int) -> np.ndarray:
    """Rotate (C, 17, 17) spatial tensor by k * 90 degrees clockwise."""
    # np.rot90 rotates counter-clockwise by default, so -k rotates clockwise
    return np.ascontiguousarray(np.rot90(spatial, -k, axes=(1, 2)))


def mirror_spatial(spatial: np.ndarray) -> np.ndarray:
    """Mirror (C, 17, 17) spatial tensor horizontally (swap left and right columns)."""
    return np.ascontiguousarray(np.flip(spatial, axis=2))


def augment_spatial_transition(
    spatial: np.ndarray,
    features: np.ndarray,
    action_idx: int,
    next_spatial: np.ndarray,
    next_features: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]]:
    """
    Produces all 8 dihedral transformations (4 rotations x 2 reflections).
    """
    out = []
    curr_s = spatial
    curr_f = features
    curr_a = action_idx
    curr_ns = next_spatial
    curr_nf = next_features

    for rot in range(4):
        # Add current rotated version
        out.append((curr_s, curr_f, curr_a, curr_ns, curr_nf))
        # Add horizontal reflection of current rotated version
        m_s = mirror_spatial(curr_s)
        m_f = mirror_features(curr_f)
        m_a = MIRROR_ACTION_MAP[curr_a]
        m_ns = mirror_spatial(curr_ns)
        m_nf = mirror_features(curr_nf)
        out.append((m_s, m_f, m_a, m_ns, m_nf))

        # Rotate clockwise by 90 degrees for next iteration
        curr_s = rotate_spatial(curr_s, 1)
        curr_f = rotate_features(curr_f, 1)
        curr_a = ROT_ACTION_MAP[curr_a]
        curr_ns = rotate_spatial(curr_ns, 1)
        curr_nf = rotate_features(curr_nf, 1)

    return out
