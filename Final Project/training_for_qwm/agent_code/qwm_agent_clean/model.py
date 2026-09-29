"""
Pure Neural Spatial QWM Inference Model & Vectorized Tree Search.
Streamlined exclusively for inference without any training bloat or hardcoded policy rules.
"""

from collections import deque
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AGENT_DIR = Path(__file__).resolve().parent

# ============================================================================
# Core Constants & Board Representations (Self-Contained)
# ============================================================================
COLS, ROWS = 17, 17
BOARD_W = 17
BOARD_H = 17
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2
INF = 999
HORIZON = BOMB_TIMER + EXPLOSION_TIMER + 1

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_ACTIONS = len(ACTIONS)
MOVE_ACTIONS = ACTIONS[:4]  # UP, RIGHT, DOWN, LEFT
DIR_VECS = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}
DIRS = [(a, DIR_VECS[a]) for a in MOVE_ACTIONS]
DIR_VECS_LIST = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

N_SPATIAL_CHANNELS = 10
N_FEATURES = 33

# Feature indices (matching 33-dim vector layout required by the trained neural network)
N_FEATURES = 33
F_SAFE_MOVE = 0
F_SAFE_WAIT = 4
F_IN_DANGER = 5
F_COIN_DIR = 6
F_COIN_PROX = 10
F_CRATE_DIR = 11
F_BOMB_UTIL = 15
F_TRAPS_SELF = 16
F_BOMBS_LEFT = 17
F_OPP_DIR = 18
F_OPP_PROX = 22
F_BLOCKED = 23
F_VISITED = 27
F_BIAS = 28
F_NBR_VISITED = 29
DIRECTIONAL_GROUPS = (F_SAFE_MOVE, F_COIN_DIR, F_CRATE_DIR, F_OPP_DIR, F_BLOCKED, F_NBR_VISITED)


def get_device(explicit_device: Optional[str] = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def agent_path(*parts) -> str:
    return str(AGENT_DIR.joinpath(*parts))


# ============================================================================
# Deterministic Board Physics & Feature Extraction (Required for Neural Inputs)
# ============================================================================

def blast_coords(pos: Tuple[int, int], field: np.ndarray, power: int = BOMB_POWER) -> List[Tuple[int, int]]:
    """Tiles hit by a bomb at `pos`; same corridor logic as items.Bomb.get_blast_coords."""
    x, y = pos
    coords = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def get_blast_zones(bombs, field: np.ndarray) -> np.ndarray:
    """
    blast_map[x, y] = minimum countdown of any bomb whose blast covers (x, y).
    0 means the bomb explodes right after the next move; INF (999) = not threatened.
    """
    blast_map = np.full(field.shape, INF, dtype=np.int32)
    for (bx, by), t in bombs:
        for (x, y) in blast_coords((bx, by), field):
            if t < blast_map[x, y]:
                blast_map[x, y] = t
    return blast_map


def danger_tensor(bombs, field: np.ndarray, explosion_map: Optional[np.ndarray], horizon: int = HORIZON) -> np.ndarray:
    """
    danger[k, x, y] = True if standing on (x, y) after the k-th move from now is lethal.
    Index k = 0 is unused (agent is alive now).
    """
    danger = np.zeros((horizon + 1,) + field.shape, dtype=bool)
    for (bx, by), t in bombs:
        k0, k1 = t + 1, min(t + EXPLOSION_TIMER, horizon)
        if k0 > horizon:
            continue
        for (x, y) in blast_coords((bx, by), field):
            danger[k0:k1 + 1, x, y] = True
    if explosion_map is not None:
        xs, ys = np.nonzero(explosion_map > 0)
        for x, y in zip(xs, ys):
            e = int(min(explosion_map[x, y], horizon))
            danger[1:e + 1, x, y] = True
    return danger


def bfs_safe_escape(pos: Tuple[int, int], field: np.ndarray, bombs, explosion_map: Optional[np.ndarray], others=(), horizon: int = HORIZON):
    """
    Time-expanded BFS over (tile, k). At each step the agent moves to a free
    neighbour or waits. Bombs, crates, walls always block; opponents block only
    for the first move (they move too).
    """
    danger = danger_tensor(bombs, field, explosion_map, horizon)
    passable = field == 0
    for (bx, by), _ in bombs:
        passable[bx, by] = False
    x0, y0 = pos
    opp_set = {tuple(o) for o in others}

    reach = np.zeros((horizon + 1,) + field.shape, dtype=np.int16)
    for i, (a, (dx, dy)) in enumerate(DIRS):
        nx, ny = x0 + dx, y0 + dy
        if passable[nx, ny] and (nx, ny) not in opp_set and not danger[1, nx, ny]:
            reach[1, nx, ny] |= (1 << i)
    if not danger[1, x0, y0]:
        reach[1, x0, y0] |= (1 << 4)

    future_danger = np.flip(np.cumsum(np.flip(danger, 0), axis=0), 0) > 0
    min_escape = INF
    if not future_danger[1, x0, y0]:
        min_escape = 0

    for k in range(1, horizon):
        xs, ys = np.nonzero(reach[k])
        if len(xs) == 0:
            break
        for x, y in zip(xs, ys):
            mask = reach[k, x, y]
            if not future_danger[k, x, y] and k < min_escape:
                min_escape = k
            if not danger[k + 1, x, y]:
                reach[k + 1, x, y] |= mask
            for _, (dx, dy) in DIRS:
                nx, ny = x + dx, y + dy
                if passable[nx, ny] and not danger[k + 1, nx, ny]:
                    reach[k + 1, nx, ny] |= mask
    final = 0
    xs, ys = np.nonzero(reach[horizon])
    for x, y in zip(xs, ys):
        final |= int(reach[horizon, x, y])
        if not future_danger[horizon, x, y] and horizon < min_escape:
            min_escape = horizon
    safe_actions = {a: bool(final & (1 << i)) for i, a in enumerate(MOVE_ACTIONS)}
    safe_actions['WAIT'] = bool(final & (1 << 4))
    if not any(safe_actions.values()):
        min_escape = INF
    return safe_actions, min_escape


def bfs_direction(pos: Tuple[int, int], field: np.ndarray, targets, obstacles=None):
    """
    Plain BFS from `pos` over free tiles to the nearest target.
    Returns (direction, distance): direction in MOVE_ACTIONS, 'WAIT' if already
    on a target, None if no target is reachable (distance INF).
    Target tiles may be entered even if they are in `obstacles` (e.g. opponents).
    """
    if not targets:
        return None, INF
    target_set = set(map(tuple, targets))
    if tuple(pos) in target_set:
        return 'WAIT', 0
    passable = field == 0
    if obstacles:
        for (ox, oy) in obstacles:
            passable[ox, oy] = False
    dist = np.full(field.shape, -1, dtype=np.int32)
    first = {}
    dist[pos] = 0
    q = deque()
    for a, (dx, dy) in DIRS:
        nx, ny = pos[0] + dx, pos[1] + dy
        if (nx, ny) in target_set:
            return a, 1
        if passable[nx, ny] and dist[nx, ny] < 0:
            dist[nx, ny] = 1
            first[(nx, ny)] = a
            q.append((nx, ny))
    while q:
        x, y = q.popleft()
        d = dist[x, y]
        for _, (dx, dy) in DIRS:
            nx, ny = x + dx, y + dy
            if dist[nx, ny] >= 0:
                continue
            if (nx, ny) in target_set:
                return first[(x, y)], d + 1
            if passable[nx, ny]:
                dist[nx, ny] = d + 1
                first[(nx, ny)] = first[(x, y)]
                q.append((nx, ny))
    return None, INF


def crate_adjacent_tiles(field: np.ndarray) -> List[Tuple[int, int]]:
    """Free tiles that have at least one crate as 4-neighbour."""
    crate = field == 1
    adj = np.zeros_like(crate)
    adj[1:, :] |= crate[:-1, :]
    adj[:-1, :] |= crate[1:, :]
    adj[:, 1:] |= crate[:, :-1]
    adj[:, :-1] |= crate[:, 1:]
    adj &= field == 0
    xs, ys = np.nonzero(adj)
    return list(zip(xs.tolist(), ys.tolist()))


def bomb_utility(pos: Tuple[int, int], field: np.ndarray, others, coins, bombs=(), explosion_map=None):
    """
    Value of dropping a bomb at `pos` right now.
    Returns dict with 'crates_hit', 'opponents_hit', 'coins_uncovered', 'traps_self'.
    """
    coords = blast_coords(pos, field)
    cset = set(coords)
    crates_hit = sum(1 for (x, y) in coords if field[x, y] == 1)
    opponents_hit = sum(1 for o in others if tuple(o) in cset)
    hypo_bombs = list(bombs) + [((pos[0], pos[1]), BOMB_TIMER)]
    safe, _ = bfs_safe_escape(pos, field, hypo_bombs, explosion_map, others=others)
    traps_self = not safe['WAIT']
    return {'crates_hit': crates_hit, 'opponents_hit': opponents_hit,
            'coins_uncovered': 0, 'traps_self': traps_self}


def state_to_features(game_state: Optional[dict], coordinate_history: Optional[deque] = None) -> Optional[np.ndarray]:
    """Full game_state dict -> float32 vector of length N_FEATURES (33-dim)."""
    if game_state is None or 'self' not in game_state:
        return None
    f = np.zeros(N_FEATURES, dtype=np.float32)
    field = game_state['field']
    bombs = game_state.get('bombs', [])
    explosion_map = game_state.get('explosion_map')
    _, _, bombs_left, (x, y) = game_state['self']
    others = [o[3] for o in game_state.get('others', [])]
    coins = game_state.get('coins', [])
    bomb_xy = {b[0] for b in bombs}
    opp_set = set(others)

    # --- A: danger / escape
    safe, _ = bfs_safe_escape((x, y), field, bombs, explosion_map, others)
    for i, a in enumerate(MOVE_ACTIONS):
        f[F_SAFE_MOVE + i] = 1.0 if safe[a] else 0.0
    f[F_SAFE_WAIT] = 1.0 if safe['WAIT'] else 0.0
    blast_map = get_blast_zones(bombs, field)
    f[F_IN_DANGER] = 1.0 if (blast_map[x, y] < INF or (explosion_map is not None and explosion_map[x, y] > 0)) else 0.0

    # Obstacles for path finding: bombs, burning tiles, opponents.
    burning = set()
    if explosion_map is not None:
        burning = set(zip(*[c.tolist() for c in np.nonzero(explosion_map > 0)]))
    path_obstacles = bomb_xy | burning | opp_set

    # --- B: coins
    d, dist = bfs_direction((x, y), field, coins, path_obstacles)
    if d in DIR_VECS:
        f[F_COIN_DIR + MOVE_ACTIONS.index(d)] = 1.0
    if dist < INF:
        f[F_COIN_PROX] = 1.0 / (1.0 + dist)

    # --- C: crates + bomb utility
    crate_targets = crate_adjacent_tiles(field)
    d, dist = bfs_direction((x, y), field, crate_targets, path_obstacles)
    if d in DIR_VECS:
        f[F_CRATE_DIR + MOVE_ACTIONS.index(d)] = 1.0
    util = bomb_utility((x, y), field, others, coins, bombs, explosion_map)
    if not util['traps_self']:
        f[F_BOMB_UTIL] = min(1.0, (util['crates_hit'] + 3 * util['opponents_hit']) / 6.0)

    # --- D, E
    f[F_TRAPS_SELF] = 1.0 if util['traps_self'] else 0.0
    f[F_BOMBS_LEFT] = 1.0 if bombs_left else 0.0

    # --- F: opponents
    d, dist = bfs_direction((x, y), field, others, bomb_xy | burning)
    if d in DIR_VECS:
        f[F_OPP_DIR + MOVE_ACTIONS.index(d)] = 1.0
    if dist < INF:
        f[F_OPP_PROX] = 1.0 / (1.0 + dist)

    # --- G: blocked neighbours
    for i, (a, (dx, dy)) in enumerate(DIRS):
        nx, ny = x + dx, y + dy
        if field[nx, ny] != 0 or (nx, ny) in bomb_xy or (nx, ny) in opp_set:
            f[F_BLOCKED + i] = 1.0

    # --- H: loop detector
    if coordinate_history is not None:
        if sum(1 for p in coordinate_history if p == (x, y)) > 2:
            f[F_VISITED] = 1.0
        for i, (a, (dx, dy)) in enumerate(DIRS):
            if sum(1 for p in coordinate_history if p == (x + dx, y + dy)) >= 2:
                f[F_NBR_VISITED + i] = 1.0
    f[F_BIAS] = 1.0
    return f


def valid_action_mask(features: np.ndarray) -> np.ndarray:
    """Boolean mask of length 6 derived strictly from game rules (blocked moves, bombs_left)."""
    mask = np.ones(N_ACTIONS, dtype=bool)
    mask[:4] = features[F_BLOCKED:F_BLOCKED + 4] < 0.5
    mask[ACTION_TO_IDX['BOMB']] = features[F_BOMBS_LEFT] > 0.5
    return mask


def state_to_spatial_tensor(
    game_state: Optional[dict],
    coordinate_history: Optional[deque] = None
) -> np.ndarray:
    """Extracts a (10, 17, 17) float32 tensor matching the trained neural network checkpoint."""
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



# ============================================================================
# Neural Network Architecture Modules
# ============================================================================

class DynActivation(nn.Module):
    """Custom learnable activation function initialized as exact Mish."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(x) * (self.alpha - self.beta) + self.beta * x


# Compatibility aliases
dynActivation = DynActivation
dynActvation = DynActivation


def load_matching_weights(model: nn.Module, state_dict: dict, partial: bool = True) -> Tuple[List[str], List[str]]:
    """Loads matching parameters from state_dict into model."""
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
        """Encodes raw spatial and feature inputs into the 256-dim fused latent vector."""
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


class LatentWorldModel(nn.Module):
    """
    Action-conditioned latent dynamics model predicting next fused representation,
    turn reward, long-term value, and legal action logits:
      z_{t+1} = z_t + Delta_psi(z_t, a_t)
      r_{t}   = r_psi(z_t, a_t)
      V_{t}   = V_psi(z_t, a_t)
      ell_t   = legal_head(z_{t+1})
    """
    def __init__(self, fused_dim: int = 256, n_actions: int = N_ACTIONS, hidden_dim: int = 256):
        super().__init__()
        self.fused_dim = fused_dim
        self.n_actions = n_actions

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

        # Immediate turn reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            DynActivation(),
            nn.Linear(64, 1),
        )

        # Long-term sequence value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            DynActivation(),
            nn.Linear(128, 1),
        )

        # Future legal moves head
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

        if z.size(0) == 1 and a_one_hot.size(0) > 1:
            z = z.expand(a_one_hot.size(0), -1)

        return torch.cat([z, a_one_hot], dim=-1)

    def forward(self, z: torch.Tensor, action: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (next_z, pred_reward, pred_value, pred_legal_logits).
        """
        x = self._prepare_input(z, action)
        h = self.trunk(x)
        delta_z = self.trans_head(h)
        next_z = (z if z.size(0) == delta_z.size(0) else z.expand(delta_z.size(0), -1)) + delta_z
        pred_rew = self.reward_head(h)
        pred_val = self.value_head(h)
        pred_legal = self.legal_head(next_z)
        return next_z, pred_rew, pred_val, pred_legal

    def predict_next(self, z: torch.Tensor, action: Any) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
        next_z, pred_rew, pred_val, pred_legal = self.forward(z, action)
        return next_z, float(pred_rew.squeeze().item()), float(pred_val.squeeze().item()), pred_legal

    def predict_legal(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.legal_head(z)


# ============================================================================
# Pure Neural Spatial QWM Agent (Inference Engine)
# ============================================================================

class SpatialQWMAgent:
    """
    Streamlined Pure Neural Spatial QWM Inference Agent.
    Implements:
      - Spatial Dueling DQN Q-network
      - Latent World Model (residual dynamics, reward, value, and action legality)
      - World Model Tree Search with Top-K Beam Pruning (Dong et al., 2026)
    """
    def __init__(self, device: Optional[torch.device] = None, **kwargs):
        self.device = device or get_device()
        self.kind = "spatial_qwm"
        self.policy_net = SpatialDuelingDQN().to(self.device)
        self.policy_net.eval()
        self.world_model = LatentWorldModel().to(self.device)
        self.world_model.eval()

    @torch.no_grad()
    def q_values(self, spatial: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Computes direct 6-action Q-values Q_phi(s, a) without tree search."""
        self.policy_net.eval()
        s = torch.as_tensor(spatial, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if s.dim() == 3:
            s = s.unsqueeze(0)
        if f.dim() == 1:
            f = f.unsqueeze(0)
        q = self.policy_net(s, f)
        return q.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def tree_search_action_scores(
        self,
        spatial: np.ndarray,
        features: np.ndarray,
        depth: int = 4,
        beam_size: int = 12,
        discount: float = 0.12,
        alpha_vq: float = 0.45,
        valid_mask: Optional[np.ndarray] = None,
        return_details: bool = False,
    ) -> Any:
        """
        Deep QWM Tree Search with Top-K Beam Pruning (Dong et al., Stanford 2026, Section 4).
        Performs multi-step imagined rollouts in the latent space of the World Model.
        Backward value aggregation follows Eq. (8) & (9) of Dong et al. (2026):
          V_r(d | z_d) = r_d + lambda * V(d+1 | z_{d+1})
          V(d | z_d)   = alpha_vq * V_Q(d | z_d) + (1 - alpha_vq) * V_r(d | z_d)
          Q_ts(s0, a0) = 0.5 * (Q_phi(s0, a0) + r_psi(s0, a0) + lambda * V(1 | z1(a0)))
        """
        self.policy_net.eval()
        self.world_model.eval()

        s = torch.as_tensor(spatial, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if s.dim() == 3:
            s = s.unsqueeze(0)
        if f.dim() == 1:
            f = f.unsqueeze(0)

        # 1. Encode root state into 256-dim fused latent representation
        z0 = self.policy_net.encode_fused(s, f)  # (1, 256)
        q0 = self.policy_net.forward_from_fused(z0).squeeze(0)  # (6,)

        depth = max(1, min(int(depth), 16))
        beam_size = max(6, int(beam_size))

        # 2. Depth 1 expansion: evaluate all 6 root actions
        a0 = torch.arange(N_ACTIONS, device=self.device, dtype=torch.int64)
        z0_rep = z0.expand(N_ACTIONS, -1)
        z1, r0, v0, legal0 = self.world_model(z0_rep, a0)
        r0_val = torch.nan_to_num(r0.squeeze(-1).clamp(-10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0)
        v0_val = torch.nan_to_num(v0.squeeze(-1).clamp(-50.0, 50.0), nan=0.0, posinf=50.0, neginf=-50.0)

        q1 = self.policy_net.forward_from_fused(z1)
        legal0_mask = (legal0 >= 0.0)
        legal0_mask[:, ACTION_TO_IDX['WAIT']] = True
        q1_masked = torch.where(legal0_mask, q1, torch.tensor(-1e9, device=self.device))
        v_q1 = q1_masked.max(dim=-1)[0]
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

        curr_z = z1
        curr_legal = legal0
        curr_root_idx = torch.arange(N_ACTIONS, device=self.device)
        curr_action_seqs = [[i] for i in range(N_ACTIONS)]
        curr_cum_r = r0_val.clone()
        curr_r_history = [r0_val]
        curr_vq_history = [v1_comb]

        best_action_seqs = None
        for d in range(1, depth):
            P = curr_z.size(0)
            z_exp = curr_z.repeat_interleave(N_ACTIONS, dim=0)
            a_exp = torch.arange(N_ACTIONS, device=self.device).repeat(P)

            cand_legality = curr_legal.reshape(-1)
            cand_legal_mask = (cand_legality >= 0.0) | (a_exp == ACTION_TO_IDX['WAIT'])

            next_z, next_r, next_v, next_legal = self.world_model(z_exp, a_exp)
            next_r_val = torch.nan_to_num(next_r.squeeze(-1).clamp(-10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0)
            next_v_val = torch.nan_to_num(next_v.squeeze(-1).clamp(-50.0, 50.0), nan=0.0, posinf=50.0, neginf=-50.0)

            next_q = self.policy_net.forward_from_fused(next_z)
            next_legal_mask = (next_legal >= 0.0)
            next_legal_mask[:, ACTION_TO_IDX['WAIT']] = True
            next_q_masked = torch.where(next_legal_mask, next_q, torch.tensor(-1e9, device=self.device))
            next_vq = next_q_masked.max(dim=-1)[0]
            next_comb_val = alpha_vq * next_vq + (1.0 - alpha_vq) * next_v_val

            cand_root_idx = curr_root_idx.repeat_interleave(N_ACTIONS)
            cand_action_seqs = [curr_action_seqs[p] + [a_idx] for p in range(P) for a_idx in range(N_ACTIONS)]
            cand_cum_r = curr_cum_r.repeat_interleave(N_ACTIONS) + (discount ** d) * next_r_val

            branch_score = cand_cum_r + (discount ** (d + 1)) * next_comb_val
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
                curr_r_history = [hist[parent_indices] for hist in curr_r_history] + [next_r_val[surv_idx]]
                curr_vq_history = [hist[parent_indices] for hist in curr_vq_history] + [next_comb_val[surv_idx]]
            else:
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
                V_curr = final_vq_hist[-1]
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

    def load(self, filepath: str, need_training: bool = False, partial: bool = True, **kwargs) -> Tuple[List[str], List[str]]:
        """Loads trained weights for policy net and world model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        loaded_keys = []
        uninit_keys = []

        # 1. Load policy net
        pol_dict = checkpoint.get('policy_state_dict', checkpoint)
        l_pol, u_pol = load_matching_weights(self.policy_net, pol_dict, partial=partial)
        loaded_keys.extend([f"policy.{k}" for k in l_pol])
        uninit_keys.extend([f"policy.{k}" for k in u_pol])

        # 2. Load world model if present
        if 'world_model_state_dict' in checkpoint:
            l_wm, u_wm = load_matching_weights(self.world_model, checkpoint['world_model_state_dict'], partial=partial)
            loaded_keys.extend([f"wm.{k}" for k in l_wm])
            uninit_keys.extend([f"wm.{k}" for k in u_wm])
        else:
            uninit_keys.extend([f"wm.{k}" for k in self.world_model.state_dict().keys()])

        self.policy_net.eval()
        self.world_model.eval()
        return loaded_keys, uninit_keys


def load_model(
    filepath: str,
    need_training: bool = False,
    partial: bool = False,
    device: Optional[torch.device] = None,
) -> SpatialQWMAgent:
    """Convenience factory function loading trained Spatial QWM agent checkpoint for inference."""
    agent = SpatialQWMAgent(device=device)
    agent.load(filepath, need_training=need_training, partial=partial)
    return agent
