# Full-Board Spatial Dueling DQN Agent (`my_spatial_dqn_agent`)

This agent feeds the DQN network the **complete 17x17 board representation** as a multi-channel spatial tensor, fused with deterministic world-model safety features and anti-looping mechanisms.

---

## 1. Motivation

In feature-only agents (such as `my_wm_agent`), the DQN sees only 33 compressed scalar features (one-hot BFS directions, proximity scalars, danger flags). While fast, the network cannot observe the global maze topology, distant crate clusters, alternative coin avenues, or corridor choke-points.

`my_spatial_dqn_agent` solves this by giving the network:
1. **Full-board 17x17 visual field** via a multi-channel CNN torso.
2. **Immediate safety & physics rules** via 33 deterministic world-model features.
3. **Anti-looping & anti-stagnation** via arbitrary-length cycle detection ($k \in [2, 12]$) and immediate step-1 non-progress penalties.

---

## 2. Spatial Representation Layout (10 Channels, 17x17)

| Channel | Content | Encoding |
|:-------:|:--------|:---------|
| **0** | Stone Walls | $1.0$ where `field == -1` |
| **1** | Destructible Crates | $1.0$ where `field == 1` |
| **2** | Free Enterable Tiles | $1.0$ where tile is passable (no walls, bombs, or opponents) |
| **3** | Active Bombs & Blast Corridors | Intensity $(5 - \text{timer}) / 5.0$ raycasted along 4 axes up to radius 3 |
| **4** | Active Explosions | Normalized $\min(1.0, \text{explosion\_map} / 2.0)$ |
| **5** | Visible Coins | $1.0$ where coins exist |
| **6** | Self Position | $1.0$ at agent $(x, y)$ |
| **7** | Opponents | $1.0$ at other agent coordinates |
| **8** | History Heatmap | Normalized recent visit count $\min(1.0, \text{visits} / 4.0)$ |
| **9** | Global BFS Distance Gradient | Reachability gradient $1.0 / (1.0 + \text{dist})$ to nearest coin/crate across the whole maze |

---

## 3. Network Architecture: Dueling Spatial DQN

```
               Spatial Board (B, 10, 17, 17)
                             │
               Conv2d(10 -> 32, 3x3, pad 1) + ReLU
                             │
               Conv2d(32 -> 64, 3x3, pad 1) + ReLU
                             │
               Conv2d(64 -> 64, 3x3, pad 1) + ReLU
                             │
               Conv2d(64 -> 32, 1x1) + ReLU
                             │
               Flatten (32 * 17 * 17 = 9248)
                             │
               Linear(9248 -> 256) + ReLU  ─── Spatial Embed (256 dims)
                             │
       ┌─────────────────────┴─────────────────────┐
       │                                           │
Spatial Embed (256)                 Deterministic Features (33)
       │                                           │
       └─────────────────────┬─────────────────────┘
                             │
                     Concat (289 dims)
                             │
                     Linear(289 -> 256) + ReLU
                             │
              ┌──────────────┴──────────────┐
              │                             │
       Value Stream V(s)             Advantage Stream A(s, a)
       Linear(256 -> 128) + ReLU     Linear(256 -> 128) + ReLU
       Linear(128 -> 1)              Linear(128 -> 6)
              │                             │
              └──────────────┬──────────────┘
                             │
             Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
```

---

## 4. Key Training & Inference Features

1. **Double DQN with Polyak Target Updates**:
   Target network parameters $\theta_{\text{target}}$ are smoothly updated via $\theta_{\text{target}} \leftarrow \tau \theta + (1 - \tau) \theta_{\text{target}}$ ($\tau = 0.005$) on every training batch.
2. **Dihedral $D_4$ Symmetry Augmentation**:
   Every spatial transition $(S, F, A, R, S', F')$ is augmented across all 8 dihedral rotations and reflections (4 rotations $\times$ 2 reflections), delivering 8× effective training data and learning rotation/reflection invariant navigation.
3. **Anti-Looping & Escalating Non-Progress Penalty**:
   - `detect_cycle` evaluates arbitrary periodic cycles of length $k \in [2, 12]$ before moves are picked.
   - Non-progress penalty sets in directly on step 1 and scales up with each unproductive move.
   - History is cleared immediately whenever coins or crates are obtained.

---

## 5. Usage

### Run Unit Tests
```powershell
python -m unittest agent_code/my_spatial_dqn_agent/tests/test_spatial_dqn.py
```

### Play a Game
```powershell
python main.py play --agents my_spatial_dqn_agent --scenario coin-heaven
```

### Run Curriculum Training
```powershell
python agent_code/my_spatial_dqn_agent/train_curriculum.py
# Or specific stages:
python agent_code/my_spatial_dqn_agent/train_curriculum.py --stages 2 --s1 300 --s2 600
```

### Plot Training Progress Curves
Training curves are automatically plotted and updated live during training into `training_curves-spatial-dqn.png` and `training_curves.png` in both the run directory and the agent folder.

To manually plot from any CSV:
```powershell
python agent_code/my_spatial_dqn_agent/plot_training.py
# Or specific file / output path:
python agent_code/my_spatial_dqn_agent/plot_training.py --csv agent_code/my_spatial_dqn_agent/training_stats-spatial-dqn.csv --out training_curves.png
```
