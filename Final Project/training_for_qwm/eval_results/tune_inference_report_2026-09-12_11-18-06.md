# Inference Hyperparameter Tuning Report

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Strategy:** `optuna`
- **Timestamp:** `2026-09-12 11:18:06`
- **Total Matched Games:** 6
- **Evaluation Duration:** 56.4s

## 1. Undisputed Best Configuration

**Recommended Configuration:** `Optuna-002`

| Hyperparameter | Recommended Value | Description |
|:---|---:|:---|
| `search_depth` | **7** | Lookahead horizon / tree depth |
| `beam_size` | **12** | Beam search width |
| `tree_discount` | **0.110** | Lookahead discount factor $\lambda$ |
| `alpha_vq` | **0.240** | Blending weight $\alpha$ ($V_Q$ vs $V_r$) |
| `predicted_wait_penalty` | **1.50** | Penalty for predicted wait actions |
| `predicted_loop_penalty` | **4.00** | Penalty for revisiting recent coordinates |

## 2. Cross-Scenario Composite Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Optuna-002** | 7 | 12 | 0.11 | 0.24 | **4042.0** | 50.0% | 100.0% | 3.50 | 3.50 | 0.00 | 0.0 |
| #2 | **Optuna-000** | 3 | 36 | 0.31 | 0.58 | **2025.0** | 50.0% | 0.0% | 2.50 | 2.50 | 0.00 | 0.0 |
| #3 | **Optuna-001** | 1 | 36 | 0.26 | 0.68 | **1003.5** | 0.0% | 50.0% | 0.50 | 0.50 | 0.00 | 0.0 |

## 3. Direct Combat Head-to-Head Tournament (2 Rounds)

| Place | Combatant | Direct Wins | Win % | Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Optuna-002** | 1 | **50.0%** | 50.0% | 3.50 | 0.00 | 3.50 |
| 🥈 #2 | **Optuna-001** | 1 | **50.0%** | 100.0% | 2.50 | 0.00 | 2.50 |
| 🥉 #3 | **my_spatial_dqn_agent** | 0 | **0.0%** | 0.0% | 2.00 | 0.00 | 2.00 |
| #4 | **Optuna-000** | 0 | **0.0%** | 50.0% | 0.50 | 0.00 | 0.50 |
