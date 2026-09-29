# Inference Hyperparameter Tuning Report (Pure QWM Agent)

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Strategy:** `optuna`
- **Optuna Persistent DB:** `sqlite:////home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/optuna_study.db`
- **Optuna Study Name:** `qwm_agent_inference_tuning`
- **Timestamp:** `2026-09-13 00:15:04`
- **Total Matched Games:** 2
- **Evaluation Duration:** 12.0s

## 1. Undisputed Best Configuration

**Recommended Configuration:** `Optuna-001`

| Hyperparameter | Recommended Value | Description |
|:---|---:|:---|
| `search_depth` | **2** | Lookahead horizon / tree depth |
| `beam_size` | **12** | Beam search width |
| `tree_discount` | **0.070** | Lookahead discount factor $\lambda$ |
| `alpha_vq` | **0.830** | Blending weight $\alpha$ ($V_Q$ vs $V_r$) |

## 2. Cross-Scenario Composite Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Optuna-000** | 4 | 48 | 0.31 | 0.59 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
| #2 | **Optuna-001** | 2 | 12 | 0.07 | 0.83 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |

## 3. Direct Combat Head-to-Head Tournament (2 Rounds)

| Place | Combatant | Direct Wins | Win % | Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Optuna-001** | 1 | **50.0%** | 100.0% | 3.50 | 0.00 | 3.50 |
| 🥈 #2 | **Optuna-000** | 0 | **0.0%** | 50.0% | 2.00 | 0.00 | 2.00 |
| 🥉 #3 | **my_spatial_dqn_agent** | 0 | **0.0%** | 0.0% | 2.00 | 0.00 | 2.00 |
