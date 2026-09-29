# Inference Hyperparameter Tuning Report

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Strategy:** `optuna`
- **Optuna Persistent DB:** `sqlite:////home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/optuna_study.db`
- **Optuna Study Name:** `qwm_inference_tuning`
- **Timestamp:** `2026-09-12 11:24:01`
- **Total Matched Games:** 1
- **Evaluation Duration:** 14.8s

## 1. Undisputed Best Configuration

**Recommended Configuration:** `Optuna-001`

| Hyperparameter | Recommended Value | Description |
|:---|---:|:---|
| `search_depth` | **3** | Lookahead horizon / tree depth |
| `beam_size` | **36** | Beam search width |
| `tree_discount` | **0.310** | Lookahead discount factor $\lambda$ |
| `alpha_vq` | **0.580** | Blending weight $\alpha$ ($V_Q$ vs $V_r$) |
| `predicted_wait_penalty` | **0.75** | Penalty for predicted wait actions |
| `predicted_loop_penalty` | **1.00** | Penalty for revisiting recent coordinates |

## 2. Cross-Scenario Composite Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Optuna-001** | 3 | 36 | 0.31 | 0.58 | **6091.0** | 100.0% | 100.0% | 8.00 | 3.00 | 1.00 | 0.0 |
| #2 | **Optuna-003** | 3 | 36 | 0.31 | 0.58 | **6079.0** | 100.0% | 100.0% | 7.00 | 2.00 | 1.00 | 0.0 |
| #3 | **Optuna-002** | 3 | 36 | 0.31 | 0.58 | **2012.0** | 0.0% | 100.0% | 1.00 | 1.00 | 0.00 | 0.0 |
