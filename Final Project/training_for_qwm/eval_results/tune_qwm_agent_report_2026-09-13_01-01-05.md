# Inference Hyperparameter Tuning Report (Pure QWM Agent)

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Strategy:** `optuna`
- **Optuna Persistent DB:** `sqlite:////home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/optuna_study.db`
- **Optuna Study Name:** `qwm_agent_inference_tuning`
- **Timestamp:** `2026-09-13 01:01:05`
- **Total Matched Games:** 2
- **Evaluation Duration:** 13.5s

## 1. Undisputed Best Configuration

**Recommended Configuration:** `Optuna-000`

| Hyperparameter | Recommended Value | Description |
|:---|---:|:---|
| `search_depth` | **6** | Lookahead horizon / tree depth |
| `beam_size` | **48** | Beam search width |
| `tree_discount` | **0.300** | Lookahead discount factor $\lambda$ |
| `alpha_vq` | **0.580** | Blending weight $\alpha$ ($V_Q$ vs $V_r$) |

## 2. Cross-Scenario Composite Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Optuna-000** | 6 | 48 | 0.30 | 0.58 | **2039.0** | 0.0% | 100.0% | 3.00 | 3.00 | 0.00 | 17.1 |
| #2 | **Optuna-001** | 4 | 12 | 0.02 | 0.84 | **13.0** | 0.0% | 0.0% | 1.00 | 1.00 | 0.00 | 22.8 |
