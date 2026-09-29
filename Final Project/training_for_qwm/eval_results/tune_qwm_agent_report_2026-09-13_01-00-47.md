# Inference Hyperparameter Tuning Report (Pure QWM Agent)

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Strategy:** `optuna`
- **Optuna Persistent DB:** `sqlite:////home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/optuna_study.db`
- **Optuna Study Name:** `qwm_agent_inference_tuning`
- **Timestamp:** `2026-09-13 01:00:47`
- **Total Matched Games:** 1
- **Evaluation Duration:** 11.2s

## 1. Undisputed Best Configuration

**Recommended Configuration:** `Optuna-012`

| Hyperparameter | Recommended Value | Description |
|:---|---:|:---|
| `search_depth` | **8** | Lookahead horizon / tree depth |
| `beam_size` | **24** | Beam search width |
| `tree_discount` | **0.050** | Lookahead discount factor $\lambda$ |
| `alpha_vq` | **0.360** | Blending weight $\alpha$ ($V_Q$ vs $V_r$) |

## 2. Cross-Scenario Composite Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Optuna-012** | 8 | 24 | 0.05 | 0.36 | **7094.0** | 100.0% | 100.0% | 8.00 | 3.00 | 1.00 | 20.9 |
| #2 | **Optuna-000** | 6 | 42 | 0.09 | 0.00 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
| #3 | **Optuna-001** | 10 | 48 | 0.38 | 0.24 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
| #4 | **Optuna-002** | 3 | 12 | 0.02 | 0.54 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
| #5 | **Optuna-003** | 5 | 27 | 0.12 | 0.16 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
| #6 | **Optuna-004** | 11 | 18 | 0.18 | 0.00 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
| #7 | **Optuna-005** | 12 | 12 | 0.10 | 0.94 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
| #8 | **Optuna-006** | 5 | 42 | 0.16 | 0.92 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
| #9 | **Optuna-007** | 6 | 45 | 0.20 | 0.82 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
| #10 | **Optuna-008** | 4 | 45 | 0.27 | 0.38 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
| #11 | **Optuna-009** | 3 | 18 | 0.12 | 0.26 | **0.0** | 0.0% | 0.0% | 0.00 | 0.00 | 0.00 | 0.0 |
