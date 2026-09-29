# Inference Hyperparameter Tuning Report

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Strategy:** `optuna`
- **Optuna Persistent DB:** `sqlite:////home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/optuna_study.db`
- **Optuna Study Name:** `qwm_inference_tuning`
- **Timestamp:** `2026-09-12 16:38:02`
- **Total Matched Games:** 1000
- **Evaluation Duration:** 5418.9s

## 1. Undisputed Best Configuration

**Recommended Configuration:** `Optuna-042`

| Hyperparameter | Recommended Value | Description |
|:---|---:|:---|
| `search_depth` | **6** | Lookahead horizon / tree depth |
| `beam_size` | **12** | Beam search width |
| `tree_discount` | **0.310** | Lookahead discount factor $\lambda$ |
| `alpha_vq` | **0.460** | Blending weight $\alpha$ ($V_Q$ vs $V_r$) |
| `predicted_wait_penalty` | **0.50** | Penalty for predicted wait actions |
| `predicted_loop_penalty` | **2.00** | Penalty for revisiting recent coordinates |

## 2. Cross-Scenario Composite Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Optuna-026** | 11 | 18 | 0.36 | 0.32 | **3866.7** | 58.3% | 72.5% | 7.14 | 5.68 | 0.29 | 0.0 |
| #2 | **Optuna-049** | 6 | 12 | 0.24 | 0.18 | **3827.2** | 52.5% | 81.2% | 8.70 | 7.08 | 0.33 | 0.0 |
| #3 | **Optuna-042** | 6 | 12 | 0.31 | 0.46 | **3751.2** | 49.2% | 84.2% | 8.57 | 7.74 | 0.17 | 0.0 |
| #4 | **Optuna-044** | 7 | 12 | 0.36 | 0.26 | **3645.9** | 40.8% | 95.8% | 8.16 | 6.20 | 0.39 | 0.0 |
| #5 | **Optuna-014** | 9 | 30 | 0.32 | 0.36 | **3621.6** | 51.2% | 73.8% | 8.31 | 6.33 | 0.40 | 0.0 |
| #6 | **Optuna-019** | 9 | 6 | 0.27 | 0.38 | **3389.7** | 42.9% | 79.2% | 7.61 | 6.36 | 0.25 | 0.0 |
| #7 | **Optuna-010** | 5 | 6 | 0.38 | 0.46 | **3375.3** | 48.3% | 68.8% | 5.69 | 5.38 | 0.06 | 0.0 |
| #8 | **Optuna-025** | 6 | 12 | 0.31 | 0.44 | **3367.8** | 41.7% | 81.2% | 6.50 | 5.56 | 0.19 | 0.0 |
| #9 | **Optuna-041** | 5 | 12 | 0.27 | 0.30 | **3366.1** | 49.2% | 65.4% | 7.75 | 6.92 | 0.17 | 0.0 |
| #10 | **Optuna-024** | 10 | 6 | 0.23 | 0.34 | **3242.2** | 39.6% | 79.2% | 6.46 | 5.30 | 0.23 | 0.0 |
| #11 | **Optuna-022** | 10 | 6 | 0.29 | 0.48 | **3158.8** | 43.8% | 66.2% | 7.18 | 5.93 | 0.25 | 0.0 |
| #12 | **Optuna-036** | 10 | 12 | 0.32 | 0.46 | **3141.3** | 42.1% | 68.8% | 7.16 | 5.75 | 0.28 | 0.0 |
| #13 | **Optuna-043** | 6 | 6 | 0.30 | 0.44 | **3086.9** | 37.5% | 75.8% | 6.01 | 5.08 | 0.19 | 0.0 |
| #14 | **Optuna-011** | 3 | 30 | 0.10 | 0.40 | **3015.7** | 42.1% | 62.5% | 7.05 | 5.90 | 0.23 | 0.0 |
| #15 | **Optuna-021** | 9 | 12 | 0.25 | 0.38 | **3014.9** | 45.0% | 57.1% | 6.26 | 5.43 | 0.17 | 0.0 |
| #16 | **Optuna-008** | 3 | 30 | 0.09 | 0.12 | **3014.0** | 39.6% | 67.5% | 6.93 | 5.26 | 0.33 | 0.0 |
| #17 | **Optuna-032** | 9 | 12 | 0.37 | 0.28 | **2949.2** | 40.8% | 62.5% | 5.71 | 4.88 | 0.17 | 0.0 |
| #18 | **Optuna-020** | 8 | 36 | 0.34 | 0.60 | **2869.6** | 41.7% | 55.8% | 7.46 | 6.21 | 0.25 | 0.0 |
| #19 | **Optuna-033** | 6 | 6 | 0.26 | 0.52 | **2855.5** | 40.8% | 57.1% | 6.90 | 6.27 | 0.13 | 0.0 |
| #20 | **Optuna-029** | 11 | 18 | 0.30 | 0.22 | **2833.7** | 35.4% | 67.1% | 6.53 | 5.18 | 0.27 | 0.0 |

## 3. Direct Combat Head-to-Head Tournament (2 Rounds)

| Place | Combatant | Direct Wins | Win % | Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Optuna-042** | 1 | **50.0%** | 100.0% | 4.50 | 0.00 | 4.50 |
| 🥈 #2 | **Optuna-044** | 1 | **50.0%** | 0.0% | 3.50 | 0.50 | 1.00 |
| 🥉 #3 | **Optuna-049** | 1 | **50.0%** | 100.0% | 3.00 | 0.00 | 3.00 |
| #4 | **Optuna-026** | 1 | **50.0%** | 0.0% | 3.00 | 0.50 | 0.50 |
