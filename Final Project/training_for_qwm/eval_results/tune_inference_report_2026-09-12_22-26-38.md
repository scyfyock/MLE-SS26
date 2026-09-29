# Inference Hyperparameter Tuning Report

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Strategy:** `optuna`
- **Optuna Persistent DB:** `sqlite:////home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/optuna_study.db`
- **Optuna Study Name:** `qwm_inference_tuning`
- **Timestamp:** `2026-09-12 22:26:38`
- **Total Matched Games:** 4000
- **Evaluation Duration:** 19417.6s

## 1. Undisputed Best Configuration

**Recommended Configuration:** `Optuna-112`

| Hyperparameter | Recommended Value | Description |
|:---|---:|:---|
| `search_depth` | **5** | Lookahead horizon / tree depth |
| `beam_size` | **18** | Beam search width |
| `tree_discount` | **0.080** | Lookahead discount factor $\lambda$ |
| `alpha_vq` | **0.160** | Blending weight $\alpha$ ($V_Q$ vs $V_r$) |
| `predicted_wait_penalty` | **1.25** | Penalty for predicted wait actions |
| `predicted_loop_penalty` | **1.50** | Penalty for revisiting recent coordinates |

## 2. Cross-Scenario Composite Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Optuna-112** | 5 | 18 | 0.08 | 0.16 | **4016.6** | 57.9% | 80.2% | 8.22 | 6.36 | 0.37 | 0.0 |
| #2 | **Optuna-091** | 7 | 12 | 0.05 | 0.52 | **4002.1** | 56.9% | 81.5% | 8.40 | 6.47 | 0.39 | 0.0 |
| #3 | **Optuna-057** | 5 | 18 | 0.02 | 0.20 | **3870.6** | 53.8% | 81.5% | 7.80 | 6.37 | 0.29 | 0.0 |
| #4 | **Optuna-026** | 11 | 18 | 0.36 | 0.32 | **3866.7** | 58.3% | 72.5% | 7.14 | 5.68 | 0.29 | 0.0 |
| #5 | **Optuna-049** | 6 | 12 | 0.24 | 0.18 | **3827.2** | 52.5% | 81.2% | 8.70 | 7.08 | 0.33 | 0.0 |
| #6 | **Optuna-061** | 5 | 18 | 0.11 | 0.22 | **3810.4** | 57.5% | 70.8% | 8.07 | 6.06 | 0.40 | 0.0 |
| #7 | **Optuna-042** | 6 | 12 | 0.31 | 0.46 | **3751.2** | 49.2% | 84.2% | 8.57 | 7.74 | 0.17 | 0.0 |
| #8 | **Optuna-087** | 7 | 18 | 0.05 | 0.48 | **3740.9** | 51.2% | 80.2% | 7.37 | 6.48 | 0.18 | 0.0 |
| #9 | **Optuna-074** | 5 | 18 | 0.11 | 0.22 | **3731.0** | 53.1% | 75.8% | 7.61 | 6.46 | 0.23 | 0.0 |
| #10 | **Optuna-082** | 6 | 18 | 0.36 | 0.54 | **3706.6** | 51.0% | 78.5% | 8.07 | 6.04 | 0.41 | 0.0 |
| #11 | **Optuna-108** | 5 | 18 | 0.07 | 0.18 | **3670.6** | 52.7% | 73.3% | 8.16 | 6.72 | 0.29 | 0.0 |
| #12 | **Optuna-103** | 8 | 18 | 0.07 | 0.46 | **3650.5** | 53.8% | 70.6% | 7.56 | 6.12 | 0.29 | 0.0 |
| #13 | **Optuna-044** | 7 | 12 | 0.36 | 0.26 | **3645.9** | 40.8% | 95.8% | 8.16 | 6.20 | 0.39 | 0.0 |
| #14 | **Optuna-014** | 9 | 30 | 0.32 | 0.36 | **3621.6** | 51.2% | 73.8% | 8.31 | 6.33 | 0.40 | 0.0 |
| #15 | **Optuna-063** | 5 | 18 | 0.11 | 0.22 | **3620.7** | 48.5% | 79.6% | 7.45 | 6.20 | 0.25 | 0.0 |
| #16 | **Optuna-129** | 4 | 18 | 0.00 | 0.24 | **3583.6** | 46.0% | 82.7% | 7.53 | 5.83 | 0.34 | 0.0 |
| #17 | **Optuna-124** | 8 | 12 | 0.38 | 0.22 | **3549.8** | 51.2% | 70.2% | 8.23 | 6.30 | 0.39 | 0.0 |
| #18 | **Optuna-148** | 5 | 18 | 0.15 | 0.28 | **3543.6** | 51.2% | 70.0% | 8.03 | 6.70 | 0.27 | 0.0 |
| #19 | **Optuna-060** | 5 | 12 | 0.17 | 0.10 | **3537.0** | 51.7% | 69.2% | 7.46 | 6.33 | 0.23 | 0.0 |
| #20 | **Optuna-070** | 8 | 12 | 0.19 | 0.22 | **3532.3** | 51.0% | 70.2% | 7.43 | 6.16 | 0.25 | 0.0 |

## 3. Direct Combat Head-to-Head Tournament (21 Rounds)

| Place | Combatant | Direct Wins | Win % | Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Optuna-112** | 9 | **42.9%** | 57.1% | 3.29 | 0.14 | 2.57 |
| 🥈 #2 | **Optuna-057** | 7 | **33.3%** | 57.1% | 2.71 | 0.10 | 2.24 |
| 🥉 #3 | **Optuna-091** | 6 | **28.6%** | 66.7% | 2.71 | 0.14 | 2.00 |
| #4 | **Optuna-026** | 4 | **19.0%** | 23.8% | 1.86 | 0.05 | 1.62 |
