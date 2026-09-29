# Inference Hyperparameter Tuning Report (Pure QWM Agent)

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Strategy:** `optuna`
- **Optuna Persistent DB:** `sqlite:////home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/optuna_study.db`
- **Optuna Study Name:** `qwm_agent_inference_tuning`
- **Timestamp:** `2026-09-19 19:53:14`
- **Total Matched Games:** 4000
- **Evaluation Duration:** 19160.8s

## 1. Undisputed Best Configuration

**Recommended Configuration:** `Optuna-222`

| Hyperparameter | Recommended Value | Description |
|:---|---:|:---|
| `search_depth` | **8** | Lookahead horizon / tree depth |
| `beam_size` | **36** | Beam search width |
| `tree_discount` | **0.080** | Lookahead discount factor $\lambda$ |
| `alpha_vq` | **0.720** | Blending weight $\alpha$ ($V_Q$ vs $V_r$) |

## 2. Cross-Scenario Composite Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Optuna-160** | 7 | 36 | 0.19 | 0.84 | **4592.5** | 59.6% | 75.8% | 7.67 | 6.12 | 0.31 | 233.2 |
| #2 | **Optuna-031** | 3 | 39 | 0.09 | 0.60 | **4456.5** | 54.4% | 82.3% | 7.23 | 6.23 | 0.20 | 70.1 |
| #3 | **Optuna-207** | 3 | 24 | 0.05 | 0.56 | **4422.4** | 54.4% | 80.2% | 7.89 | 6.37 | 0.30 | 62.1 |
| #4 | **Optuna-222** | 8 | 36 | 0.08 | 0.72 | **4404.2** | 54.4% | 79.2% | 8.05 | 6.80 | 0.25 | 242.3 |
| #5 | **Optuna-145** | 3 | 39 | 0.09 | 0.42 | **4334.7** | 57.9% | 67.1% | 7.64 | 6.55 | 0.22 | 68.5 |
| #6 | **Optuna-053** | 6 | 27 | 0.07 | 0.74 | **4308.1** | 53.3% | 77.1% | 7.89 | 6.49 | 0.28 | 190.0 |
| #7 | **Optuna-295** | 7 | 27 | 0.18 | 0.54 | **4299.0** | 50.2% | 84.8% | 7.29 | 6.30 | 0.20 | 199.1 |
| #8 | **Optuna-126** | 4 | 39 | 0.17 | 0.74 | **4296.7** | 54.8% | 73.3% | 7.11 | 6.11 | 0.20 | 108.7 |
| #9 | **Optuna-036** | 7 | 39 | 0.04 | 0.78 | **4271.7** | 51.5% | 80.0% | 7.87 | 6.13 | 0.35 | 235.3 |
| #10 | **Optuna-105** | 7 | 39 | 0.09 | 0.76 | **4269.6** | 48.5% | 87.5% | 7.31 | 6.08 | 0.25 | 235.8 |
| #11 | **Optuna-017** | 4 | 42 | 0.14 | 0.40 | **4260.9** | 55.4% | 69.6% | 7.87 | 5.93 | 0.39 | 106.4 |
| #12 | **Optuna-177** | 7 | 36 | 0.20 | 0.52 | **4234.5** | 51.2% | 78.5% | 8.05 | 6.26 | 0.36 | 236.4 |
| #13 | **Optuna-021** | 6 | 42 | 0.22 | 0.76 | **4199.7** | 49.0% | 82.7% | 7.79 | 5.92 | 0.38 | 190.1 |
| #14 | **Optuna-156** | 3 | 36 | 0.18 | 0.80 | **4188.0** | 50.6% | 78.1% | 7.51 | 5.83 | 0.34 | 68.2 |
| #15 | **Optuna-153** | 3 | 39 | 0.17 | 0.80 | **4181.1** | 51.7% | 74.8% | 8.12 | 6.27 | 0.37 | 69.6 |
| #16 | **Optuna-277** | 3 | 39 | 0.07 | 0.62 | **4169.2** | 50.8% | 76.5% | 7.78 | 6.41 | 0.27 | 61.0 |
| #17 | **Optuna-080** | 7 | 6 | 0.12 | 0.62 | **4167.2** | 50.2% | 78.1% | 7.45 | 6.14 | 0.26 | 229.6 |
| #18 | **Optuna-288** | 7 | 24 | 0.12 | 0.54 | **4165.6** | 47.5% | 84.8% | 7.57 | 5.74 | 0.36 | 201.2 |
| #19 | **Optuna-003** | 7 | 30 | 0.08 | 0.50 | **4164.2** | 50.8% | 76.5% | 7.38 | 6.07 | 0.26 | 236.7 |
| #20 | **Optuna-107** | 7 | 39 | 0.09 | 0.80 | **4139.5** | 51.9% | 72.7% | 7.22 | 6.07 | 0.23 | 235.0 |

## 3. Direct Combat Head-to-Head Tournament (44 Rounds)

| Place | Combatant | Direct Wins | Win % | Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Optuna-222** | 18 | **40.9%** | 68.2% | 3.75 | 0.23 | 2.61 |
| 🥈 #2 | **Optuna-031** | 14 | **31.8%** | 65.9% | 3.25 | 0.20 | 2.23 |
| 🥉 #3 | **Optuna-160** | 11 | **25.0%** | 72.7% | 2.98 | 0.14 | 2.30 |
| #4 | **Optuna-207** | 7 | **15.9%** | 63.6% | 2.09 | 0.05 | 1.86 |
