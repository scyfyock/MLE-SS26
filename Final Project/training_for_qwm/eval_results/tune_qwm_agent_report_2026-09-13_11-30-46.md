# Inference Hyperparameter Tuning Report (Pure QWM Agent)

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Strategy:** `optuna`
- **Optuna Persistent DB:** `sqlite:////home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/optuna_study.db`
- **Optuna Study Name:** `qwm_agent_inference_tuning`
- **Timestamp:** `2026-09-13 11:30:46`
- **Total Matched Games:** 8000
- **Evaluation Duration:** 36372.5s

## 1. Undisputed Best Configuration

**Recommended Configuration:** `Optuna-145`

| Hyperparameter | Recommended Value | Description |
|:---|---:|:---|
| `search_depth` | **3** | Lookahead horizon / tree depth |
| `beam_size` | **39** | Beam search width |
| `tree_discount` | **0.090** | Lookahead discount factor $\lambda$ |
| `alpha_vq` | **0.420** | Blending weight $\alpha$ ($V_Q$ vs $V_r$) |

## 2. Cross-Scenario Composite Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Optuna-160** | 7 | 36 | 0.19 | 0.84 | **4592.5** | 59.6% | 75.8% | 7.67 | 6.12 | 0.31 | 233.2 |
| #2 | **Optuna-031** | 3 | 39 | 0.09 | 0.60 | **4456.5** | 54.4% | 82.3% | 7.23 | 6.23 | 0.20 | 70.1 |
| #3 | **Optuna-145** | 3 | 39 | 0.09 | 0.42 | **4334.7** | 57.9% | 67.1% | 7.64 | 6.55 | 0.22 | 68.5 |
| #4 | **Optuna-053** | 6 | 27 | 0.07 | 0.74 | **4308.1** | 53.3% | 77.1% | 7.89 | 6.49 | 0.28 | 190.0 |
| #5 | **Optuna-126** | 4 | 39 | 0.17 | 0.74 | **4296.7** | 54.8% | 73.3% | 7.11 | 6.11 | 0.20 | 108.7 |
| #6 | **Optuna-036** | 7 | 39 | 0.04 | 0.78 | **4271.7** | 51.5% | 80.0% | 7.87 | 6.13 | 0.35 | 235.3 |
| #7 | **Optuna-105** | 7 | 39 | 0.09 | 0.76 | **4269.6** | 48.5% | 87.5% | 7.31 | 6.08 | 0.25 | 235.8 |
| #8 | **Optuna-017** | 4 | 42 | 0.14 | 0.40 | **4260.9** | 55.4% | 69.6% | 7.87 | 5.93 | 0.39 | 106.4 |
| #9 | **Optuna-177** | 7 | 36 | 0.20 | 0.52 | **4234.5** | 51.2% | 78.5% | 8.05 | 6.26 | 0.36 | 236.4 |
| #10 | **Optuna-021** | 6 | 42 | 0.22 | 0.76 | **4199.7** | 49.0% | 82.7% | 7.79 | 5.92 | 0.38 | 190.1 |
| #11 | **Optuna-156** | 3 | 36 | 0.18 | 0.80 | **4188.0** | 50.6% | 78.1% | 7.51 | 5.83 | 0.34 | 68.2 |
| #12 | **Optuna-153** | 3 | 39 | 0.17 | 0.80 | **4181.1** | 51.7% | 74.8% | 8.12 | 6.27 | 0.37 | 69.6 |
| #13 | **Optuna-080** | 7 | 6 | 0.12 | 0.62 | **4167.2** | 50.2% | 78.1% | 7.45 | 6.14 | 0.26 | 229.6 |
| #14 | **Optuna-003** | 7 | 30 | 0.08 | 0.50 | **4164.2** | 50.8% | 76.5% | 7.38 | 6.07 | 0.26 | 236.7 |
| #15 | **Optuna-107** | 7 | 39 | 0.09 | 0.80 | **4139.5** | 51.9% | 72.7% | 7.22 | 6.07 | 0.23 | 235.0 |
| #16 | **Optuna-118** | 8 | 39 | 0.20 | 0.06 | **4130.1** | 51.7% | 72.7% | 7.34 | 5.89 | 0.29 | 281.0 |
| #17 | **Optuna-117** | 5 | 42 | 0.30 | 0.72 | **4115.8** | 50.8% | 74.4% | 6.91 | 5.34 | 0.31 | 146.7 |
| #18 | **Optuna-171** | 8 | 39 | 0.21 | 0.12 | **4112.6** | 55.4% | 62.7% | 6.89 | 5.88 | 0.20 | 280.0 |
| #19 | **Optuna-138** | 4 | 12 | 0.20 | 0.76 | **4102.2** | 49.6% | 76.5% | 7.39 | 6.30 | 0.22 | 103.2 |
| #20 | **Optuna-174** | 8 | 39 | 0.22 | 0.00 | **4090.8** | 51.0% | 72.2% | 7.48 | 6.38 | 0.22 | 281.5 |

## 3. Direct Combat Head-to-Head Tournament (88 Rounds)

| Place | Combatant | Direct Wins | Win % | Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Optuna-145** | 30 | **34.1%** | 60.2% | 3.20 | 0.17 | 2.35 |
| 🥈 #2 | **Optuna-031** | 28 | **31.8%** | 51.1% | 2.95 | 0.18 | 2.05 |
| 🥉 #3 | **Optuna-053** | 25 | **28.4%** | 48.9% | 3.27 | 0.22 | 2.19 |
| #4 | **Optuna-160** | 25 | **28.4%** | 60.2% | 3.15 | 0.17 | 2.30 |
