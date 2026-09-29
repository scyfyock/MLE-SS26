# Inference Hyperparameter Tuning Report

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Timestamp:** `2026-09-12 00:01:30`
- **Total Matched Games:** 8
- **Evaluation Duration:** 33.2s

## 1. Undisputed Best Configuration

**Recommended Configuration:** `Standard-D3-B12`

| Hyperparameter | Recommended Value | Description |
|:---|---:|:---|
| `search_depth` | **3** | Lookahead tree depth |
| `beam_size` | **12** | Beam search width |
| `tree_discount` | **0.120** | Lookahead discount factor $\lambda$ |
| `alpha_vq` | **0.450** | Blending weight $\alpha$ ($V_Q$ vs $V_r$) |
| `predicted_wait_penalty` | **1.00** | Penalty for predicted wait actions |
| `predicted_loop_penalty` | **2.00** | Penalty for revisiting recent coordinates |

## 2. Cross-Scenario Composite Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Deep-D4-B12** | 4 | 12 | 0.12 | 0.40 | **5033.5** | 100.0% | 50.0% | 3.00 | 3.00 | 0.00 | 0.0 |
| #2 | **ValueHeavy-D3** | 3 | 12 | 0.15 | 0.60 | **3076.5** | 50.0% | 50.0% | 7.00 | 2.00 | 1.00 | 0.0 |
| #3 | **Standard-D3-B12** | 3 | 12 | 0.12 | 0.45 | **3030.0** | 50.0% | 50.0% | 2.50 | 2.50 | 0.00 | 0.0 |
| #4 | **Fast-D2-B6** | 2 | 6 | 0.15 | 0.45 | **2057.5** | 50.0% | 0.0% | 5.00 | 5.00 | 0.00 | 0.0 |

## 3. Direct Combat Head-to-Head Tournament (2 Rounds)

| Place | Combatant | Direct Wins | Win % | Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Standard-D3-B12** | 1 | **50.0%** | 100.0% | 5.50 | 0.50 | 3.00 |
| 🥈 #2 | **Fast-D2-B6** | 1 | **50.0%** | 50.0% | 4.50 | 0.50 | 2.00 |
| 🥉 #3 | **Deep-D4-B12** | 0 | **0.0%** | 50.0% | 5.50 | 0.50 | 3.00 |
| #4 | **ValueHeavy-D3** | 0 | **0.0%** | 50.0% | 1.00 | 0.00 | 1.00 |
