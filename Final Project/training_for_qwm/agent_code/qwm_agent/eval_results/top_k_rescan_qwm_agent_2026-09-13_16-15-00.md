# Top-K Candidate Rescan & Tournament Report (`qwm_agent`)

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Agent Architecture:** `qwm_agent` (Pure Neural World Model)
- **Timestamp:** `2026-09-13 16:15:00`
- **Candidates Benchmarked:** 4
- **Total Phase 1 Matches:** 2000

## 1. Executive Champion Recommendation

**Undisputed Winner:** `Top1-D7-B36` (Phase 2 Direct Combat Tournament)

| Hyperparameter | Optimal Value | Parameter Purpose |
|:---|---:|:---|
| `search_depth` | **7** | Lookahead horizon steps |
| `beam_size` | **36** | Beam search pruning width |
| `tree_discount` ($\lambda$) | **0.190** | Lookahead prospective reward discount |
| `alpha_vq` ($\alpha$) | **0.840** | Balance between Q-critic and prospective return |

---

## 2. Phase 1: Explicit Per-Scenario Performance Breakdowns

> [!NOTE]
> Wins are explicitly separated into **Point Wins** (highest score) and **Survival Wins** (sole survivor).

### Scenario: Classic vs 3x Rule-Based (`classic`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top2-D3-B39** | 48.0% (48) | 39.0% (39) | 13.0% (13) | 67.0% | 4.22 | 2.92 | 0.26 | 28.4 | 69.1 |
| #2 | **Top1-D7-B36** | 47.0% (47) | 30.0% (30) | 16.0% (16) | 66.0% | 4.09 | 2.79 | 0.26 | 27.7 | 232.9 |
| #3 | **Top4-D6-B27** | 45.0% (45) | 30.0% (30) | 11.0% (11) | 55.0% | 3.73 | 2.53 | 0.24 | 25.5 | 185.0 |
| #4 | **Top3-D3-B39** | 38.0% (38) | 32.0% (32) | 10.0% (10) | 66.0% | 3.70 | 2.60 | 0.22 | 27.4 | 67.1 |

### Scenario: DQN Rivalry (1x Spatial DQN + 2x Rule-Based) (`classic`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D7-B36** | 50.0% (50) | 32.0% (32) | 24.0% (24) | 71.0% | 3.68 | 2.58 | 0.22 | 28.2 | 233.4 |
| #2 | **Top4-D6-B27** | 49.0% (49) | 32.0% (32) | 22.0% (22) | 62.0% | 3.96 | 2.41 | 0.31 | 26.4 | 185.7 |
| #3 | **Top3-D3-B39** | 46.0% (46) | 28.0% (28) | 20.0% (20) | 67.0% | 3.56 | 2.41 | 0.23 | 27.4 | 65.5 |
| #4 | **Top2-D3-B39** | 43.0% (43) | 23.0% (23) | 21.0% (21) | 63.0% | 3.35 | 2.30 | 0.21 | 27.2 | 66.3 |

### Scenario: Spatial Clash (2x Spatial DQN + 1x Rule-Based) (`classic`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D7-B36** | 55.0% (55) | 33.0% (33) | 31.0% (31) | 69.0% | 4.02 | 2.37 | 0.33 | 28.4 | 232.2 |
| #2 | **Top2-D3-B39** | 54.0% (54) | 28.0% (28) | 35.0% (35) | 65.0% | 3.47 | 2.22 | 0.25 | 26.7 | 65.0 |
| #3 | **Top4-D6-B27** | 47.0% (47) | 15.0% (15) | 31.0% (31) | 70.0% | 2.67 | 2.27 | 0.08 | 27.7 | 186.3 |
| #4 | **Top3-D3-B39** | 43.0% (43) | 24.0% (24) | 29.0% (29) | 67.0% | 3.68 | 2.58 | 0.22 | 28.6 | 65.8 |

### Scenario: Loot Crate Mixed (Spatial DQN + Rule-Based + Coin Collector) (`loot-crate`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top3-D3-B39** | 42.0% (42) | 25.0% (25) | 21.0% (21) | 70.0% | 14.20 | 12.75 | 0.29 | 24.8 | 66.8 |
| #2 | **Top2-D3-B39** | 40.0% (40) | 26.0% (26) | 17.0% (17) | 66.0% | 13.74 | 12.49 | 0.25 | 24.5 | 67.7 |
| #3 | **Top4-D6-B27** | 40.0% (40) | 21.0% (21) | 23.0% (23) | 73.0% | 13.56 | 12.66 | 0.18 | 25.1 | 186.7 |
| #4 | **Top1-D7-B36** | 38.0% (38) | 23.0% (23) | 14.0% (14) | 63.0% | 13.51 | 12.16 | 0.27 | 24.7 | 233.4 |

### Scenario: Coin Heaven Sprint (3x Coin Collector) (`coin-heaven`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top2-D3-B39** | 10.0% (10) | 8.0% (8) | 0.0% (0) | 90.0% | 9.73 | 9.08 | 0.13 | 0.0 | 72.1 |
| #2 | **Top1-D7-B36** | 9.0% (9) | 6.0% (6) | 0.0% (0) | 90.0% | 9.44 | 8.94 | 0.10 | 0.0 | 236.1 |
| #3 | **Top4-D6-B27** | 9.0% (9) | 5.0% (5) | 0.0% (0) | 87.0% | 8.99 | 8.84 | 0.03 | 0.0 | 188.6 |
| #4 | **Top3-D3-B39** | 6.0% (6) | 4.0% (4) | 0.0% (0) | 91.0% | 9.50 | 9.20 | 0.06 | 0.0 | 72.5 |

### Cross-Scenario Weighted Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Total Win % | Point Win % | Surv Win % | Surv % | Avg Score | Coins | Kills |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D7-B36** | 7 | 36 | 0.19 | 0.84 | **4636.5** | 39.8% | 24.8% | 17.0% | 71.8% | 6.95 | 5.77 | 0.24 |
| #2 | **Top2-D3-B39** | 3 | 39 | 0.09 | 0.60 | **4517.7** | 39.0% | 24.8% | 17.2% | 70.2% | 6.90 | 5.80 | 0.22 |
| #3 | **Top3-D3-B39** | 3 | 39 | 0.09 | 0.42 | **4403.4** | 35.0% | 22.6% | 16.0% | 72.2% | 6.93 | 5.91 | 0.20 |
| #4 | **Top4-D6-B27** | 6 | 27 | 0.07 | 0.74 | **4402.9** | 38.0% | 20.6% | 17.4% | 69.4% | 6.58 | 5.74 | 0.17 |

---

## 3. Phase 2: Direct Multi-Scenario Head-to-Head Tournament

### Direct Combat: Scenario `coin-heaven`

| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top3-D3-B39** | 12 (36.4%) | 9 (27.3%) | 0 (0.0%) | 78.8% | 14.27 | 0.12 | 13.67 |
| 🥈 #2 | **Top2-D3-B39** | 9 (27.3%) | 8 (24.2%) | 0 (0.0%) | 66.7% | 13.48 | 0.27 | 12.12 |
| 🥉 #3 | **Top4-D6-B27** | 9 (27.3%) | 5 (15.2%) | 0 (0.0%) | 51.5% | 11.55 | 0.12 | 10.94 |
| #4 | **Top1-D7-B36** | 8 (24.2%) | 6 (18.2%) | 0 (0.0%) | 39.4% | 11.76 | 0.12 | 11.15 |

### Direct Combat: Scenario `classic`

| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top2-D3-B39** | 20 (36.4%) | 14 (25.5%) | 3 (5.5%) | 58.2% | 3.40 | 0.20 | 2.40 |
| 🥈 #2 | **Top3-D3-B39** | 18 (32.7%) | 15 (27.3%) | 5 (9.1%) | 58.2% | 3.51 | 0.24 | 2.33 |
| 🥉 #3 | **Top1-D7-B36** | 14 (25.5%) | 11 (20.0%) | 1 (1.8%) | 38.2% | 2.55 | 0.09 | 2.09 |
| #4 | **Top4-D6-B27** | 13 (23.6%) | 9 (16.4%) | 1 (1.8%) | 43.6% | 2.42 | 0.09 | 1.96 |

### Direct Combat: Scenario `loot-crate`

| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top1-D7-B36** | 23 (36.5%) | 19 (30.2%) | 3 (4.8%) | 50.8% | 14.78 | 0.27 | 13.43 |
| 🥈 #2 | **Top4-D6-B27** | 18 (28.6%) | 17 (27.0%) | 0 (0.0%) | 50.8% | 13.06 | 0.10 | 12.59 |
| 🥉 #3 | **Top2-D3-B39** | 16 (25.4%) | 14 (22.2%) | 3 (4.8%) | 49.2% | 12.51 | 0.19 | 11.56 |
| #4 | **Top3-D3-B39** | 13 (20.6%) | 10 (15.9%) | 5 (7.9%) | 50.8% | 12.03 | 0.10 | 11.56 |

### Combined Tournament Leaderboard (151 Rounds)

| Place | Candidate | Total Wins | Point Wins | Surv Wins | Overall Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top1-D7-B36** | 45 (29.8%) | 36 (23.8%) | 4 (2.6%) | 43.7% | 9.66 | 0.17 | 8.80 |
| 🥈 #2 | **Top2-D3-B39** | 45 (29.8%) | 36 (23.8%) | 6 (4.0%) | 56.3% | 9.40 | 0.21 | 8.34 |
| 🥉 #3 | **Top3-D3-B39** | 43 (28.5%) | 34 (22.5%) | 10 (6.6%) | 59.6% | 9.42 | 0.15 | 8.66 |
| #4 | **Top4-D6-B27** | 40 (26.5%) | 31 (20.5%) | 1 (0.7%) | 48.3% | 8.85 | 0.10 | 8.36 |
