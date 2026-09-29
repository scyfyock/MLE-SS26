# Top-K Candidate Rescan & Tournament Report (`qwm_agent_clean`)

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Agent Architecture:** `qwm_agent_clean` (Pure Neural World Model)
- **Timestamp:** `2026-09-14 01:21:51`
- **Candidates Benchmarked:** 1
- **Total Phase 1 Matches:** 5

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
| #1 | **Top1-D7-B36** | 100.0% (1) | 100.0% (1) | 0.0% (0) | 100.0% | 5.00 | 5.00 | 0.00 | 40.0 | 38.7 |

### Scenario: DQN Rivalry (1x Spatial DQN + 2x Rule-Based) (`classic`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D7-B36** | 100.0% (1) | 100.0% (1) | 0.0% (0) | 0.0% | 8.00 | 3.00 | 1.00 | 32.0 | 46.7 |

### Scenario: Spatial Clash (2x Spatial DQN + 1x Rule-Based) (`classic`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D7-B36** | 100.0% (1) | 100.0% (1) | 0.0% (0) | 0.0% | 4.00 | 4.00 | 0.00 | 28.0 | 29.6 |

### Scenario: Loot Crate Mixed (Spatial DQN + Rule-Based + Coin Collector) (`loot-crate`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D7-B36** | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% | 9.00 | 9.00 | 0.00 | 19.0 | 35.4 |

### Scenario: Coin Heaven Sprint (3x Coin Collector) (`coin-heaven`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D7-B36** | 0.0% (0) | 0.0% (0) | 0.0% (0) | 100.0% | 10.00 | 10.00 | 0.00 | 0.0 | 17.8 |

### Cross-Scenario Weighted Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Total Win % | Point Win % | Surv Win % | Surv % | Avg Score | Coins | Kills |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D7-B36** | 7 | 36 | 0.19 | 0.84 | **5022.5** | 60.0% | 60.0% | 0.0% | 40.0% | 7.20 | 6.20 | 0.20 |

---

## 3. Phase 2: Direct Multi-Scenario Head-to-Head Tournament

### Direct Combat: Scenario `coin-heaven`

| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **my_spatial_dqn_agent** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 13.00 | 0.00 | 13.00 |
| 🥈 #2 | **Top1-D7-B36** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 8.00 | 0.00 | 8.00 |

### Direct Combat: Scenario `classic`

| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top1-D7-B36** | 1 (100.0%) | 0 (0.0%) | 1 (100.0%) | 100.0% | 5.00 | 0.00 | 5.00 |
| 🥈 #2 | **my_spatial_dqn_agent** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 1.00 | 0.00 | 1.00 |

### Direct Combat: Scenario `loot-crate`

| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **my_spatial_dqn_agent** | 1 (100.0%) | 1 (100.0%) | 0 (0.0%) | 100.0% | 18.00 | 1.00 | 13.00 |
| 🥈 #2 | **Top1-D7-B36** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 100.0% | 13.00 | 0.00 | 13.00 |

### Combined Tournament Leaderboard (3 Rounds)

| Place | Candidate | Total Wins | Point Wins | Surv Wins | Overall Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **my_spatial_dqn_agent** | 1 (33.3%) | 1 (33.3%) | 0 (0.0%) | 33.3% | 10.67 | 0.33 | 9.00 |
| 🥈 #2 | **Top1-D7-B36** | 1 (33.3%) | 0 (0.0%) | 1 (33.3%) | 66.7% | 8.67 | 0.00 | 8.67 |
