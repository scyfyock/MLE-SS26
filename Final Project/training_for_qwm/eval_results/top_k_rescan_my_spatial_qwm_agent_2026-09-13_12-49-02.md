# Top-K Candidate Rescan & Tournament Report (`my_spatial_qwm_agent`)

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Agent Architecture:** `my_spatial_qwm_agent`
- **Timestamp:** `2026-09-13 12:49:02`
- **Candidates Benchmarked:** 2
- **Total Phase 1 Matches:** 4

## 1. Executive Champion Recommendation

**Undisputed Winner:** `Top1-D5-B18` (Phase 2 Direct Combat Tournament)

| Hyperparameter | Optimal Value | Parameter Purpose |
|:---|---:|:---|
| `search_depth` | **5** | Lookahead horizon steps |
| `beam_size` | **18** | Beam search pruning width |
| `tree_discount` ($\lambda$) | **0.080** | Lookahead prospective reward discount |
| `alpha_vq` ($\alpha$) | **0.160** | Balance between Q-critic and prospective return |
| `predicted_wait_penalty` | **1.25** | Tactical heuristic wait discouragement |
| `predicted_loop_penalty` | **1.50** | Heuristic loop avoidance penalty |

---

## 2. Phase 1: Explicit Per-Scenario Performance Breakdowns

> [!NOTE]
> Wins are explicitly separated into **Point Wins** (highest score) and **Survival Wins** (last survivor).

### Scenario: Classic vs 3x Rule-Based (`classic`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D5-B18** | 100.0% (1) | 100.0% (1) | 0.0% (0) | 100.0% | 4.00 | 4.00 | 0.00 | 25.0 | 29.7 |
| #2 | **Top2-D7-B12** | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% | 2.00 | 2.00 | 0.00 | 13.0 | 55.2 |

### Scenario: Loot Crate Mixed (Spatial DQN + Rule-Based + Coin Collector) (`loot-crate`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D5-B18** | 100.0% (1) | 100.0% (1) | 0.0% (0) | 0.0% | 20.00 | 10.00 | 2.00 | 14.0 | 30.3 |
| #2 | **Top2-D7-B12** | 0.0% (0) | 0.0% (0) | 0.0% (0) | 100.0% | 15.00 | 15.00 | 0.00 | 37.0 | 24.2 |

### Cross-Scenario Weighted Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Total Win % | Point Win % | Surv Win % | Surv % | Avg Score | Coins | Kills |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D5-B18** | 5 | 18 | 0.08 | 0.16 | **6327.2** | 100.0% | 100.0% | 0.0% | 60.0% | 10.40 | 6.40 | 0.80 |
| #2 | **Top2-D7-B12** | 7 | 12 | 0.05 | 0.52 | **893.6** | 0.0% | 0.0% | 0.0% | 40.0% | 7.20 | 7.20 | 0.00 |

---

## 3. Phase 2: Direct Multi-Scenario Head-to-Head Tournament

### Direct Combat: Scenario `classic`

| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top1-D5-B18** | 1 (100.0%) | 1 (100.0%) | 0 (0.0%) | 100.0% | 4.00 | 0.00 | 4.00 |
| 🥈 #2 | **Top2-D7-B12** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 100.0% | 2.00 | 0.00 | 2.00 |
| 🥉 #3 | **my_spatial_dqn_agent** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 2.00 | 0.00 | 2.00 |

### Direct Combat: Scenario `loot-crate`

| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top1-D5-B18** | 1 (100.0%) | 1 (100.0%) | 1 (100.0%) | 100.0% | 27.00 | 0.00 | 27.00 |
| 🥈 #2 | **my_spatial_dqn_agent** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 5.00 | 0.00 | 5.00 |
| 🥉 #3 | **Top2-D7-B12** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 3.00 | 0.00 | 3.00 |

### Combined Tournament Leaderboard (2 Rounds)

| Place | Candidate | Total Wins | Point Wins | Surv Wins | Overall Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top1-D5-B18** | 2 (100.0%) | 2 (100.0%) | 1 (50.0%) | 100.0% | 15.50 | 0.00 | 15.50 |
| 🥈 #2 | **my_spatial_dqn_agent** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 3.50 | 0.00 | 3.50 |
| 🥉 #3 | **Top2-D7-B12** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 50.0% | 2.50 | 0.00 | 2.50 |
