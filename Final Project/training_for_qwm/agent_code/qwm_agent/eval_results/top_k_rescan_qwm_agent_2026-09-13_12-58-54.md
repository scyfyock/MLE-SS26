# Top-K Candidate Rescan & Tournament Report (`qwm_agent`)

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Agent Architecture:** `qwm_agent` (Pure Neural World Model)
- **Timestamp:** `2026-09-13 12:58:54`
- **Candidates Benchmarked:** 2
- **Total Phase 1 Matches:** 4

## 1. Executive Champion Recommendation

**Undisputed Winner:** `Top2-D4-B21` (Phase 2 Direct Combat Tournament)

| Hyperparameter | Optimal Value | Parameter Purpose |
|:---|---:|:---|
| `search_depth` | **4** | Lookahead horizon steps |
| `beam_size` | **21** | Beam search pruning width |
| `tree_discount` ($\lambda$) | **0.170** | Lookahead prospective reward discount |
| `alpha_vq` ($\alpha$) | **0.440** | Balance between Q-critic and prospective return |

---

## 2. Phase 1: Explicit Per-Scenario Performance Breakdowns

> [!NOTE]
> Wins are explicitly separated into **Point Wins** (highest score) and **Survival Wins** (sole survivor).

### Scenario: Classic vs 3x Rule-Based (`classic`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D3-B42** | 100.0% (1) | 100.0% (1) | 0.0% (0) | 100.0% | 10.00 | 5.00 | 1.00 | 25.0 | 12.4 |
| #2 | **Top2-D4-B21** | 100.0% (1) | 100.0% (1) | 0.0% (0) | 0.0% | 8.00 | 3.00 | 1.00 | 21.0 | 18.4 |

### Scenario: Loot Crate Mixed (Spatial DQN + Rule-Based + Coin Collector) (`loot-crate`)

| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top2-D4-B21** | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% | 19.00 | 14.00 | 1.00 | 26.0 | 23.7 |
| #2 | **Top1-D3-B42** | 0.0% (0) | 0.0% (0) | 0.0% (0) | 0.0% | 11.00 | 11.00 | 0.00 | 24.0 | 21.0 |

### Cross-Scenario Weighted Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Total Win % | Point Win % | Surv Win % | Surv % | Avg Score | Coins | Kills |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D3-B42** | 3 | 42 | 0.08 | 0.10 | **5706.0** | 50.0% | 50.0% | 0.0% | 50.0% | 10.50 | 8.00 | 0.50 |
| #2 | **Top2-D4-B21** | 4 | 21 | 0.17 | 0.44 | **4458.0** | 50.0% | 50.0% | 0.0% | 0.0% | 13.50 | 8.50 | 1.00 |

---

## 3. Phase 2: Direct Multi-Scenario Head-to-Head Tournament

### Direct Combat: Scenario `classic`

| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top1-D3-B42** | 1 (100.0%) | 1 (100.0%) | 0 (0.0%) | 0.0% | 3.00 | 0.00 | 3.00 |
| 🥈 #2 | **Top2-D4-B21** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 100.0% | 2.00 | 0.00 | 2.00 |
| 🥉 #3 | **my_spatial_dqn_agent** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 2.00 | 0.00 | 2.00 |

### Direct Combat: Scenario `loot-crate`

| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top2-D4-B21** | 1 (100.0%) | 1 (100.0%) | 0 (0.0%) | 100.0% | 19.00 | 1.00 | 14.00 |
| 🥈 #2 | **Top1-D3-B42** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 16.00 | 0.00 | 16.00 |
| 🥉 #3 | **my_spatial_dqn_agent** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 2.00 | 0.00 | 2.00 |

### Combined Tournament Leaderboard (2 Rounds)

| Place | Candidate | Total Wins | Point Wins | Surv Wins | Overall Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 #1 | **Top2-D4-B21** | 1 (50.0%) | 1 (50.0%) | 0 (0.0%) | 100.0% | 10.50 | 0.50 | 8.00 |
| 🥈 #2 | **Top1-D3-B42** | 1 (50.0%) | 1 (50.0%) | 0 (0.0%) | 0.0% | 9.50 | 0.00 | 9.50 |
| 🥉 #3 | **my_spatial_dqn_agent** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0.0% | 2.00 | 0.00 | 2.00 |
