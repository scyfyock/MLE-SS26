# Top-K Candidate Rescan & Tournament Report (`qwm_agent`)

- **Evaluated Checkpoint:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Agent Architecture:** `qwm_agent` (Pure Neural World Model)
- **Timestamp:** `2026-09-13 13:14:42`
- **Candidates Benchmarked:** 1
- **Total Phase 1 Matches:** 2

## 1. Executive Champion Recommendation

**Undisputed Winner:** `Top1-D7-B36` (Phase 1 Cross-Scenario Matched Benchmark)

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
| #1 | **Top1-D7-B36** | 100.0% (2) | 100.0% (2) | 50.0% (1) | 100.0% | 9.50 | 4.50 | 1.00 | 30.0 | 43.7 |

### Cross-Scenario Weighted Leaderboard

| Rank | Candidate | Depth | Beam | Disc ($\lambda$) | Alpha ($\alpha$) | Comp Score | Total Win % | Point Win % | Surv Win % | Surv % | Avg Score | Coins | Kills |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #1 | **Top1-D7-B36** | 7 | 36 | 0.19 | 0.84 | **8225.0** | 100.0% | 100.0% | 50.0% | 100.0% | 9.50 | 4.50 | 1.00 |

---
