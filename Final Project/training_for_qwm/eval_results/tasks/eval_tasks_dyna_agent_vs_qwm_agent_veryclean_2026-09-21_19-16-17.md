# BombeRLe Multi-Task Evaluation Report

**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)

- **Generated:** `2026-09-21 19:16:17`
- **Evaluated Agents:** `dyna_agent`, `qwm_agent_veryclean`

## 1. Executive Task Summary

The project handout defines 4 progressive tasks that agents must solve to attain full tournament readiness:

1. **Task 1: Coin Navigation** (`coin-heaven`, solo) — Efficient pathfinding and coin collection without bombs.
2. **Task 2: Crate Clearing & Bomb Safety** (`loot-crate`, solo) — Destroying crates, discovering coins, and escaping explosions with zero self-kills.
3. **Task 3: Hunting Passive & Collector Enemies** (`classic`, vs peaceful + coin_collector) — Eliminating moving adversaries while dodging counter-attacks.
4. **Task 4: Full Competitive Combat** (`classic`, vs 3x rule_based_agent) — Full tournament combat and beating the benchmark.

## 3. Direct Head-to-Head Benchmarks (Matched Seeds)

Direct head-to-head competition between candidate models on identical matched random seeds, with alternating starting slot positions to guarantee symmetric fairness.

### 1v1 Duel (No Opponents)

- **Rounds Played:** 1 matched rounds
- **Verdict:** **`dyna_agent WINS`**
- **Score Delta:** Δ = `+7.00` (±`0.00` 95% CI, p = `1.0000`)

| Metric | `dyna_agent` | `qwm_agent_veryclean` | Advantage / Notes |
|:---|:---:|:---:|:---|
| **Wins / Win Rate** | **1 (100.0%)** | **0 (0.0%)** | Ties: 0 (0.0%) |
| **Survival Rate** | 100.0% | 0.0% | +100.0% difference |
| **Mean Score** | **10.00** | **3.00** | Δ = +7.00 |
| **Coins Collected** | 5.0 (62.5%) | 3.0 (37.5%) | Resource share |
| **Crates Destroyed** | 63.0 | 60.0 | Destructive power |
| **Total Kills** | 1.00 / rnd | 0.00 / rnd | Elimination frequency |
| **Direct Rival Kills** | **1 kills** | **0 kills** | Eliminations of rival agent |
| **Suicides** | 0 | 0 | Self-blast errors |

