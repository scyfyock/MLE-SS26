# BombeRLe Multi-Task Evaluation Report

**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)

- **Generated:** `2026-09-21 18:48:43`
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

- **Rounds Played:** 2 matched rounds
- **Verdict:** **`dyna_agent WINS`**
- **Score Delta:** Δ = `+7.00` (±`0.00` 95% CI, p = `1.0000`)

| Metric | `dyna_agent` | `qwm_agent_veryclean` | Advantage / Notes |
|:---|:---:|:---:|:---|
| **Wins / Win Rate** | **2 (100.0%)** | **0 (0.0%)** | Ties: 0 (0.0%) |
| **Survival Rate** | 50.0% | 0.0% | +50.0% difference |
| **Mean Score** | **10.00** | **3.00** | Δ = +7.00 |
| **Coins Collected** | 5.0 (62.5%) | 3.0 (37.5%) | Resource share |
| **Crates Destroyed** | 53.5 | 59.5 | Destructive power |
| **Total Kills** | 1.00 / rnd | 0.00 / rnd | Elimination frequency |
| **Direct Rival Kills** | **2 kills** | **0 kills** | Eliminations of rival agent |
| **Suicides** | 1 | 0 | Self-blast errors |


### 4-Player Tournament (1v1 + 2 Rule-Based Opponents)

- **Rounds Played:** 2 matched rounds
- **Verdict:** **`DEAD EVEN TIE`**
- **Score Delta:** Δ = `-3.50` (±`8.82` 95% CI, p = `0.5792`)

| Metric | `dyna_agent` | `qwm_agent_veryclean` | Advantage / Notes |
|:---|:---:|:---:|:---|
| **Wins / Win Rate** | **1 (50.0%)** | **1 (50.0%)** | Ties: 0 (0.0%) |
| **Survival Rate** | 0.0% | 50.0% | -50.0% difference |
| **Mean Score** | **1.50** | **5.00** | Δ = -3.50 |
| **Coins Collected** | 1.5 (37.5%) | 2.5 (62.5%) | Resource share |
| **Crates Destroyed** | 15.5 | 30.5 | Destructive power |
| **Total Kills** | 0.00 / rnd | 0.50 / rnd | Elimination frequency |
| **Direct Rival Kills** | **0 kills** | **0 kills** | Eliminations of rival agent |
| **Suicides** | 2 | 1 | Self-blast errors |

