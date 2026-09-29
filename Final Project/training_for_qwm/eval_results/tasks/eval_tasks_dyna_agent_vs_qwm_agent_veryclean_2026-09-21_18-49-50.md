# BombeRLe Multi-Task Evaluation Report

**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)

- **Generated:** `2026-09-21 18:49:50`
- **Evaluated Agents:** `dyna_agent`, `qwm_agent_veryclean`

## 1. Executive Task Summary

The project handout defines 4 progressive tasks that agents must solve to attain full tournament readiness:

1. **Task 1: Coin Navigation** (`coin-heaven`, solo) — Efficient pathfinding and coin collection without bombs.
2. **Task 2: Crate Clearing & Bomb Safety** (`loot-crate`, solo) — Destroying crates, discovering coins, and escaping explosions with zero self-kills.
3. **Task 3: Hunting Passive & Collector Enemies** (`classic`, vs peaceful + coin_collector) — Eliminating moving adversaries while dodging counter-attacks.
4. **Task 4: Full Competitive Combat** (`classic`, vs 3x rule_based_agent) — Full tournament combat and beating the benchmark.

### Performance Summary: `dyna_agent`

| Task | Scenario | Rounds | Win % | Surv % | Score | Coins | Crates | Kills | Suicides | Grade |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Task 1: Coin Navigation** | `coin-heaven` | 2 | **100.0%** | 100.0% | 47.00 | 47.0 | 0.0 | 0.00 | 0 | **A+** |
| **Task 4: Full Competitive Combat (Tournament)** | `classic` | 2 | **0.0%** | 50.0% | 1.00 | 1.0 | 22.0 | 0.00 | 1 | **Needs Improvement** |


#### Key Scientific KPIs: `dyna_agent`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 47.0 coins / round | 8.5 steps / coin | High-speed coin navigation; 5.9% waits |
| **Task 4: Tournament Combat** | 0.0% win rate | Score Δ vs Rule-Based: +1.00 pts | Outperforms rule-based agent baseline |


### Performance Summary: `qwm_agent_veryclean`

| Task | Scenario | Rounds | Win % | Surv % | Score | Coins | Crates | Kills | Suicides | Grade |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Task 1: Coin Navigation** | `coin-heaven` | 2 | **100.0%** | 100.0% | 48.00 | 48.0 | 0.0 | 0.00 | 0 | **A+** |
| **Task 4: Full Competitive Combat (Tournament)** | `classic` | 2 | **50.0%** | 100.0% | 5.50 | 3.0 | 25.5 | 0.50 | 0 | **A+** |


#### Key Scientific KPIs: `qwm_agent_veryclean`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 48.0 coins / round | 7.2 steps / coin | High-speed coin navigation; 0.0% waits |
| **Task 4: Tournament Combat** | 50.0% win rate | Score Δ vs Rule-Based: +5.50 pts | Outperforms rule-based agent baseline |


## 2. Cross-Model Comparative Table (Section 6 Report Ready)

| Model / Architecture | Task 1 (Coins) | Task 2 (Crates / Suicides) | Task 3 (Kills / Win%) | Task 4 (Win% vs Rule-Based) | Overall Rating |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`dyna_agent`** | 47.0 (A+) | - | - | 0.0% (1.0 pts) | **STRONG** |
| **`qwm_agent_veryclean`** | 48.0 (A+) | - | - | 50.0% (5.5 pts) | **CHAMPION** |


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
- **Score Delta:** Δ = `-0.50` (±`4.90` 95% CI, p = `0.8743`)

| Metric | `dyna_agent` | `qwm_agent_veryclean` | Advantage / Notes |
|:---|:---:|:---:|:---|
| **Wins / Win Rate** | **1 (50.0%)** | **1 (50.0%)** | Ties: 0 (0.0%) |
| **Survival Rate** | 0.0% | 50.0% | -50.0% difference |
| **Mean Score** | **1.50** | **2.00** | Δ = -0.50 |
| **Coins Collected** | 1.5 (42.9%) | 2.0 (57.1%) | Resource share |
| **Crates Destroyed** | 16.0 | 26.0 | Destructive power |
| **Total Kills** | 0.00 / rnd | 0.00 / rnd | Elimination frequency |
| **Direct Rival Kills** | **0 kills** | **0 kills** | Eliminations of rival agent |
| **Suicides** | 2 | 1 | Self-blast errors |

