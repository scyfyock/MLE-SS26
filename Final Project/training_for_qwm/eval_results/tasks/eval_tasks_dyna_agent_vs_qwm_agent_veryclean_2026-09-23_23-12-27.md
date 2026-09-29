# BombeRLe Multi-Task Evaluation Report

**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)

- **Generated:** `2026-09-23 23:12:27`
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
| **Task 1: Coin Navigation** | `coin-heaven` | 1000 | **100.0%** | 100.0% | 44.98 | 45.0 | 0.0 | 0.00 | 0 | **A+** |
| **Task 2: Crate Clearing & Bomb Safety** | `loot-crate` | 1000 | **100.0%** | 58.3% | 30.09 | 30.1 | 80.0 | 0.00 | 417 | **A+** |
| **Task 3: Hunting Passive & Collector Enemies** | `classic` | 1000 | **58.4%** | 74.0% | 7.15 | 4.4 | 55.4 | 0.56 | 234 | **B** |
| **Task 4: Full Competitive Combat (Tournament)** | `classic` | 1000 | **38.2%** | 49.8% | 3.71 | 2.5 | 31.0 | 0.25 | 408 | **B** |


#### Key Scientific KPIs: `dyna_agent`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 45.0 coins / round | 8.7 steps / coin | High-speed coin navigation; 10.7% waits |
| **Task 2: Crate Clearing** | 80.0 crates destroyed | 2.27 crates / bomb | 417 suicides; 58.3% survival |
| **Task 3: Hunting Enemies** | 0.56 kills / round | 58.4% win rate | Neutralizes moving targets while evading blasts |
| **Task 4: Tournament Combat** | 38.2% win rate | Score Δ vs Rule-Based: +3.71 pts | Outperforms rule-based agent baseline |


### Performance Summary: `qwm_agent_veryclean`

| Task | Scenario | Rounds | Win % | Surv % | Score | Coins | Crates | Kills | Suicides | Grade |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Task 1: Coin Navigation** | `coin-heaven` | 1000 | **100.0%** | 100.0% | 45.29 | 45.3 | 0.0 | 0.00 | 0 | **A+** |
| **Task 2: Crate Clearing & Bomb Safety** | `loot-crate` | 1000 | **99.8%** | 83.3% | 31.62 | 31.6 | 81.7 | 0.00 | 167 | **A+** |
| **Task 3: Hunting Passive & Collector Enemies** | `classic` | 1000 | **67.6%** | 89.4% | 7.34 | 4.4 | 52.2 | 0.59 | 78 | **B** |
| **Task 4: Full Competitive Combat (Tournament)** | `classic` | 1000 | **41.0%** | 63.4% | 3.87 | 2.5 | 28.1 | 0.27 | 242 | **A** |


#### Key Scientific KPIs: `qwm_agent_veryclean`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 45.3 coins / round | 6.6 steps / coin | High-speed coin navigation; 0.4% waits |
| **Task 2: Crate Clearing** | 81.7 crates destroyed | 2.08 crates / bomb | 167 suicides; 83.3% survival |
| **Task 3: Hunting Enemies** | 0.59 kills / round | 67.6% win rate | Neutralizes moving targets while evading blasts |
| **Task 4: Tournament Combat** | 41.0% win rate | Score Δ vs Rule-Based: +3.87 pts | Outperforms rule-based agent baseline |


## 2. Cross-Model Comparative Table (Section 6 Report Ready)

| Model / Architecture | Task 1 (Coins) | Task 2 (Crates / Suicides) | Task 3 (Kills / Win%) | Task 4 (Win% vs Rule-Based) | Overall Rating |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`dyna_agent`** | 45.0 (A+) | 80.0 cr (417 suic) | 0.56 k (58%) | 38.2% (3.7 pts) | **CHAMPION** |
| **`qwm_agent_veryclean`** | 45.3 (A+) | 81.7 cr (167 suic) | 0.59 k (68%) | 41.0% (3.9 pts) | **CHAMPION** |


## 3. Direct Head-to-Head Benchmarks (Matched Seeds)

Direct head-to-head competition between candidate models on identical matched random seeds, with alternating starting slot positions to guarantee symmetric fairness.

### 1v1 Duel (No Opponents)

- **Rounds Played:** 1000 matched rounds
- **Verdict:** **`qwm_agent_veryclean SLIGHT EDGE`**
- **Score Delta:** Δ = `-0.25` (±`0.29` 95% CI, p = `0.0872`)

| Metric | `dyna_agent` | `qwm_agent_veryclean` | Advantage / Notes |
|:---|:---:|:---:|:---|
| **Wins / Win Rate** | **465 (46.5%)** | **526 (52.6%)** | Ties: 9 (0.9%) |
| **Survival Rate** | 61.6% | 79.6% | -18.0% difference |
| **Mean Score** | **4.51** | **4.76** | Δ = -0.25 |
| **Coins Collected** | 4.1 (48.5%) | 4.3 (51.5%) | Resource share |
| **Crates Destroyed** | 57.5 | 58.8 | Destructive power |
| **Total Kills** | 0.08 / rnd | 0.09 / rnd | Elimination frequency |
| **Direct Rival Kills** | **84 kills** | **85 kills** | Eliminations of rival agent |
| **Suicides** | 309 | 125 | Self-blast errors |


### 4-Player Tournament (1v1 + 2 Rule-Based Opponents)

- **Rounds Played:** 1000 matched rounds
- **Verdict:** **`qwm_agent_veryclean SLIGHT EDGE`**
- **Score Delta:** Δ = `-0.01` (±`0.27` 95% CI, p = `0.9542`)

| Metric | `dyna_agent` | `qwm_agent_veryclean` | Advantage / Notes |
|:---|:---:|:---:|:---|
| **Wins / Win Rate** | **451 (45.1%)** | **499 (49.9%)** | Ties: 50 (5.0%) |
| **Survival Rate** | 48.3% | 63.2% | -14.9% difference |
| **Mean Score** | **3.63** | **3.64** | Δ = -0.01 |
| **Coins Collected** | 2.4 (49.2%) | 2.4 (50.8%) | Resource share |
| **Crates Destroyed** | 31.9 | 27.3 | Destructive power |
| **Total Kills** | 0.25 / rnd | 0.24 / rnd | Elimination frequency |
| **Direct Rival Kills** | **70 kills** | **56 kills** | Eliminations of rival agent |
| **Suicides** | 425 | 234 | Self-blast errors |

