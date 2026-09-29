# BombeRLe Multi-Task Evaluation Report

**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)

- **Generated:** `2026-09-21 19:09:37`
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
| **Task 1: Coin Navigation** | `coin-heaven` | 100 | **100.0%** | 100.0% | 43.71 | 43.7 | 0.0 | 0.00 | 0 | **A+** |
| **Task 2: Crate Clearing & Bomb Safety** | `loot-crate` | 100 | **100.0%** | 68.0% | 32.54 | 32.5 | 87.6 | 0.00 | 32 | **A+** |
| **Task 3: Hunting Passive & Collector Enemies** | `classic` | 100 | **60.0%** | 71.0% | 7.01 | 4.4 | 53.8 | 0.53 | 23 | **B** |
| **Task 4: Full Competitive Combat (Tournament)** | `classic` | 100 | **43.0%** | 48.0% | 3.77 | 2.4 | 31.1 | 0.27 | 43 | **A** |


#### Key Scientific KPIs: `dyna_agent`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 43.7 coins / round | 9.0 steps / coin | High-speed coin navigation; 11.1% waits |
| **Task 2: Crate Clearing** | 87.6 crates destroyed | 2.25 crates / bomb | 32 suicides; 68.0% survival |
| **Task 3: Hunting Enemies** | 0.53 kills / round | 60.0% win rate | Neutralizes moving targets while evading blasts |
| **Task 4: Tournament Combat** | 43.0% win rate | Score Δ vs Rule-Based: +3.77 pts | Outperforms rule-based agent baseline |


### Performance Summary: `qwm_agent_veryclean`

| Task | Scenario | Rounds | Win % | Surv % | Score | Coins | Crates | Kills | Suicides | Grade |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Task 1: Coin Navigation** | `coin-heaven` | 100 | **100.0%** | 100.0% | 45.01 | 45.0 | 0.0 | 0.00 | 0 | **A+** |
| **Task 2: Crate Clearing & Bomb Safety** | `loot-crate` | 100 | **100.0%** | 85.0% | 32.57 | 32.6 | 85.2 | 0.00 | 15 | **A+** |
| **Task 3: Hunting Passive & Collector Enemies** | `classic` | 100 | **70.0%** | 84.0% | 7.53 | 4.6 | 53.2 | 0.58 | 12 | **B** |
| **Task 4: Full Competitive Combat (Tournament)** | `classic` | 100 | **48.0%** | 67.0% | 4.26 | 2.7 | 28.1 | 0.31 | 22 | **A** |


#### Key Scientific KPIs: `qwm_agent_veryclean`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 45.0 coins / round | 6.8 steps / coin | High-speed coin navigation; 0.5% waits |
| **Task 2: Crate Clearing** | 85.2 crates destroyed | 2.11 crates / bomb | 15 suicides; 85.0% survival |
| **Task 3: Hunting Enemies** | 0.58 kills / round | 70.0% win rate | Neutralizes moving targets while evading blasts |
| **Task 4: Tournament Combat** | 48.0% win rate | Score Δ vs Rule-Based: +4.26 pts | Outperforms rule-based agent baseline |


## 2. Cross-Model Comparative Table (Section 6 Report Ready)

| Model / Architecture | Task 1 (Coins) | Task 2 (Crates / Suicides) | Task 3 (Kills / Win%) | Task 4 (Win% vs Rule-Based) | Overall Rating |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`dyna_agent`** | 43.7 (A+) | 87.6 cr (32 suic) | 0.53 k (60%) | 43.0% (3.8 pts) | **CHAMPION** |
| **`qwm_agent_veryclean`** | 45.0 (A+) | 85.2 cr (15 suic) | 0.58 k (70%) | 48.0% (4.3 pts) | **CHAMPION** |


## 3. Direct Head-to-Head Benchmarks (Matched Seeds)

Direct head-to-head competition between candidate models on identical matched random seeds, with alternating starting slot positions to guarantee symmetric fairness.

### 1v1 Duel (No Opponents)

- **Rounds Played:** 100 matched rounds
- **Verdict:** **`qwm_agent_veryclean WINS`**
- **Score Delta:** Δ = `-0.42` (±`0.88` 95% CI, p = `0.3514`)

| Metric | `dyna_agent` | `qwm_agent_veryclean` | Advantage / Notes |
|:---|:---:|:---:|:---|
| **Wins / Win Rate** | **43 (43.0%)** | **57 (57.0%)** | Ties: 0 (0.0%) |
| **Survival Rate** | 59.0% | 80.0% | -21.0% difference |
| **Mean Score** | **4.45** | **4.87** | Δ = -0.42 |
| **Coins Collected** | 4.0 (46.9%) | 4.5 (53.1%) | Resource share |
| **Crates Destroyed** | 56.5 | 61.1 | Destructive power |
| **Total Kills** | 0.10 / rnd | 0.08 / rnd | Elimination frequency |
| **Direct Rival Kills** | **10 kills** | **8 kills** | Eliminations of rival agent |
| **Suicides** | 35 | 10 | Self-blast errors |


### 4-Player Tournament (1v1 + 2 Rule-Based Opponents)

- **Rounds Played:** 100 matched rounds
- **Verdict:** **`qwm_agent_veryclean WINS`**
- **Score Delta:** Δ = `+0.05` (±`0.80` 95% CI, p = `0.9033`)

| Metric | `dyna_agent` | `qwm_agent_veryclean` | Advantage / Notes |
|:---|:---:|:---:|:---|
| **Wins / Win Rate** | **38 (38.0%)** | **58 (58.0%)** | Ties: 4 (4.0%) |
| **Survival Rate** | 50.0% | 59.0% | -9.0% difference |
| **Mean Score** | **3.46** | **3.41** | Δ = +0.05 |
| **Coins Collected** | 2.4 (51.1%) | 2.3 (48.9%) | Resource share |
| **Crates Destroyed** | 30.9 | 28.0 | Destructive power |
| **Total Kills** | 0.22 / rnd | 0.23 / rnd | Elimination frequency |
| **Direct Rival Kills** | **4 kills** | **3 kills** | Eliminations of rival agent |
| **Suicides** | 45 | 31 | Self-blast errors |

