# BombeRLe Multi-Task Evaluation Report

**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)

- **Generated:** `2026-09-21 17:15:46`
- **Evaluated Agents:** `qwm_agent_veryclean`

## 1. Executive Task Summary

The project handout defines 4 progressive tasks that agents must solve to attain full tournament readiness:

1. **Task 1: Coin Navigation** (`coin-heaven`, solo) — Efficient pathfinding and coin collection without bombs.
2. **Task 2: Crate Clearing & Bomb Safety** (`loot-crate`, solo) — Destroying crates, discovering coins, and escaping explosions with zero self-kills.
3. **Task 3: Hunting Passive & Collector Enemies** (`classic`, vs peaceful + coin_collector) — Eliminating moving adversaries while dodging counter-attacks.
4. **Task 4: Full Competitive Combat** (`classic`, vs 3x rule_based_agent) — Full tournament combat and beating the benchmark.

### Performance Summary: `qwm_agent_veryclean`

| Task | Scenario | Rounds | Win % | Surv % | Score | Coins | Crates | Kills | Suicides | Grade |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Task 1: Coin Navigation** | `coin-heaven` | 20 | **100.0%** | 50.0% | 32.40 | 32.4 | 0.0 | 0.00 | 10 | **B** |
| **Task 2: Crate Clearing & Bomb Safety** | `loot-crate` | 20 | **100.0%** | 95.0% | 34.25 | 34.2 | 86.7 | 0.00 | 1 | **A+** |
| **Task 3: Hunting Passive & Collector Enemies** | `classic` | 20 | **95.0%** | 90.0% | 8.40 | 4.7 | 53.0 | 0.75 | 2 | **A** |
| **Task 4: Full Competitive Combat (Tournament)** | `classic` | 20 | **40.0%** | 65.0% | 3.55 | 3.0 | 26.8 | 0.10 | 3 | **A** |


#### Key Scientific KPIs: `qwm_agent_veryclean`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 32.4 coins / round | 5.5 steps / coin | High-speed coin navigation; 1.0% waits |
| **Task 2: Crate Clearing** | 86.7 crates destroyed | 2.04 crates / bomb | 1 suicides; 95.0% survival |
| **Task 3: Hunting Enemies** | 0.75 kills / round | 95.0% win rate | Neutralizes moving targets while evading blasts |
| **Task 4: Tournament Combat** | 40.0% win rate | Score Δ vs Rule-Based: +3.55 pts | Outperforms rule-based agent baseline |

