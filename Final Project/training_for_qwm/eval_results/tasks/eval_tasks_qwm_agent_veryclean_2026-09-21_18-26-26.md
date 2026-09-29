# BombeRLe Multi-Task Evaluation Report

**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)

- **Generated:** `2026-09-21 18:26:26`
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
| **Task 1: Coin Navigation** | `coin-heaven` | 100 | **100.0%** | 32.0% | 28.00 | 28.0 | 0.0 | 0.00 | 68 | **B** |
| **Task 2: Crate Clearing & Bomb Safety** | `loot-crate` | 100 | **100.0%** | 85.0% | 32.63 | 32.6 | 85.2 | 0.00 | 15 | **A+** |
| **Task 3: Hunting Passive & Collector Enemies** | `classic` | 100 | **68.0%** | 91.0% | 7.12 | 4.6 | 52.7 | 0.51 | 6 | **B** |
| **Task 4: Full Competitive Combat (Tournament)** | `classic` | 100 | **41.0%** | 65.0% | 3.79 | 2.5 | 26.9 | 0.25 | 20 | **A** |


#### Key Scientific KPIs: `qwm_agent_veryclean`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 28.0 coins / round | 5.6 steps / coin | High-speed coin navigation; 0.8% waits |
| **Task 2: Crate Clearing** | 85.2 crates destroyed | 2.11 crates / bomb | 15 suicides; 85.0% survival |
| **Task 3: Hunting Enemies** | 0.51 kills / round | 68.0% win rate | Neutralizes moving targets while evading blasts |
| **Task 4: Tournament Combat** | 41.0% win rate | Score Δ vs Rule-Based: +3.79 pts | Outperforms rule-based agent baseline |

