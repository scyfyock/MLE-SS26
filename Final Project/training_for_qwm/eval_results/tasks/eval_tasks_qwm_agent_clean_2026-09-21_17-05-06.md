# BombeRLe Multi-Task Evaluation Report

**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)

- **Generated:** `2026-09-21 17:05:06`
- **Evaluated Agents:** `qwm_agent_clean`

## 1. Executive Task Summary

The project handout defines 4 progressive tasks that agents must solve to attain full tournament readiness:

1. **Task 1: Coin Navigation** (`coin-heaven`, solo) — Efficient pathfinding and coin collection without bombs.
2. **Task 2: Crate Clearing & Bomb Safety** (`loot-crate`, solo) — Destroying crates, discovering coins, and escaping explosions with zero self-kills.
3. **Task 3: Hunting Passive & Collector Enemies** (`classic`, vs peaceful + coin_collector) — Eliminating moving adversaries while dodging counter-attacks.
4. **Task 4: Full Competitive Combat** (`classic`, vs 3x rule_based_agent) — Full tournament combat and beating the benchmark.

### Performance Summary: `qwm_agent_clean`

| Task | Scenario | Rounds | Win % | Surv % | Score | Coins | Crates | Kills | Suicides | Grade |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Task 1: Coin Navigation** | `coin-heaven` | 2 | **100.0%** | 50.0% | 35.50 | 35.5 | 0.0 | 0.00 | 1 | **A** |
| **Task 2: Crate Clearing & Bomb Safety** | `loot-crate` | 2 | **100.0%** | 100.0% | 34.00 | 34.0 | 97.5 | 0.00 | 0 | **A+** |


#### Key Scientific KPIs: `qwm_agent_clean`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 35.5 coins / round | 7.2 steps / coin | High-speed coin navigation; 0.0% waits |
| **Task 2: Crate Clearing** | 97.5 crates destroyed | 2.12 crates / bomb | Zero Suicides (Safe); 100.0% survival |

