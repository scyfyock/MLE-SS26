# BombeRLe Multi-Task Evaluation Report

**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)

- **Generated:** `2026-09-21 16:48:42`
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
| **Task 1: Coin Navigation** | `coin-heaven` | 1 | **100.0%** | 100.0% | 50.00 | 50.0 | 0.0 | 0.00 | 0 | **A+** |
| **Task 2: Crate Clearing & Bomb Safety** | `loot-crate` | 1 | **100.0%** | 100.0% | 35.00 | 35.0 | 108.0 | 0.00 | 0 | **A+** |
| **Task 3: Hunting Passive & Collector Enemies** | `classic` | 1 | **100.0%** | 100.0% | 11.00 | 6.0 | 74.0 | 1.00 | 0 | **A+** |
| **Task 4: Full Competitive Combat (Tournament)** | `classic` | 1 | **100.0%** | 100.0% | 4.00 | 4.0 | 29.0 | 0.00 | 0 | **A+** |


#### Key Scientific KPIs: `qwm_agent_clean`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 50.0 coins / round | 6.6 steps / coin | High-speed coin navigation; 0.3% waits |
| **Task 2: Crate Clearing** | 108.0 crates destroyed | 2.30 crates / bomb | Zero Suicides (Safe); 100.0% survival |
| **Task 3: Hunting Enemies** | 1.00 kills / round | 100.0% win rate | Neutralizes moving targets while evading blasts |
| **Task 4: Tournament Combat** | 100.0% win rate | Score Δ vs Rule-Based: +4.00 pts | Outperforms rule-based agent baseline |

