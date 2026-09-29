# BombeRLe Multi-Task Evaluation Report

**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)

- **Generated:** `2026-09-23 20:22:52`
- **Evaluated Agents:** `dyna_agent`

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
| **Task 3: Hunting Passive & Collector Enemies** | `classic` | 1000 | **57.8%** | 72.0% | 7.10 | 4.3 | 54.4 | 0.55 | 263 | **B** |
| **Task 4: Full Competitive Combat (Tournament)** | `classic` | 1000 | **39.9%** | 49.4% | 3.77 | 2.5 | 30.8 | 0.26 | 409 | **B** |


#### Key Scientific KPIs: `dyna_agent`

| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |
|:---|:---|:---|:---|
| **Task 1: Coin Navigation** | 45.0 coins / round | 8.7 steps / coin | High-speed coin navigation; 10.7% waits |
| **Task 2: Crate Clearing** | 80.0 crates destroyed | 2.27 crates / bomb | 417 suicides; 58.3% survival |
| **Task 3: Hunting Enemies** | 0.55 kills / round | 57.8% win rate | Neutralizes moving targets while evading blasts |
| **Task 4: Tournament Combat** | 39.9% win rate | Score Δ vs Rule-Based: +3.77 pts | Outperforms rule-based agent baseline |

