# BombeRLe Evaluation Report: `my_spatial_dqn_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-06_18-24-53_spatial_dqn (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 10:18:26

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **46.0%** | 37.3% | `+8.7%` |
| **Survival Rate** | **4.0%** | 41.3% | `-37.3%` |
| **Score / Round** | **4.46 ± 3.4** | 2.81 | `+1.65` |
| **Coins / Round** | **2.66 ± 1.6** | 2.11 | `+0.55` |
| **Coin Share** | **29.6%** | 23.5% | `+6.1%` |
| **Kills / Round** | **0.36 ± 0.6** | 0.14 | `+0.22` |
| **Kill / Death Ratio (KDR)** | **0.36** | 0.21 | `+0.15` |
| **Steps Survived** | **136.2 ± 64** | 212.0 | `-75.8` |
| **Suicides / Round** | **0.84** | 0.46 | `+0.38` |
| **Wait % of Actions** | **7.8%** | 0.4% | `+7.3%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_dqn_agent** | **46.0%** | 4.0% | 4.46 | 2.66 | 0.36 | 42 | 8 | 136.2 | 7.8% |
| rule_based_agent_0 | 42.0% | 52.0% | 2.56 | 1.96 | 0.12 | 17 | 11 | 230.1 | 0.6% |
| rule_based_agent_1 | 34.0% | 40.0% | 2.68 | 2.28 | 0.08 | 24 | 12 | 216.4 | 0.3% |
| rule_based_agent_2 | 36.0% | 32.0% | 3.20 | 2.10 | 0.22 | 28 | 8 | 189.6 | 0.5% |
| *Rule-Based Avg* | *37.3%* | *41.3%* | *2.81* | *2.11* | *0.14* | *23.0* | *10.3* | *212.0* | *0.4%* |

## Action Profile for `my_spatial_dqn_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 5409 | 108.2 | 79.4% |
| Bombs Placed | 785 | 15.7 | 11.5% |
| Waited (WAIT) | 529 | 10.6 | 7.8% |
| Invalid Actions | 89 | 1.8 | 1.3% |
| Crates Destroyed | 1542 | 30.8 | - |
| **Total Actions** | **6812** | **136.2** | **100.0%** |
