# BombeRLe Evaluation Report: `my_spatial_dqn_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-06_18-24-53_spatial_dqn (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 01:07:23

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **40.0%** | 40.0% | `+0.0%` |
| **Survival Rate** | **0.0%** | 50.0% | `-50.0%` |
| **Score / Round** | **3.00 ± 1.9** | 2.67 | `+0.33` |
| **Coins / Round** | **2.50 ± 1.7** | 2.17 | `+0.33` |
| **Coin Share** | **27.8%** | 24.1% | `+3.7%` |
| **Kills / Round** | **0.10 ± 0.3** | 0.10 | `-0.00` |
| **Kill / Death Ratio (KDR)** | **0.10** | 0.20 | `-0.10` |
| **Steps Survived** | **152.6 ± 77** | 227.6 | `-75.0` |
| **Suicides / Round** | **1.00** | 0.47 | `+0.53` |
| **Wait % of Actions** | **7.5%** | 0.5% | `+7.0%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_dqn_agent** | **40.0%** | 0.0% | 3.00 | 2.50 | 0.10 | 10 | 0 | 152.6 | 7.5% |
| rule_based_agent_0 | 40.0% | 40.0% | 2.70 | 2.20 | 0.10 | 6 | 2 | 203.5 | 0.6% |
| rule_based_agent_1 | 50.0% | 70.0% | 3.00 | 2.50 | 0.10 | 3 | 0 | 292.1 | 0.4% |
| rule_based_agent_2 | 30.0% | 40.0% | 2.30 | 1.80 | 0.10 | 5 | 2 | 187.1 | 0.4% |
| *Rule-Based Avg* | *40.0%* | *50.0%* | *2.67* | *2.17* | *0.10* | *4.7* | *1.3* | *227.6* | *0.5%* |

## Action Profile for `my_spatial_dqn_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1219 | 121.9 | 79.9% |
| Bombs Placed | 180 | 18.0 | 11.8% |
| Waited (WAIT) | 115 | 11.5 | 7.5% |
| Invalid Actions | 12 | 1.2 | 0.8% |
| Crates Destroyed | 315 | 31.5 | - |
| **Total Actions** | **1526** | **152.6** | **100.0%** |
