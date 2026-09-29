# BombeRLe Evaluation Report: `my_spatial_dqn_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-06_18-24-53_spatial_dqn (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-06 19:40:29

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **50.0%** | 33.3% | `+16.7%` |
| **Survival Rate** | **0.0%** | 50.0% | `-50.0%` |
| **Score / Round** | **3.70 ± 2.2** | 2.43 | `+1.27` |
| **Coins / Round** | **3.20 ± 1.7** | 1.93 | `+1.27` |
| **Coin Share** | **35.6%** | 21.5% | `+14.1%` |
| **Kills / Round** | **0.10 ± 0.3** | 0.10 | `+0.00` |
| **Kill / Death Ratio (KDR)** | **0.10** | 0.20 | `-0.10` |
| **Steps Survived** | **143.8 ± 71** | 253.8 | `-110.0` |
| **Suicides / Round** | **1.00** | 0.40 | `+0.60` |
| **Wait % of Actions** | **8.8%** | 0.3% | `+8.5%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_dqn_agent** | **50.0%** | 0.0% | 3.70 | 3.20 | 0.10 | 10 | 0 | 143.8 | 8.8% |
| rule_based_agent_0 | 50.0% | 50.0% | 3.80 | 2.30 | 0.30 | 5 | 0 | 236.6 | 0.5% |
| rule_based_agent_1 | 20.0% | 50.0% | 1.60 | 1.60 | 0.00 | 4 | 2 | 267.3 | 0.4% |
| rule_based_agent_2 | 30.0% | 50.0% | 1.90 | 1.90 | 0.00 | 3 | 2 | 257.6 | 0.0% |
| *Rule-Based Avg* | *33.3%* | *50.0%* | *2.43* | *1.93* | *0.10* | *4.0* | *1.3* | *253.8* | *0.3%* |

## Action Profile for `my_spatial_dqn_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1127 | 112.7 | 78.4% |
| Bombs Placed | 166 | 16.6 | 11.5% |
| Waited (WAIT) | 126 | 12.6 | 8.8% |
| Invalid Actions | 19 | 1.9 | 1.3% |
| Crates Destroyed | 303 | 30.3 | - |
| **Total Actions** | **1438** | **143.8** | **100.0%** |
