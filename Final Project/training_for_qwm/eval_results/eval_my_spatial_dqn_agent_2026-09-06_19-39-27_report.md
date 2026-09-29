# BombeRLe Evaluation Report: `my_spatial_dqn_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-06_18-24-53_spatial_dqn (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-06 19:39:34

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **20.0%** | 33.3% | `-13.3%` |
| **Survival Rate** | **0.0%** | 46.7% | `-46.7%` |
| **Score / Round** | **2.20 ± 2.5** | 3.93 | `-1.73` |
| **Coins / Round** | **1.20 ± 0.7** | 2.60 | `-1.40` |
| **Coin Share** | **13.3%** | 28.9% | `-15.6%` |
| **Kills / Round** | **0.20 ± 0.4** | 0.27 | `-0.07` |
| **Kill / Death Ratio (KDR)** | **0.20** | 1.33 | `-1.13` |
| **Steps Survived** | **143.8 ± 86** | 234.7 | `-90.9` |
| **Suicides / Round** | **1.00** | 0.53 | `+0.47` |
| **Wait % of Actions** | **7.8%** | 0.3% | `+7.5%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_dqn_agent** | **20.0%** | 0.0% | 2.20 | 1.20 | 0.20 | 5 | 0 | 143.8 | 7.8% |
| rule_based_agent_0 | 20.0% | 40.0% | 2.20 | 2.20 | 0.00 | 3 | 2 | 240.2 | 0.3% |
| rule_based_agent_1 | 80.0% | 80.0% | 6.80 | 2.80 | 0.80 | 1 | 0 | 291.0 | 0.6% |
| rule_based_agent_2 | 0.0% | 20.0% | 2.80 | 2.80 | 0.00 | 4 | 3 | 173.0 | 0.0% |
| *Rule-Based Avg* | *33.3%* | *46.7%* | *3.93* | *2.60* | *0.27* | *2.7* | *1.7* | *234.7* | *0.3%* |

## Action Profile for `my_spatial_dqn_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 574 | 114.8 | 79.8% |
| Bombs Placed | 85 | 17.0 | 11.8% |
| Waited (WAIT) | 56 | 11.2 | 7.8% |
| Invalid Actions | 4 | 0.8 | 0.6% |
| Crates Destroyed | 126 | 25.2 | - |
| **Total Actions** | **719** | **143.8** | **100.0%** |
