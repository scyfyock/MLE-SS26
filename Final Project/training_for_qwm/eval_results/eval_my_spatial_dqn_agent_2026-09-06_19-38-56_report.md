# BombeRLe Evaluation Report: `my_spatial_dqn_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_19-35-54_spatial_dqn/my-saved-model-spatial-dqn.pt`
- **Source:** Latest run (run_2026-09-06_19-35-54_spatial_dqn) final model
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-06 19:39:04

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **20.0%** | 40.0% | `-20.0%` |
| **Survival Rate** | **0.0%** | 46.7% | `-46.7%` |
| **Score / Round** | **2.00 ± 0.9** | 3.33 | `-1.33` |
| **Coins / Round** | **2.00 ± 0.9** | 2.33 | `-0.33` |
| **Coin Share** | **22.2%** | 25.9% | `-3.7%` |
| **Kills / Round** | **0.00 ± 0.0** | 0.20 | `-0.20` |
| **Kill / Death Ratio (KDR)** | **0.00** | 0.34 | `-0.34` |
| **Steps Survived** | **139.0 ± 88** | 222.6 | `-83.6` |
| **Suicides / Round** | **1.00** | 0.47 | `+0.53` |
| **Wait % of Actions** | **7.6%** | 0.3% | `+7.3%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_dqn_agent** | **20.0%** | 0.0% | 2.00 | 2.00 | 0.00 | 5 | 0 | 139.0 | 7.6% |
| rule_based_agent_0 | 20.0% | 0.0% | 3.20 | 2.20 | 0.20 | 4 | 1 | 122.4 | 0.0% |
| rule_based_agent_1 | 20.0% | 60.0% | 3.00 | 2.00 | 0.20 | 2 | 0 | 255.4 | 0.3% |
| rule_based_agent_2 | 80.0% | 80.0% | 3.80 | 2.80 | 0.20 | 1 | 2 | 290.0 | 0.6% |
| *Rule-Based Avg* | *40.0%* | *46.7%* | *3.33* | *2.33* | *0.20* | *2.3* | *1.0* | *222.6* | *0.3%* |

## Action Profile for `my_spatial_dqn_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 555 | 111.0 | 79.9% |
| Bombs Placed | 81 | 16.2 | 11.7% |
| Waited (WAIT) | 53 | 10.6 | 7.6% |
| Invalid Actions | 6 | 1.2 | 0.9% |
| Crates Destroyed | 151 | 30.2 | - |
| **Total Actions** | **695** | **139.0** | **100.0%** |
