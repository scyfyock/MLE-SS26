# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 2
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 00:50:30

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **50.0%** | 33.3% | `+16.7%` |
| **Survival Rate** | **0.0%** | 50.0% | `-50.0%` |
| **Score / Round** | **7.00 ± 5.0** | 3.17 | `+3.83` |
| **Coins / Round** | **2.00 ± 0.0** | 2.33 | `-0.33` |
| **Coin Share** | **22.2%** | 25.9% | `-3.7%` |
| **Kills / Round** | **1.00 ± 1.0** | 0.17 | `+0.83` |
| **Kill / Death Ratio (KDR)** | **1.00** | 0.33 | `+0.67` |
| **Steps Survived** | **164.0 ± 70** | 246.8 | `-82.8` |
| **Suicides / Round** | **1.00** | 0.17 | `+0.83` |
| **Wait % of Actions** | **6.7%** | 0.8% | `+5.9%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **50.0%** | 0.0% | 7.00 | 2.00 | 1.00 | 2 | 0 | 164.0 | 6.7% |
| rule_based_agent_0 | 0.0% | 50.0% | 2.00 | 2.00 | 0.00 | 0 | 1 | 280.5 | 0.4% |
| rule_based_agent_1 | 50.0% | 50.0% | 4.00 | 1.50 | 0.50 | 0 | 1 | 233.0 | 0.6% |
| rule_based_agent_2 | 50.0% | 50.0% | 3.50 | 3.50 | 0.00 | 1 | 1 | 227.0 | 1.3% |
| *Rule-Based Avg* | *33.3%* | *50.0%* | *3.17* | *2.33* | *0.17* | *0.3* | *1.0* | *246.8* | *0.8%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 267 | 133.5 | 81.4% |
| Bombs Placed | 36 | 18.0 | 11.0% |
| Waited (WAIT) | 22 | 11.0 | 6.7% |
| Invalid Actions | 3 | 1.5 | 0.9% |
| Crates Destroyed | 45 | 22.5 | - |
| **Total Actions** | **328** | **164.0** | **100.0%** |
