# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit file: agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 00:49:29

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **40.0%** | 26.7% | `+13.3%` |
| **Survival Rate** | **0.0%** | 60.0% | `-60.0%` |
| **Score / Round** | **3.20 ± 2.4** | 2.27 | `+0.93` |
| **Coins / Round** | **2.20 ± 1.5** | 2.27 | `-0.07` |
| **Coin Share** | **24.4%** | 25.2% | `-0.7%` |
| **Kills / Round** | **0.20 ± 0.4** | 0.00 | `+0.20` |
| **Kill / Death Ratio (KDR)** | **0.20** | 0.00 | `+0.20` |
| **Steps Survived** | **130.6 ± 74** | 290.5 | `-159.9` |
| **Suicides / Round** | **1.00** | 0.40 | `+0.60` |
| **Wait % of Actions** | **6.6%** | 0.1% | `+6.5%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **40.0%** | 0.0% | 3.20 | 2.20 | 0.20 | 5 | 0 | 130.6 | 6.6% |
| rule_based_agent_0 | 20.0% | 60.0% | 2.40 | 2.40 | 0.00 | 2 | 0 | 344.0 | 0.1% |
| rule_based_agent_1 | 20.0% | 60.0% | 1.80 | 1.80 | 0.00 | 2 | 0 | 266.6 | 0.1% |
| rule_based_agent_2 | 40.0% | 60.0% | 2.60 | 2.60 | 0.00 | 2 | 1 | 261.0 | 0.2% |
| *Rule-Based Avg* | *26.7%* | *60.0%* | *2.27* | *2.27* | *0.00* | *2.0* | *0.3* | *290.5* | *0.1%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 535 | 107.0 | 81.9% |
| Bombs Placed | 71 | 14.2 | 10.9% |
| Waited (WAIT) | 43 | 8.6 | 6.6% |
| Invalid Actions | 4 | 0.8 | 0.6% |
| Crates Destroyed | 148 | 29.6 | - |
| **Total Actions** | **653** | **130.6** | **100.0%** |
