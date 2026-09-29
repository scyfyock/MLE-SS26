# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit file: agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 01:08:18

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **40.0%** | 33.3% | `+6.7%` |
| **Survival Rate** | **10.0%** | 46.7% | `-36.7%` |
| **Score / Round** | **3.30 ± 2.2** | 2.70 | `+0.60` |
| **Coins / Round** | **2.80 ± 1.2** | 2.03 | `+0.77` |
| **Coin Share** | **31.5%** | 22.8% | `+8.6%` |
| **Kills / Round** | **0.10 ± 0.3** | 0.13 | `-0.03` |
| **Kill / Death Ratio (KDR)** | **0.11** | 0.28 | `-0.17` |
| **Steps Survived** | **129.4 ± 65** | 240.7 | `-111.3` |
| **Suicides / Round** | **0.80** | 0.47 | `+0.33` |
| **Wait % of Actions** | **7.6%** | 0.2% | `+7.3%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **40.0%** | 10.0% | 3.30 | 2.80 | 0.10 | 8 | 1 | 129.4 | 7.6% |
| rule_based_agent_0 | 30.0% | 50.0% | 2.90 | 1.90 | 0.20 | 4 | 2 | 259.0 | 0.2% |
| rule_based_agent_1 | 30.0% | 30.0% | 2.50 | 2.50 | 0.00 | 6 | 2 | 191.5 | 0.3% |
| rule_based_agent_2 | 40.0% | 60.0% | 2.70 | 1.70 | 0.20 | 4 | 0 | 271.6 | 0.3% |
| *Rule-Based Avg* | *33.3%* | *46.7%* | *2.70* | *2.03* | *0.13* | *4.7* | *1.3* | *240.7* | *0.2%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1039 | 103.9 | 80.3% |
| Bombs Placed | 137 | 13.7 | 10.6% |
| Waited (WAIT) | 98 | 9.8 | 7.6% |
| Invalid Actions | 20 | 2.0 | 1.5% |
| Crates Destroyed | 272 | 27.2 | - |
| **Total Actions** | **1294** | **129.4** | **100.0%** |
