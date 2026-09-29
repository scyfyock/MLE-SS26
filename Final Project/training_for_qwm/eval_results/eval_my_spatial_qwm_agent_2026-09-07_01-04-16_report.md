# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit file: agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 20
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 01:04:52

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **20.0%** | 36.7% | `-16.7%` |
| **Survival Rate** | **0.0%** | 46.7% | `-46.7%` |
| **Score / Round** | **2.10 ± 2.3** | 2.88 | `-0.78` |
| **Coins / Round** | **1.85 ± 1.5** | 2.38 | `-0.53` |
| **Coin Share** | **20.6%** | 26.5% | `-5.9%` |
| **Kills / Round** | **0.05 ± 0.2** | 0.10 | `-0.05` |
| **Kill / Death Ratio (KDR)** | **0.05** | 0.22 | `-0.17` |
| **Steps Survived** | **108.9 ± 55** | 225.2 | `-116.3` |
| **Suicides / Round** | **0.90** | 0.48 | `+0.42` |
| **Wait % of Actions** | **7.9%** | 0.3% | `+7.6%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **20.0%** | 0.0% | 2.10 | 1.85 | 0.05 | 18 | 2 | 108.9 | 7.9% |
| rule_based_agent_0 | 45.0% | 75.0% | 3.15 | 2.65 | 0.10 | 3 | 2 | 273.5 | 0.4% |
| rule_based_agent_1 | 40.0% | 55.0% | 3.10 | 2.85 | 0.05 | 8 | 3 | 253.0 | 0.4% |
| rule_based_agent_2 | 25.0% | 10.0% | 2.40 | 1.65 | 0.15 | 18 | 0 | 149.2 | 0.1% |
| *Rule-Based Avg* | *36.7%* | *46.7%* | *2.88* | *2.38* | *0.10* | *9.7* | *1.7* | *225.2* | *0.3%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1722 | 86.1 | 79.1% |
| Bombs Placed | 260 | 13.0 | 11.9% |
| Waited (WAIT) | 172 | 8.6 | 7.9% |
| Invalid Actions | 24 | 1.2 | 1.1% |
| Crates Destroyed | 473 | 23.6 | - |
| **Total Actions** | **2178** | **108.9** | **100.0%** |
