# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit file: agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 01:09:50

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **30.0%** | 46.7% | `-16.7%` |
| **Survival Rate** | **10.0%** | 40.0% | `-30.0%` |
| **Score / Round** | **3.60 ± 2.9** | 3.30 | `+0.30` |
| **Coins / Round** | **2.60 ± 1.6** | 2.13 | `+0.47` |
| **Coin Share** | **28.9%** | 23.7% | `+5.2%` |
| **Kills / Round** | **0.20 ± 0.4** | 0.23 | `-0.03` |
| **Kill / Death Ratio (KDR)** | **0.22** | 0.46 | `-0.24` |
| **Steps Survived** | **155.5 ± 101** | 231.2 | `-75.7` |
| **Suicides / Round** | **0.50** | 0.47 | `+0.03` |
| **Wait % of Actions** | **5.1%** | 0.5% | `+4.6%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **30.0%** | 10.0% | 3.60 | 2.60 | 0.20 | 5 | 4 | 155.5 | 5.1% |
| rule_based_agent_0 | 50.0% | 40.0% | 2.90 | 2.40 | 0.10 | 3 | 3 | 235.7 | 0.6% |
| rule_based_agent_1 | 20.0% | 20.0% | 2.40 | 1.40 | 0.20 | 7 | 2 | 194.6 | 0.3% |
| rule_based_agent_2 | 70.0% | 60.0% | 4.60 | 2.60 | 0.40 | 4 | 0 | 263.4 | 0.6% |
| *Rule-Based Avg* | *46.7%* | *40.0%* | *3.30* | *2.13* | *0.23* | *4.7* | *1.7* | *231.2* | *0.5%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1302 | 130.2 | 83.7% |
| Bombs Placed | 159 | 15.9 | 10.2% |
| Waited (WAIT) | 80 | 8.0 | 5.1% |
| Invalid Actions | 14 | 1.4 | 0.9% |
| Crates Destroyed | 275 | 27.5 | - |
| **Total Actions** | **1555** | **155.5** | **100.0%** |
