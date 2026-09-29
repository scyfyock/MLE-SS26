# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit file: agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 02:09:47

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **48.0%** | 26.7% | `+21.3%` |
| **Survival Rate** | **80.0%** | 43.3% | `+36.7%` |
| **Score / Round** | **4.26 ± 2.7** | 2.88 | `+1.38` |
| **Coins / Round** | **2.96 ± 1.4** | 2.01 | `+0.95` |
| **Coin Share** | **32.9%** | 22.4% | `+10.5%` |
| **Kills / Round** | **0.26 ± 0.5** | 0.17 | `+0.09` |
| **Kill / Death Ratio (KDR)** | **1.08** | 0.27 | `+0.81` |
| **Steps Survived** | **345.4 ± 104** | 250.7 | `+94.7` |
| **Suicides / Round** | **0.06** | 0.43 | `-0.37` |
| **Wait % of Actions** | **9.4%** | 0.1% | `+9.4%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **48.0%** | 80.0% | 4.26 | 2.96 | 0.26 | 3 | 9 | 345.4 | 9.4% |
| rule_based_agent_0 | 28.0% | 46.0% | 3.16 | 2.16 | 0.20 | 24 | 7 | 260.0 | 0.1% |
| rule_based_agent_1 | 30.0% | 38.0% | 2.84 | 2.04 | 0.16 | 22 | 12 | 248.0 | 0.0% |
| rule_based_agent_2 | 22.0% | 46.0% | 2.64 | 1.84 | 0.16 | 19 | 11 | 244.1 | 0.1% |
| *Rule-Based Avg* | *26.7%* | *43.3%* | *2.88* | *2.01* | *0.17* | *21.7* | *10.0* | *250.7* | *0.1%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 14395 | 287.9 | 83.4% |
| Bombs Placed | 1109 | 22.2 | 6.4% |
| Waited (WAIT) | 1625 | 32.5 | 9.4% |
| Invalid Actions | 141 | 2.8 | 0.8% |
| Crates Destroyed | 1672 | 33.4 | - |
| **Total Actions** | **17270** | **345.4** | **100.0%** |
