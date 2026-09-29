# BombeRLe Evaluation Report: `qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-13 16:53:02

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **43.5%** | 28.8% | `+14.7%` |
| **Survival Rate** | **57.5%** | 37.3% | `+20.2%` |
| **Score / Round** | **4.00 ± 3.1** | 3.04 | `+0.96` |
| **Coins / Round** | **2.75 ± 1.5** | 2.08 | `+0.67` |
| **Coin Share** | **30.6%** | 23.1% | `+7.4%` |
| **Kills / Round** | **0.25 ± 0.5** | 0.19 | `+0.06` |
| **Kill / Death Ratio (KDR)** | **0.52** | 0.27 | `+0.25` |
| **Steps Survived** | **286.8 ± 135** | 236.2 | `+50.6` |
| **Suicides / Round** | **0.28** | 0.50 | `-0.22` |
| **Wait % of Actions** | **3.1%** | 0.2% | `+2.9%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent** | **43.5%** | 57.5% | 4.00 | 2.75 | 0.25 | 55 | 41 | 286.8 | 3.1% |
| rule_based_agent_0 | 33.5% | 40.5% | 3.41 | 2.31 | 0.22 | 93 | 43 | 246.9 | 0.2% |
| rule_based_agent_1 | 27.0% | 39.5% | 2.85 | 1.84 | 0.20 | 103 | 33 | 233.7 | 0.2% |
| rule_based_agent_2 | 26.0% | 32.0% | 2.85 | 2.08 | 0.15 | 103 | 48 | 228.0 | 0.2% |
| *Rule-Based Avg* | *28.8%* | *37.3%* | *3.04* | *2.08* | *0.19* | *99.7* | *41.3* | *236.2* | *0.2%* |

## Action Profile for `qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 49375 | 246.9 | 86.1% |
| Bombs Placed | 5452 | 27.3 | 9.5% |
| Waited (WAIT) | 1751 | 8.8 | 3.1% |
| Invalid Actions | 785 | 3.9 | 1.4% |
| Crates Destroyed | 5430 | 27.1 | - |
| **Total Actions** | **57363** | **286.8** | **100.0%** |
