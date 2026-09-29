# BombeRLe Evaluation Report: `qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-13 12:33:17

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **50.0%** | 24.0% | `+26.0%` |
| **Survival Rate** | **72.0%** | 36.0% | `+36.0%` |
| **Score / Round** | **4.42 ± 3.1** | 2.89 | `+1.53` |
| **Coins / Round** | **2.92 ± 1.4** | 2.02 | `+0.90` |
| **Coin Share** | **32.5%** | 22.5% | `+10.0%` |
| **Kills / Round** | **0.30 ± 0.5** | 0.17 | `+0.13` |
| **Kill / Death Ratio (KDR)** | **1.00** | 0.25 | `+0.75` |
| **Steps Survived** | **324.6 ± 122** | 240.7 | `+83.9` |
| **Suicides / Round** | **0.24** | 0.49 | `-0.25` |
| **Wait % of Actions** | **2.8%** | 0.1% | `+2.7%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent** | **50.0%** | 72.0% | 4.42 | 2.92 | 0.30 | 12 | 3 | 324.6 | 2.8% |
| rule_based_agent_0 | 30.0% | 46.0% | 3.16 | 2.06 | 0.22 | 19 | 9 | 266.7 | 0.1% |
| rule_based_agent_1 | 18.0% | 34.0% | 2.78 | 2.08 | 0.14 | 26 | 16 | 244.8 | 0.1% |
| rule_based_agent_2 | 24.0% | 28.0% | 2.72 | 1.92 | 0.16 | 28 | 13 | 210.6 | 0.2% |
| *Rule-Based Avg* | *24.0%* | *36.0%* | *2.89* | *2.02* | *0.17* | *24.3* | *12.7* | *240.7* | *0.1%* |

## Action Profile for `qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 13994 | 279.9 | 86.2% |
| Bombs Placed | 1561 | 31.2 | 9.6% |
| Waited (WAIT) | 458 | 9.2 | 2.8% |
| Invalid Actions | 219 | 4.4 | 1.3% |
| Crates Destroyed | 1409 | 28.2 | - |
| **Total Actions** | **16232** | **324.6** | **100.0%** |
