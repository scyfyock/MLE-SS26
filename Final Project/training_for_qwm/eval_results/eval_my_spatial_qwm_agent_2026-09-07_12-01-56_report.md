# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 12:06:36

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **46.0%** | 32.0% | `+14.0%` |
| **Survival Rate** | **46.0%** | 31.3% | `+14.7%` |
| **Score / Round** | **3.86 ± 3.2** | 3.27 | `+0.59` |
| **Coins / Round** | **2.36 ± 1.5** | 2.17 | `+0.19` |
| **Coin Share** | **26.6%** | 24.5% | `+2.1%` |
| **Kills / Round** | **0.30 ± 0.6** | 0.22 | `+0.08` |
| **Kill / Death Ratio (KDR)** | **0.52** | 0.29 | `+0.23` |
| **Steps Survived** | **266.5 ± 119** | 227.7 | `+38.8` |
| **Suicides / Round** | **0.42** | 0.50 | `-0.08` |
| **Wait % of Actions** | **1.1%** | 0.2% | `+0.9%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **46.0%** | 46.0% | 3.86 | 2.36 | 0.30 | 21 | 8 | 266.5 | 1.1% |
| rule_based_agent_0 | 26.0% | 32.0% | 3.06 | 2.16 | 0.18 | 22 | 17 | 231.8 | 0.3% |
| rule_based_agent_1 | 46.0% | 34.0% | 3.66 | 2.26 | 0.28 | 26 | 9 | 227.3 | 0.3% |
| rule_based_agent_2 | 24.0% | 28.0% | 3.10 | 2.10 | 0.20 | 27 | 14 | 224.0 | 0.1% |
| *Rule-Based Avg* | *32.0%* | *31.3%* | *3.27* | *2.17* | *0.22* | *25.0* | *13.3* | *227.7* | *0.2%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 12091 | 241.8 | 90.7% |
| Bombs Placed | 910 | 18.2 | 6.8% |
| Waited (WAIT) | 150 | 3.0 | 1.1% |
| Invalid Actions | 174 | 3.5 | 1.3% |
| Crates Destroyed | 754 | 15.1 | - |
| **Total Actions** | **13325** | **266.5** | **100.0%** |
