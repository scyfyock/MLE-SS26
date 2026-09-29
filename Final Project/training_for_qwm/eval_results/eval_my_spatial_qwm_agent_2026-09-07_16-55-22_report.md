# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 17:15:42

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **42.0%** | 33.3% | `+8.7%` |
| **Survival Rate** | **50.0%** | 46.0% | `+4.0%` |
| **Score / Round** | **3.24 ± 2.5** | 2.73 | `+0.51` |
| **Coins / Round** | **2.54 ± 1.2** | 2.13 | `+0.41` |
| **Coin Share** | **28.5%** | 23.8% | `+4.6%` |
| **Kills / Round** | **0.14 ± 0.4** | 0.12 | `+0.02` |
| **Kill / Death Ratio (KDR)** | **0.27** | 0.22 | `+0.05` |
| **Steps Survived** | **267.8 ± 132** | 248.6 | `+19.3` |
| **Suicides / Round** | **0.42** | 0.46 | `-0.04` |
| **Wait % of Actions** | **4.6%** | 0.3% | `+4.4%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **42.0%** | 50.0% | 3.24 | 2.54 | 0.14 | 21 | 5 | 267.8 | 4.6% |
| rule_based_agent_0 | 30.0% | 38.0% | 2.44 | 2.04 | 0.08 | 26 | 9 | 211.0 | 0.2% |
| rule_based_agent_1 | 40.0% | 58.0% | 3.10 | 2.30 | 0.16 | 18 | 5 | 298.4 | 0.3% |
| rule_based_agent_2 | 30.0% | 42.0% | 2.64 | 2.04 | 0.12 | 25 | 6 | 236.2 | 0.3% |
| *Rule-Based Avg* | *33.3%* | *46.0%* | *2.73* | *2.13* | *0.12* | *23.0* | *6.7* | *248.6* | *0.3%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 11326 | 226.5 | 84.6% |
| Bombs Placed | 1371 | 27.4 | 10.2% |
| Waited (WAIT) | 620 | 12.4 | 4.6% |
| Invalid Actions | 75 | 1.5 | 0.6% |
| Crates Destroyed | 1154 | 23.1 | - |
| **Total Actions** | **13392** | **267.8** | **100.0%** |
