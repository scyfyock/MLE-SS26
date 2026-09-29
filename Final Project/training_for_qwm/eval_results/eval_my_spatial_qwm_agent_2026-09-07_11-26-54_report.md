# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 11:38:11

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **34.0%** | 28.0% | `+6.0%` |
| **Survival Rate** | **48.0%** | 33.3% | `+14.7%` |
| **Score / Round** | **4.12 ± 3.9** | 2.99 | `+1.13` |
| **Coins / Round** | **2.42 ± 1.6** | 2.16 | `+0.26` |
| **Coin Share** | **27.2%** | 24.3% | `+2.9%` |
| **Kills / Round** | **0.34 ± 0.7** | 0.17 | `+0.17` |
| **Kill / Death Ratio (KDR)** | **0.63** | 0.26 | `+0.37` |
| **Steps Survived** | **246.0 ± 131** | 226.2 | `+19.8` |
| **Suicides / Round** | **0.38** | 0.51 | `-0.13` |
| **Wait % of Actions** | **2.3%** | 0.2% | `+2.1%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **34.0%** | 48.0% | 4.12 | 2.42 | 0.34 | 19 | 8 | 246.0 | 2.3% |
| rule_based_agent_0 | 26.0% | 30.0% | 2.90 | 2.10 | 0.16 | 25 | 15 | 227.2 | 0.2% |
| rule_based_agent_1 | 36.0% | 48.0% | 3.16 | 1.96 | 0.24 | 20 | 6 | 255.4 | 0.4% |
| rule_based_agent_2 | 22.0% | 22.0% | 2.92 | 2.42 | 0.10 | 31 | 13 | 196.1 | 0.2% |
| *Rule-Based Avg* | *28.0%* | *33.3%* | *2.99* | *2.16* | *0.17* | *25.3* | *11.3* | *226.2* | *0.2%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 10637 | 212.7 | 86.0% |
| Bombs Placed | 1271 | 25.4 | 10.3% |
| Waited (WAIT) | 288 | 5.8 | 2.3% |
| Invalid Actions | 174 | 3.5 | 1.4% |
| Crates Destroyed | 1271 | 25.4 | - |
| **Total Actions** | **12370** | **247.4** | **100.0%** |
