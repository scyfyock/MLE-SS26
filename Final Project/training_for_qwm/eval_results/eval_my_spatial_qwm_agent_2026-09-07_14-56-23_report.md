# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 15:02:24

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **44.0%** | 37.3% | `+6.7%` |
| **Survival Rate** | **36.0%** | 43.3% | `-7.3%` |
| **Score / Round** | **3.72 ± 2.9** | 3.12 | `+0.60` |
| **Coins / Round** | **2.52 ± 1.3** | 2.15 | `+0.37` |
| **Coin Share** | **28.1%** | 24.0% | `+4.1%` |
| **Kills / Round** | **0.24 ± 0.5** | 0.19 | `+0.05` |
| **Kill / Death Ratio (KDR)** | **0.33** | 0.33 | `-0.00` |
| **Steps Survived** | **267.3 ± 121** | 242.2 | `+25.1` |
| **Suicides / Round** | **0.50** | 0.41 | `+0.09` |
| **Wait % of Actions** | **4.5%** | 0.4% | `+4.1%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **44.0%** | 36.0% | 3.72 | 2.52 | 0.24 | 25 | 11 | 267.3 | 4.5% |
| rule_based_agent_0 | 48.0% | 52.0% | 3.70 | 2.20 | 0.30 | 18 | 8 | 261.3 | 0.5% |
| rule_based_agent_1 | 38.0% | 38.0% | 3.16 | 2.46 | 0.14 | 22 | 14 | 238.0 | 0.3% |
| rule_based_agent_2 | 26.0% | 40.0% | 2.50 | 1.80 | 0.14 | 22 | 8 | 227.2 | 0.4% |
| *Rule-Based Avg* | *37.3%* | *43.3%* | *3.12* | *2.15* | *0.19* | *20.7* | *10.0* | *242.2* | *0.4%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 11421 | 228.4 | 85.5% |
| Bombs Placed | 1225 | 24.5 | 9.2% |
| Waited (WAIT) | 598 | 12.0 | 4.5% |
| Invalid Actions | 120 | 2.4 | 0.9% |
| Crates Destroyed | 1124 | 22.5 | - |
| **Total Actions** | **13364** | **267.3** | **100.0%** |
