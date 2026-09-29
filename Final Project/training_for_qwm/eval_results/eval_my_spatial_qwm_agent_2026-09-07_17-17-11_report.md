# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 17:28:10

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **44.0%** | 30.0% | `+14.0%` |
| **Survival Rate** | **48.0%** | 43.3% | `+4.7%` |
| **Score / Round** | **3.76 ± 2.1** | 2.65 | `+1.11` |
| **Coins / Round** | **2.86 ± 1.3** | 2.01 | `+0.85` |
| **Coin Share** | **32.1%** | 22.6% | `+9.5%` |
| **Kills / Round** | **0.18 ± 0.4** | 0.13 | `+0.05` |
| **Kill / Death Ratio (KDR)** | **0.31** | 0.21 | `+0.10` |
| **Steps Survived** | **292.9 ± 125** | 242.3 | `+50.6` |
| **Suicides / Round** | **0.38** | 0.49 | `-0.11` |
| **Wait % of Actions** | **5.2%** | 0.3% | `+4.8%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **44.0%** | 48.0% | 3.76 | 2.86 | 0.18 | 19 | 10 | 292.9 | 5.2% |
| rule_based_agent_0 | 32.0% | 34.0% | 2.82 | 2.12 | 0.14 | 29 | 9 | 216.7 | 0.4% |
| rule_based_agent_1 | 32.0% | 50.0% | 2.82 | 2.22 | 0.12 | 22 | 3 | 262.8 | 0.2% |
| rule_based_agent_2 | 26.0% | 46.0% | 2.30 | 1.70 | 0.12 | 22 | 6 | 247.5 | 0.4% |
| *Rule-Based Avg* | *30.0%* | *43.3%* | *2.65* | *2.01* | *0.13* | *24.3* | *6.0* | *242.3* | *0.3%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 12278 | 245.6 | 83.8% |
| Bombs Placed | 1491 | 29.8 | 10.2% |
| Waited (WAIT) | 756 | 15.1 | 5.2% |
| Invalid Actions | 120 | 2.4 | 0.8% |
| Crates Destroyed | 1162 | 23.2 | - |
| **Total Actions** | **14645** | **292.9** | **100.0%** |
