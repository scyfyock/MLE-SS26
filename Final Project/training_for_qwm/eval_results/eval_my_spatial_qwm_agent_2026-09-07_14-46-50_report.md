# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 14:52:32

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **32.0%** | 36.7% | `-4.7%` |
| **Survival Rate** | **38.0%** | 44.0% | `-6.0%` |
| **Score / Round** | **3.10 ± 2.3** | 2.99 | `+0.11` |
| **Coins / Round** | **2.30 ± 1.4** | 2.23 | `+0.07` |
| **Coin Share** | **25.6%** | 24.8% | `+0.8%` |
| **Kills / Round** | **0.16 ± 0.4** | 0.15 | `+0.01` |
| **Kill / Death Ratio (KDR)** | **0.24** | 0.24 | `+0.00` |
| **Steps Survived** | **275.2 ± 118** | 238.9 | `+36.3` |
| **Suicides / Round** | **0.52** | 0.48 | `+0.04` |
| **Wait % of Actions** | **4.9%** | 0.3% | `+4.6%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **32.0%** | 38.0% | 3.10 | 2.30 | 0.16 | 26 | 7 | 275.2 | 4.9% |
| rule_based_agent_0 | 48.0% | 42.0% | 3.48 | 2.08 | 0.28 | 25 | 8 | 241.6 | 0.3% |
| rule_based_agent_1 | 28.0% | 42.0% | 2.48 | 2.18 | 0.06 | 24 | 9 | 233.0 | 0.4% |
| rule_based_agent_2 | 34.0% | 48.0% | 3.02 | 2.42 | 0.12 | 23 | 7 | 242.3 | 0.3% |
| *Rule-Based Avg* | *36.7%* | *44.0%* | *2.99* | *2.23* | *0.15* | *24.0* | *8.0* | *238.9* | *0.3%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 11694 | 233.9 | 85.0% |
| Bombs Placed | 1281 | 25.6 | 9.3% |
| Waited (WAIT) | 679 | 13.6 | 4.9% |
| Invalid Actions | 106 | 2.1 | 0.8% |
| Crates Destroyed | 1196 | 23.9 | - |
| **Total Actions** | **13760** | **275.2** | **100.0%** |
