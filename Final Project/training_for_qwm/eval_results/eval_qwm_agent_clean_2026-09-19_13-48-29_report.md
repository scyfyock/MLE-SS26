# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 100
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-19 13:55:40

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **38.0%** | 30.7% | `+7.3%` |
| **Survival Rate** | **52.0%** | 40.7% | `+11.3%` |
| **Score / Round** | **3.77 ± 3.2** | 3.19 | `+0.58` |
| **Coins / Round** | **2.52 ± 1.6** | 2.16 | `+0.36` |
| **Coin Share** | **28.0%** | 24.0% | `+4.0%` |
| **Kills / Round** | **0.25 ± 0.5** | 0.21 | `+0.04` |
| **Kill / Death Ratio (KDR)** | **0.47** | 0.30 | `+0.17` |
| **Steps Survived** | **281.5 ± 137** | 251.7 | `+29.8` |
| **Suicides / Round** | **0.33** | 0.48 | `-0.15` |
| **Wait % of Actions** | **9.2%** | 0.3% | `+8.9%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **38.0%** | 52.0% | 3.77 | 2.52 | 0.25 | 33 | 20 | 281.5 | 9.2% |
| rule_based_agent_0 | 33.0% | 42.0% | 3.37 | 2.32 | 0.21 | 47 | 24 | 258.9 | 0.3% |
| rule_based_agent_1 | 34.0% | 38.0% | 3.41 | 2.11 | 0.26 | 50 | 16 | 242.1 | 0.2% |
| rule_based_agent_2 | 25.0% | 42.0% | 2.80 | 2.05 | 0.15 | 46 | 27 | 253.9 | 0.5% |
| *Rule-Based Avg* | *30.7%* | *40.7%* | *3.19* | *2.16* | *0.21* | *47.7* | *22.3* | *251.7* | *0.3%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 22339 | 223.4 | 79.4% |
| Bombs Placed | 2934 | 29.3 | 10.4% |
| Waited (WAIT) | 2600 | 26.0 | 9.2% |
| Invalid Actions | 276 | 2.8 | 1.0% |
| Crates Destroyed | 2789 | 27.9 | - |
| **Total Actions** | **28149** | **281.5** | **100.0%** |
