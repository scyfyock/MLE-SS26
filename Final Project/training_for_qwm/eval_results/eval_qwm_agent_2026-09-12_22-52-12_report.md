# BombeRLe Evaluation Report: `qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-12 23:06:11

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **43.5%** | 29.2% | `+14.3%` |
| **Survival Rate** | **66.0%** | 33.8% | `+32.2%` |
| **Score / Round** | **4.05 ± 3.1** | 3.02 | `+1.03` |
| **Coins / Round** | **2.68 ± 1.4** | 2.10 | `+0.58` |
| **Coin Share** | **29.8%** | 23.4% | `+6.4%` |
| **Kills / Round** | **0.28 ± 0.6** | 0.18 | `+0.09` |
| **Kill / Death Ratio (KDR)** | **0.71** | 0.24 | `+0.46` |
| **Steps Survived** | **299.6 ± 130** | 226.8 | `+72.8` |
| **Suicides / Round** | **0.26** | 0.53 | `-0.27` |
| **Wait % of Actions** | **2.7%** | 0.2% | `+2.5%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent** | **43.5%** | 66.0% | 4.05 | 2.68 | 0.28 | 51 | 27 | 299.6 | 2.7% |
| rule_based_agent_0 | 29.0% | 36.5% | 3.02 | 2.17 | 0.17 | 103 | 38 | 240.6 | 0.2% |
| rule_based_agent_1 | 32.5% | 35.0% | 3.16 | 2.13 | 0.20 | 97 | 54 | 221.9 | 0.2% |
| rule_based_agent_2 | 26.0% | 30.0% | 2.88 | 2.00 | 0.17 | 117 | 46 | 218.0 | 0.2% |
| *Rule-Based Avg* | *29.2%* | *33.8%* | *3.02* | *2.10* | *0.18* | *105.7* | *46.0* | *226.8* | *0.2%* |

## Action Profile for `qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 51863 | 259.3 | 86.6% |
| Bombs Placed | 5612 | 28.1 | 9.4% |
| Waited (WAIT) | 1625 | 8.1 | 2.7% |
| Invalid Actions | 820 | 4.1 | 1.4% |
| Crates Destroyed | 5604 | 28.0 | - |
| **Total Actions** | **59920** | **299.6** | **100.0%** |
