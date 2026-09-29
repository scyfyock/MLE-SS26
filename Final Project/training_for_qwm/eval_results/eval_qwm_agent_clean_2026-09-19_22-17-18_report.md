# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-19 22:40:11

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **32.5%** | 29.2% | `+3.3%` |
| **Survival Rate** | **60.5%** | 41.7% | `+18.8%` |
| **Score / Round** | **3.31 ± 2.6** | 3.11 | `+0.21` |
| **Coins / Round** | **2.39 ± 1.5** | 2.19 | `+0.20` |
| **Coin Share** | **26.7%** | 24.4% | `+2.2%` |
| **Kills / Round** | **0.18 ± 0.4** | 0.18 | `+0.00` |
| **Kill / Death Ratio (KDR)** | **0.40** | 0.28 | `+0.12` |
| **Steps Survived** | **294.4 ± 140** | 247.2 | `+47.3` |
| **Suicides / Round** | **0.26** | 0.49 | `-0.23` |
| **Wait % of Actions** | **9.1%** | 0.3% | `+8.8%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **32.5%** | 60.5% | 3.31 | 2.39 | 0.18 | 52 | 41 | 294.4 | 9.1% |
| rule_based_agent_0 | 36.5% | 41.0% | 3.46 | 2.39 | 0.21 | 101 | 33 | 243.0 | 0.3% |
| rule_based_agent_1 | 26.5% | 45.0% | 3.00 | 2.23 | 0.15 | 84 | 35 | 258.1 | 0.3% |
| rule_based_agent_2 | 24.5% | 39.0% | 2.85 | 1.95 | 0.18 | 106 | 38 | 240.4 | 0.3% |
| *Rule-Based Avg* | *29.2%* | *41.7%* | *3.11* | *2.19* | *0.18* | *97.0* | *35.3* | *247.2* | *0.3%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 46937 | 234.7 | 79.7% |
| Bombs Placed | 5944 | 29.7 | 10.1% |
| Waited (WAIT) | 5365 | 26.8 | 9.1% |
| Invalid Actions | 643 | 3.2 | 1.1% |
| Crates Destroyed | 5281 | 26.4 | - |
| **Total Actions** | **58889** | **294.4** | **100.0%** |
