# BombeRLe Evaluation Report: `qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-13 17:23:31

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **46.5%** | 29.0% | `+17.5%` |
| **Survival Rate** | **60.5%** | 37.8% | `+22.7%` |
| **Score / Round** | **3.90 ± 3.0** | 3.01 | `+0.88` |
| **Coins / Round** | **2.72 ± 1.5** | 2.09 | `+0.63` |
| **Coin Share** | **30.3%** | 23.2% | `+7.1%` |
| **Kills / Round** | **0.23 ± 0.5** | 0.18 | `+0.05` |
| **Kill / Death Ratio (KDR)** | **0.52** | 0.27 | `+0.25` |
| **Steps Survived** | **279.0 ± 138** | 237.5 | `+41.5` |
| **Suicides / Round** | **0.28** | 0.49 | `-0.21` |
| **Wait % of Actions** | **2.9%** | 0.2% | `+2.7%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent** | **46.5%** | 60.5% | 3.90 | 2.72 | 0.23 | 56 | 35 | 279.0 | 2.9% |
| rule_based_agent_0 | 26.5% | 38.0% | 2.90 | 2.12 | 0.15 | 88 | 49 | 232.6 | 0.2% |
| rule_based_agent_1 | 29.5% | 35.5% | 2.97 | 2.02 | 0.19 | 105 | 44 | 233.3 | 0.1% |
| rule_based_agent_2 | 31.0% | 40.0% | 3.17 | 2.12 | 0.21 | 101 | 30 | 246.7 | 0.2% |
| *Rule-Based Avg* | *29.0%* | *37.8%* | *3.01* | *2.09* | *0.18* | *98.0* | *41.0* | *237.5* | *0.2%* |

## Action Profile for `qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 48136 | 240.7 | 86.3% |
| Bombs Placed | 5324 | 26.6 | 9.5% |
| Waited (WAIT) | 1619 | 8.1 | 2.9% |
| Invalid Actions | 720 | 3.6 | 1.3% |
| Crates Destroyed | 5254 | 26.3 | - |
| **Total Actions** | **55799** | **279.0** | **100.0%** |
