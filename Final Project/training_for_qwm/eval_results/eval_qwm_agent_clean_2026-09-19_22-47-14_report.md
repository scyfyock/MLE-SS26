# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-19 23:10:31

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **40.5%** | 29.5% | `+11.0%` |
| **Survival Rate** | **57.5%** | 41.7% | `+15.8%` |
| **Score / Round** | **3.66 ± 2.9** | 3.02 | `+0.64` |
| **Coins / Round** | **2.48 ± 1.5** | 2.17 | `+0.32` |
| **Coin Share** | **27.7%** | 24.1% | `+3.5%` |
| **Kills / Round** | **0.23 ± 0.5** | 0.17 | `+0.06` |
| **Kill / Death Ratio (KDR)** | **0.51** | 0.26 | `+0.25` |
| **Steps Survived** | **296.0 ± 136** | 250.5 | `+45.5` |
| **Suicides / Round** | **0.26** | 0.47 | `-0.21` |
| **Wait % of Actions** | **9.4%** | 0.3% | `+9.1%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **40.5%** | 57.5% | 3.66 | 2.48 | 0.23 | 52 | 40 | 296.0 | 9.4% |
| rule_based_agent_0 | 34.0% | 44.5% | 3.35 | 2.27 | 0.21 | 94 | 33 | 252.6 | 0.3% |
| rule_based_agent_1 | 27.0% | 37.5% | 2.76 | 2.11 | 0.13 | 95 | 40 | 239.1 | 0.3% |
| rule_based_agent_2 | 27.5% | 43.0% | 2.94 | 2.12 | 0.17 | 92 | 36 | 259.8 | 0.4% |
| *Rule-Based Avg* | *29.5%* | *41.7%* | *3.02* | *2.17* | *0.17* | *93.7* | *36.3* | *250.5* | *0.3%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 47010 | 235.1 | 79.4% |
| Bombs Placed | 5992 | 30.0 | 10.1% |
| Waited (WAIT) | 5569 | 27.8 | 9.4% |
| Invalid Actions | 625 | 3.1 | 1.1% |
| Crates Destroyed | 5319 | 26.6 | - |
| **Total Actions** | **59196** | **296.0** | **100.0%** |
