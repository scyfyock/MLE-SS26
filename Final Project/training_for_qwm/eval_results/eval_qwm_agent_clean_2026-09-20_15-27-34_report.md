# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-20 15:44:21

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **43.0%** | 27.7% | `+15.3%` |
| **Survival Rate** | **69.5%** | 35.0% | `+34.5%` |
| **Score / Round** | **3.67 ± 3.0** | 3.11 | `+0.56` |
| **Coins / Round** | **2.45 ± 1.6** | 2.17 | `+0.28` |
| **Coin Share** | **27.3%** | 24.2% | `+3.1%` |
| **Kills / Round** | **0.24 ± 0.5** | 0.19 | `+0.06` |
| **Kill / Death Ratio (KDR)** | **0.70** | 0.26 | `+0.44` |
| **Steps Survived** | **308.6 ± 127** | 234.9 | `+73.7` |
| **Suicides / Round** | **0.18** | 0.52 | `-0.34` |
| **Wait % of Actions** | **1.4%** | 0.2% | `+1.3%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **43.0%** | 69.5% | 3.67 | 2.45 | 0.24 | 36 | 34 | 308.6 | 1.4% |
| rule_based_agent_0 | 31.5% | 36.5% | 3.21 | 2.21 | 0.20 | 101 | 41 | 237.2 | 0.2% |
| rule_based_agent_1 | 27.0% | 35.5% | 3.17 | 2.09 | 0.21 | 107 | 40 | 232.0 | 0.1% |
| rule_based_agent_2 | 24.5% | 33.0% | 2.96 | 2.21 | 0.15 | 106 | 47 | 235.6 | 0.2% |
| *Rule-Based Avg* | *27.7%* | *35.0%* | *3.11* | *2.17* | *0.19* | *104.7* | *42.7* | *234.9* | *0.2%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 54349 | 271.7 | 88.1% |
| Bombs Placed | 5742 | 28.7 | 9.3% |
| Waited (WAIT) | 887 | 4.4 | 1.4% |
| Invalid Actions | 741 | 3.7 | 1.2% |
| Crates Destroyed | 5423 | 27.1 | - |
| **Total Actions** | **61719** | **308.6** | **100.0%** |
