# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 100
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-20 16:43:38

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **42.0%** | 28.0% | `+14.0%` |
| **Survival Rate** | **76.0%** | 32.7% | `+43.3%` |
| **Score / Round** | **3.80 ± 3.0** | 2.73 | `+1.07` |
| **Coins / Round** | **2.70 ± 1.3** | 2.10 | `+0.60` |
| **Coin Share** | **30.0%** | 23.3% | `+6.7%` |
| **Kills / Round** | **0.22 ± 0.5** | 0.13 | `+0.09` |
| **Kill / Death Ratio (KDR)** | **0.88** | 0.17 | `+0.71` |
| **Steps Survived** | **316.9 ± 128** | 227.2 | `+89.7` |
| **Suicides / Round** | **0.18** | 0.55 | `-0.37` |
| **Wait % of Actions** | **1.7%** | 0.2% | `+1.5%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **42.0%** | 76.0% | 3.80 | 2.70 | 0.22 | 18 | 7 | 316.9 | 1.7% |
| rule_based_agent_0 | 31.0% | 33.0% | 2.82 | 2.37 | 0.09 | 53 | 18 | 230.1 | 0.2% |
| rule_based_agent_1 | 28.0% | 34.0% | 2.81 | 1.96 | 0.17 | 55 | 16 | 228.4 | 0.2% |
| rule_based_agent_2 | 25.0% | 31.0% | 2.57 | 1.97 | 0.12 | 57 | 19 | 223.0 | 0.1% |
| *Rule-Based Avg* | *28.0%* | *32.7%* | *2.73* | *2.10* | *0.13* | *55.0* | *17.7* | *227.2* | *0.2%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 27843 | 278.4 | 87.9% |
| Bombs Placed | 2987 | 29.9 | 9.4% |
| Waited (WAIT) | 531 | 5.3 | 1.7% |
| Invalid Actions | 329 | 3.3 | 1.0% |
| Crates Destroyed | 2742 | 27.4 | - |
| **Total Actions** | **31690** | **316.9** | **100.0%** |
