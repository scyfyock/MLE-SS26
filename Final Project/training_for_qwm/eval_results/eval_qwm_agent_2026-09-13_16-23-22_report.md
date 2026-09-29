# BombeRLe Evaluation Report: `qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-13 16:41:14

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **42.5%** | 26.7% | `+15.8%` |
| **Survival Rate** | **60.5%** | 38.2% | `+22.3%` |
| **Score / Round** | **3.98 ± 2.9** | 3.00 | `+0.98` |
| **Coins / Round** | **2.77 ± 1.6** | 2.06 | `+0.71` |
| **Coin Share** | **31.0%** | 23.0% | `+7.9%` |
| **Kills / Round** | **0.24 ± 0.5** | 0.19 | `+0.05` |
| **Kill / Death Ratio (KDR)** | **0.53** | 0.26 | `+0.27` |
| **Steps Survived** | **285.8 ± 141** | 238.5 | `+47.3` |
| **Suicides / Round** | **0.27** | 0.51 | `-0.24` |
| **Wait % of Actions** | **3.0%** | 0.2% | `+2.8%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent** | **42.5%** | 60.5% | 3.98 | 2.77 | 0.24 | 54 | 37 | 285.8 | 3.0% |
| rule_based_agent_0 | 25.5% | 38.0% | 3.02 | 2.08 | 0.19 | 95 | 53 | 238.8 | 0.2% |
| rule_based_agent_1 | 28.5% | 42.5% | 3.02 | 2.17 | 0.17 | 97 | 35 | 249.8 | 0.2% |
| rule_based_agent_2 | 26.0% | 34.0% | 2.94 | 1.95 | 0.20 | 112 | 35 | 226.8 | 0.2% |
| *Rule-Based Avg* | *26.7%* | *38.2%* | *3.00* | *2.06* | *0.19* | *101.3* | *41.0* | *238.5* | *0.2%* |

## Action Profile for `qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 49148 | 245.7 | 86.0% |
| Bombs Placed | 5502 | 27.5 | 9.6% |
| Waited (WAIT) | 1727 | 8.6 | 3.0% |
| Invalid Actions | 780 | 3.9 | 1.4% |
| Crates Destroyed | 5481 | 27.4 | - |
| **Total Actions** | **57157** | **285.8** | **100.0%** |
