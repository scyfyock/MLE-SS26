# BombeRLe Evaluation Report: `qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent`
- **Timestamp:** 2026-09-13 17:50:07

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **62.0%** | 48.5% | `+13.5%` |
| **Survival Rate** | **71.5%** | 62.5% | `+9.0%` |
| **Score / Round** | **5.06 ± 2.6** | 4.13 | `+0.92` |
| **Coins / Round** | **4.58 ± 1.9** | 3.91 | `+0.67` |
| **Coin Share** | **54.0%** | 46.0% | `+7.9%` |
| **Kills / Round** | **0.10 ± 0.3** | 0.04 | `+0.05` |
| **Kill / Death Ratio (KDR)** | **0.32** | 0.12 | `+0.20` |
| **Steps Survived** | **335.9 ± 111** | 297.7 | `+38.1` |
| **Suicides / Round** | **0.26** | 0.29 | `-0.04` |
| **Wait % of Actions** | **3.0%** | 0.3% | `+2.7%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent** | **62.0%** | 71.5% | 5.06 | 4.58 | 0.10 | 51 | 9 | 335.9 | 3.0% |
| rule_based_agent | 48.5% | 62.5% | 4.13 | 3.91 | 0.04 | 59 | 19 | 297.7 | 0.3% |
| *Rule-Based Avg* | *48.5%* | *62.5%* | *4.13* | *3.91* | *0.04* | *59.0* | *19.0* | *297.7* | *0.3%* |

## Action Profile for `qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 58009 | 290.0 | 86.4% |
| Bombs Placed | 6752 | 33.8 | 10.1% |
| Waited (WAIT) | 2041 | 10.2 | 3.0% |
| Invalid Actions | 369 | 1.8 | 0.5% |
| Crates Destroyed | 10125 | 50.6 | - |
| **Total Actions** | **67171** | **335.9** | **100.0%** |
