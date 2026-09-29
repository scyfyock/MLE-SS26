# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-14 02:39:25

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **46.0%** | 29.2% | `+16.8%` |
| **Survival Rate** | **62.5%** | 36.5% | `+26.0%` |
| **Score / Round** | **4.04 ± 3.0** | 2.90 | `+1.14` |
| **Coins / Round** | **2.82 ± 1.6** | 2.05 | `+0.77` |
| **Coin Share** | **31.4%** | 22.9% | `+8.5%` |
| **Kills / Round** | **0.24 ± 0.5** | 0.17 | `+0.07` |
| **Kill / Death Ratio (KDR)** | **0.63** | 0.24 | `+0.39` |
| **Steps Survived** | **309.2 ± 127** | 238.0 | `+71.2` |
| **Suicides / Round** | **0.26** | 0.50 | `-0.24` |
| **Wait % of Actions** | **6.7%** | 0.3% | `+6.4%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **46.0%** | 62.5% | 4.04 | 2.82 | 0.24 | 52 | 26 | 309.2 | 6.7% |
| rule_based_agent_0 | 26.0% | 39.5% | 2.70 | 2.05 | 0.13 | 96 | 36 | 248.4 | 0.3% |
| rule_based_agent_1 | 32.5% | 39.0% | 3.15 | 2.23 | 0.18 | 93 | 44 | 246.4 | 0.4% |
| rule_based_agent_2 | 29.0% | 31.0% | 2.86 | 1.89 | 0.20 | 109 | 45 | 219.2 | 0.3% |
| *Rule-Based Avg* | *29.2%* | *36.5%* | *2.90* | *2.05* | *0.17* | *99.3* | *41.7* | *238.0* | *0.3%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 50799 | 254.0 | 82.1% |
| Bombs Placed | 6192 | 31.0 | 10.0% |
| Waited (WAIT) | 4151 | 20.8 | 6.7% |
| Invalid Actions | 696 | 3.5 | 1.1% |
| Crates Destroyed | 5638 | 28.2 | - |
| **Total Actions** | **61838** | **309.2** | **100.0%** |
