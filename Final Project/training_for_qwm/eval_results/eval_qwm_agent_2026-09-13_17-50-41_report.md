# BombeRLe Evaluation Report: `qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent`
- **Timestamp:** 2026-09-13 18:09:27

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **67.0%** | 45.0% | `+22.0%` |
| **Survival Rate** | **72.5%** | 59.5% | `+13.0%` |
| **Score / Round** | **5.03 ± 2.6** | 4.12 | `+0.91` |
| **Coins / Round** | **4.66 ± 2.0** | 3.87 | `+0.79` |
| **Coin Share** | **54.6%** | 45.4% | `+9.2%` |
| **Kills / Round** | **0.07 ± 0.3** | 0.05 | `+0.02` |
| **Kill / Death Ratio (KDR)** | **0.25** | 0.12 | `+0.13` |
| **Steps Survived** | **324.7 ± 125** | 296.2 | `+28.5` |
| **Suicides / Round** | **0.26** | 0.35 | `-0.09` |
| **Wait % of Actions** | **2.9%** | 0.3% | `+2.6%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent** | **67.0%** | 72.5% | 5.03 | 4.66 | 0.07 | 51 | 10 | 324.7 | 2.9% |
| rule_based_agent | 45.0% | 59.5% | 4.12 | 3.87 | 0.05 | 70 | 15 | 296.2 | 0.3% |
| *Rule-Based Avg* | *45.0%* | *59.5%* | *4.12* | *3.87* | *0.05* | *70.0* | *15.0* | *296.2* | *0.3%* |

## Action Profile for `qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 56153 | 280.8 | 86.5% |
| Bombs Placed | 6583 | 32.9 | 10.1% |
| Waited (WAIT) | 1889 | 9.4 | 2.9% |
| Invalid Actions | 314 | 1.6 | 0.5% |
| Crates Destroyed | 10033 | 50.2 | - |
| **Total Actions** | **64939** | **324.7** | **100.0%** |
