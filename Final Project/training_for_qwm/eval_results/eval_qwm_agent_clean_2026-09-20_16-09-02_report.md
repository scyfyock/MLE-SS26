# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 100
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-20 16:20:08

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **53.0%** | 26.0% | `+27.0%` |
| **Survival Rate** | **76.0%** | 34.7% | `+41.3%` |
| **Score / Round** | **4.42 ± 3.3** | 2.83 | `+1.59` |
| **Coins / Round** | **2.67 ± 1.6** | 2.10 | `+0.57` |
| **Coin Share** | **29.8%** | 23.4% | `+6.4%` |
| **Kills / Round** | **0.35 ± 0.5** | 0.15 | `+0.20` |
| **Kill / Death Ratio (KDR)** | **1.35** | 0.21 | `+1.14` |
| **Steps Survived** | **323.0 ± 119** | 235.6 | `+87.4` |
| **Suicides / Round** | **0.16** | 0.51 | `-0.35` |
| **Wait % of Actions** | **1.8%** | 0.2% | `+1.6%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **53.0%** | 76.0% | 4.42 | 2.67 | 0.35 | 16 | 10 | 323.0 | 1.8% |
| rule_based_agent_0 | 26.0% | 29.0% | 2.62 | 1.92 | 0.14 | 53 | 28 | 224.7 | 0.2% |
| rule_based_agent_1 | 30.0% | 44.0% | 3.03 | 2.08 | 0.19 | 43 | 17 | 256.1 | 0.2% |
| rule_based_agent_2 | 22.0% | 31.0% | 2.85 | 2.30 | 0.11 | 56 | 24 | 225.9 | 0.1% |
| *Rule-Based Avg* | *26.0%* | *34.7%* | *2.83* | *2.10* | *0.15* | *50.7* | *23.0* | *235.6* | *0.2%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 28299 | 283.0 | 87.6% |
| Bombs Placed | 3063 | 30.6 | 9.5% |
| Waited (WAIT) | 566 | 5.7 | 1.8% |
| Invalid Actions | 372 | 3.7 | 1.2% |
| Crates Destroyed | 2851 | 28.5 | - |
| **Total Actions** | **32300** | **323.0** | **100.0%** |
