# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-14 02:01:01

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **36.0%** | 31.0% | `+5.0%` |
| **Survival Rate** | **54.5%** | 41.2% | `+13.3%` |
| **Score / Round** | **3.34 ± 2.8** | 3.18 | `+0.16` |
| **Coins / Round** | **2.24 ± 1.4** | 2.24 | `+0.00` |
| **Coin Share** | **25.0%** | 25.0% | `+0.0%` |
| **Kills / Round** | **0.22 ± 0.4** | 0.19 | `+0.03` |
| **Kill / Death Ratio (KDR)** | **0.44** | 0.28 | `+0.16` |
| **Steps Survived** | **288.5 ± 136** | 253.3 | `+35.1` |
| **Suicides / Round** | **0.29** | 0.48 | `-0.18` |
| **Wait % of Actions** | **5.1%** | 0.2% | `+4.9%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **36.0%** | 54.5% | 3.34 | 2.24 | 0.22 | 59 | 41 | 288.5 | 5.1% |
| rule_based_agent_0 | 27.5% | 37.0% | 3.17 | 2.29 | 0.17 | 97 | 47 | 248.1 | 0.2% |
| rule_based_agent_1 | 35.0% | 42.0% | 3.35 | 2.19 | 0.23 | 96 | 32 | 253.5 | 0.2% |
| rule_based_agent_2 | 30.5% | 44.5% | 3.02 | 2.23 | 0.16 | 93 | 37 | 258.4 | 0.2% |
| *Rule-Based Avg* | *31.0%* | *41.2%* | *3.18* | *2.24* | *0.19* | *95.3* | *38.7* | *253.3* | *0.2%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 48964 | 244.8 | 84.9% |
| Bombs Placed | 5283 | 26.4 | 9.2% |
| Waited (WAIT) | 2950 | 14.8 | 5.1% |
| Invalid Actions | 494 | 2.5 | 0.9% |
| Crates Destroyed | 4892 | 24.5 | - |
| **Total Actions** | **57691** | **288.5** | **100.0%** |
