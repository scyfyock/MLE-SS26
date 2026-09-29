# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-21 15:08:08

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **39.0%** | 29.7% | `+9.3%` |
| **Survival Rate** | **71.0%** | 38.5% | `+32.5%` |
| **Score / Round** | **3.69 ± 2.8** | 2.98 | `+0.70` |
| **Coins / Round** | **2.63 ± 1.5** | 2.11 | `+0.52` |
| **Coin Share** | **29.3%** | 23.6% | `+5.8%` |
| **Kills / Round** | **0.21 ± 0.4** | 0.17 | `+0.04` |
| **Kill / Death Ratio (KDR)** | **0.64** | 0.25 | `+0.39` |
| **Steps Survived** | **325.2 ± 122** | 241.3 | `+83.9` |
| **Suicides / Round** | **0.17** | 0.51 | `-0.34` |
| **Wait % of Actions** | **6.6%** | 0.3% | `+6.3%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **39.0%** | 71.0% | 3.69 | 2.63 | 0.21 | 34 | 32 | 325.2 | 6.6% |
| rule_based_agent_0 | 30.0% | 38.5% | 2.87 | 2.09 | 0.15 | 104 | 47 | 244.9 | 0.3% |
| rule_based_agent_1 | 25.5% | 34.5% | 2.83 | 2.08 | 0.15 | 107 | 36 | 225.1 | 0.2% |
| rule_based_agent_2 | 33.5% | 42.5% | 3.25 | 2.17 | 0.21 | 97 | 31 | 253.8 | 0.3% |
| *Rule-Based Avg* | *29.7%* | *38.5%* | *2.98* | *2.11* | *0.17* | *102.7* | *38.0* | *241.3* | *0.3%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 53617 | 268.1 | 82.4% |
| Bombs Placed | 6390 | 31.9 | 9.8% |
| Waited (WAIT) | 4299 | 21.5 | 6.6% |
| Invalid Actions | 734 | 3.7 | 1.1% |
| Crates Destroyed | 5930 | 29.6 | - |
| **Total Actions** | **65040** | **325.2** | **100.0%** |
