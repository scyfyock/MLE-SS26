# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-21 14:42:40

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **37.5%** | 29.0% | `+8.5%` |
| **Survival Rate** | **65.0%** | 38.7% | `+26.3%` |
| **Score / Round** | **3.69 ± 2.7** | 2.97 | `+0.72` |
| **Coins / Round** | **2.62 ± 1.6** | 2.12 | `+0.49` |
| **Coin Share** | **29.1%** | 23.6% | `+5.5%` |
| **Kills / Round** | **0.21 ± 0.4** | 0.17 | `+0.04` |
| **Kill / Death Ratio (KDR)** | **0.56** | 0.25 | `+0.31` |
| **Steps Survived** | **312.5 ± 128** | 242.2 | `+70.3` |
| **Suicides / Round** | **0.22** | 0.50 | `-0.28` |
| **Wait % of Actions** | **7.2%** | 0.3% | `+6.8%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **37.5%** | 65.0% | 3.69 | 2.62 | 0.21 | 44 | 33 | 312.5 | 7.2% |
| rule_based_agent_0 | 28.0% | 38.0% | 2.89 | 2.14 | 0.15 | 98 | 41 | 247.6 | 0.3% |
| rule_based_agent_1 | 28.5% | 36.0% | 3.08 | 2.13 | 0.19 | 111 | 37 | 236.1 | 0.3% |
| rule_based_agent_2 | 30.5% | 42.0% | 2.95 | 2.10 | 0.17 | 92 | 34 | 243.0 | 0.3% |
| *Rule-Based Avg* | *29.0%* | *38.7%* | *2.97* | *2.12* | *0.17* | *100.3* | *37.3* | *242.2* | *0.3%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 51125 | 255.6 | 81.8% |
| Bombs Placed | 6251 | 31.3 | 10.0% |
| Waited (WAIT) | 4470 | 22.4 | 7.2% |
| Invalid Actions | 653 | 3.3 | 1.0% |
| Crates Destroyed | 5669 | 28.3 | - |
| **Total Actions** | **62499** | **312.5** | **100.0%** |
