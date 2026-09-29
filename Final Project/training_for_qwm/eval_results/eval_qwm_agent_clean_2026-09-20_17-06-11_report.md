# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 100
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-20 17:14:51

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **44.0%** | 27.3% | `+16.7%` |
| **Survival Rate** | **60.0%** | 36.3% | `+23.7%` |
| **Score / Round** | **4.20 ± 3.2** | 3.14 | `+1.06` |
| **Coins / Round** | **2.55 ± 1.4** | 2.14 | `+0.41` |
| **Coin Share** | **28.4%** | 23.9% | `+4.5%` |
| **Kills / Round** | **0.33 ± 0.6** | 0.20 | `+0.13` |
| **Kill / Death Ratio (KDR)** | **0.75** | 0.27 | `+0.48` |
| **Steps Survived** | **295.8 ± 133** | 241.9 | `+53.9` |
| **Suicides / Round** | **0.21** | 0.50 | `-0.29` |
| **Wait % of Actions** | **8.0%** | 0.3% | `+7.7%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **44.0%** | 60.0% | 4.20 | 2.55 | 0.33 | 21 | 23 | 295.8 | 8.0% |
| rule_based_agent_0 | 27.0% | 38.0% | 3.01 | 2.31 | 0.14 | 50 | 25 | 244.4 | 0.3% |
| rule_based_agent_1 | 29.0% | 35.0% | 3.35 | 2.00 | 0.27 | 47 | 24 | 237.8 | 0.4% |
| rule_based_agent_2 | 26.0% | 36.0% | 3.07 | 2.12 | 0.19 | 53 | 21 | 243.6 | 0.3% |
| *Rule-Based Avg* | *27.3%* | *36.3%* | *3.14* | *2.14* | *0.20* | *50.0* | *23.3* | *241.9* | *0.3%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 23899 | 239.0 | 80.8% |
| Bombs Placed | 3006 | 30.1 | 10.2% |
| Waited (WAIT) | 2371 | 23.7 | 8.0% |
| Invalid Actions | 306 | 3.1 | 1.0% |
| Crates Destroyed | 2795 | 27.9 | - |
| **Total Actions** | **29582** | **295.8** | **100.0%** |
