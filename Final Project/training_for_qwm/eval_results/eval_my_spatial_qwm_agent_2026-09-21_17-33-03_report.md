# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-11_23-46-21_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-11_23-46-21_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-21 17:50:59

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **41.0%** | 29.7% | `+11.3%` |
| **Survival Rate** | **62.0%** | 36.8% | `+25.2%` |
| **Score / Round** | **3.69 ± 2.8** | 2.90 | `+0.79` |
| **Coins / Round** | **2.62 ± 1.5** | 2.12 | `+0.50` |
| **Coin Share** | **29.1%** | 23.6% | `+5.5%` |
| **Kills / Round** | **0.21 ± 0.5** | 0.15 | `+0.06` |
| **Kill / Death Ratio (KDR)** | **0.49** | 0.23 | `+0.27` |
| **Steps Survived** | **299.9 ± 126** | 234.7 | `+65.2` |
| **Suicides / Round** | **0.28** | 0.51 | `-0.23` |
| **Wait % of Actions** | **1.3%** | 0.2% | `+1.2%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **41.0%** | 62.0% | 3.69 | 2.62 | 0.21 | 56 | 31 | 299.9 | 1.3% |
| rule_based_agent_0 | 29.5% | 40.5% | 2.88 | 2.15 | 0.14 | 96 | 33 | 240.9 | 0.2% |
| rule_based_agent_1 | 32.5% | 35.5% | 3.04 | 2.19 | 0.17 | 102 | 40 | 233.4 | 0.2% |
| rule_based_agent_2 | 27.0% | 34.5% | 2.77 | 2.02 | 0.15 | 108 | 32 | 229.7 | 0.1% |
| *Rule-Based Avg* | *29.7%* | *36.8%* | *2.90* | *2.12* | *0.15* | *102.0* | *35.0* | *234.7* | *0.2%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 53445 | 267.2 | 89.1% |
| Bombs Placed | 5102 | 25.5 | 8.5% |
| Waited (WAIT) | 796 | 4.0 | 1.3% |
| Invalid Actions | 635 | 3.2 | 1.1% |
| Crates Destroyed | 5351 | 26.8 | - |
| **Total Actions** | **59978** | **299.9** | **100.0%** |
