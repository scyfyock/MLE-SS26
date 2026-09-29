# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_02-21-03_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_02-21-03_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 20
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 02:24:57

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **75.0%** | 11.7% | `+63.3%` |
| **Survival Rate** | **95.0%** | 38.3% | `+56.7%` |
| **Score / Round** | **5.15 ± 2.1** | 2.70 | `+2.45` |
| **Coins / Round** | **2.90 ± 1.5** | 2.03 | `+0.87` |
| **Coin Share** | **32.2%** | 22.6% | `+9.6%` |
| **Kills / Round** | **0.45 ± 0.5** | 0.13 | `+0.32` |
| **Kill / Death Ratio (KDR)** | **9.00** | 0.21 | `+8.79` |
| **Steps Survived** | **383.4 ± 72** | 254.1 | `+129.3` |
| **Suicides / Round** | **0.00** | 0.42 | `-0.42` |
| **Wait % of Actions** | **9.6%** | 0.0% | `+9.5%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **75.0%** | 95.0% | 5.15 | 2.90 | 0.45 | 0 | 1 | 383.4 | 9.6% |
| rule_based_agent_0 | 25.0% | 40.0% | 3.25 | 2.00 | 0.25 | 7 | 5 | 278.4 | 0.0% |
| rule_based_agent_1 | 5.0% | 40.0% | 2.15 | 1.90 | 0.05 | 8 | 6 | 254.9 | 0.0% |
| rule_based_agent_2 | 5.0% | 35.0% | 2.70 | 2.20 | 0.10 | 10 | 5 | 228.9 | 0.0% |
| *Rule-Based Avg* | *11.7%* | *38.3%* | *2.70* | *2.03* | *0.13* | *8.3* | *5.3* | *254.1* | *0.0%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 6345 | 317.2 | 82.7% |
| Bombs Placed | 526 | 26.3 | 6.9% |
| Waited (WAIT) | 733 | 36.6 | 9.6% |
| Invalid Actions | 65 | 3.2 | 0.8% |
| Crates Destroyed | 682 | 34.1 | - |
| **Total Actions** | **7669** | **383.4** | **100.0%** |
