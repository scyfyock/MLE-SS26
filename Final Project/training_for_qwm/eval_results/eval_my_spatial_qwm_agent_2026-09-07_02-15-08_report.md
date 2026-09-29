# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_01-37-50_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_01-37-50_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 20
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 02:16:47

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **50.0%** | 26.7% | `+23.3%` |
| **Survival Rate** | **80.0%** | 41.7% | `+38.3%` |
| **Score / Round** | **4.75 ± 2.5** | 2.75 | `+2.00` |
| **Coins / Round** | **3.25 ± 1.3** | 1.92 | `+1.33` |
| **Coin Share** | **36.1%** | 21.3% | `+14.8%` |
| **Kills / Round** | **0.30 ± 0.5** | 0.17 | `+0.13` |
| **Kill / Death Ratio (KDR)** | **1.50** | 0.28 | `+1.22` |
| **Steps Survived** | **349.2 ± 97** | 253.4 | `+95.8` |
| **Suicides / Round** | **0.05** | 0.43 | `-0.38` |
| **Wait % of Actions** | **9.3%** | 0.1% | `+9.3%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **50.0%** | 80.0% | 4.75 | 3.25 | 0.30 | 1 | 3 | 349.2 | 9.3% |
| rule_based_agent_0 | 15.0% | 25.0% | 2.20 | 1.95 | 0.05 | 13 | 4 | 219.1 | 0.2% |
| rule_based_agent_1 | 40.0% | 45.0% | 3.50 | 2.00 | 0.30 | 6 | 7 | 268.4 | 0.1% |
| rule_based_agent_2 | 25.0% | 55.0% | 2.55 | 1.80 | 0.15 | 7 | 2 | 272.6 | 0.0% |
| *Rule-Based Avg* | *26.7%* | *41.7%* | *2.75* | *1.92* | *0.17* | *8.7* | *4.3* | *253.4* | *0.1%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 5832 | 291.6 | 83.5% |
| Bombs Placed | 440 | 22.0 | 6.3% |
| Waited (WAIT) | 652 | 32.6 | 9.3% |
| Invalid Actions | 60 | 3.0 | 0.9% |
| Crates Destroyed | 681 | 34.0 | - |
| **Total Actions** | **6984** | **349.2** | **100.0%** |
