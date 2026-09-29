# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_00-50-35_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Explicit file: agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_00-50-35_spatial_qwm/best_model_s4_vs_rule_based.pt
- **Scenario:** `classic` | **Rounds:** 20
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 02:19:13

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **45.0%** | 25.0% | `+20.0%` |
| **Survival Rate** | **80.0%** | 38.3% | `+41.7%` |
| **Score / Round** | **4.65 ± 2.9** | 2.85 | `+1.80` |
| **Coins / Round** | **2.90 ± 1.3** | 2.02 | `+0.88` |
| **Coin Share** | **32.4%** | 22.5% | `+9.9%` |
| **Kills / Round** | **0.35 ± 0.5** | 0.17 | `+0.18` |
| **Kill / Death Ratio (KDR)** | **1.75** | 0.26 | `+1.49` |
| **Steps Survived** | **349.4 ± 97** | 250.6 | `+98.8` |
| **Suicides / Round** | **0.10** | 0.43 | `-0.33` |
| **Wait % of Actions** | **9.8%** | 0.0% | `+9.8%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **45.0%** | 80.0% | 4.65 | 2.90 | 0.35 | 2 | 2 | 349.4 | 9.8% |
| rule_based_agent_0 | 25.0% | 25.0% | 2.75 | 2.00 | 0.15 | 11 | 6 | 214.3 | 0.0% |
| rule_based_agent_1 | 30.0% | 50.0% | 3.10 | 1.85 | 0.25 | 7 | 4 | 280.4 | 0.1% |
| rule_based_agent_2 | 20.0% | 40.0% | 2.70 | 2.20 | 0.10 | 8 | 5 | 257.1 | 0.0% |
| *Rule-Based Avg* | *25.0%* | *38.3%* | *2.85* | *2.02* | *0.17* | *8.7* | *5.0* | *250.6* | *0.0%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 5738 | 286.9 | 82.1% |
| Bombs Placed | 493 | 24.6 | 7.1% |
| Waited (WAIT) | 687 | 34.4 | 9.8% |
| Invalid Actions | 71 | 3.5 | 1.0% |
| Crates Destroyed | 666 | 33.3 | - |
| **Total Actions** | **6989** | **349.4** | **100.0%** |
