# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_02-21-03_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_02-21-03_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 10:32:28

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **10.0%** | 36.7% | `-26.7%` |
| **Survival Rate** | **40.0%** | 46.7% | `-6.7%` |
| **Score / Round** | **2.10 ± 1.6** | 2.93 | `-0.83` |
| **Coins / Round** | **1.60 ± 1.4** | 2.43 | `-0.83` |
| **Coin Share** | **18.0%** | 27.3% | `-9.4%` |
| **Kills / Round** | **0.10 ± 0.3** | 0.10 | `-0.00` |
| **Kill / Death Ratio (KDR)** | **0.17** | 0.28 | `-0.11` |
| **Steps Survived** | **255.7 ± 132** | 240.0 | `+15.7` |
| **Suicides / Round** | **0.60** | 0.50 | `+0.10` |
| **Wait % of Actions** | **21.6%** | 0.1% | `+21.5%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **10.0%** | 40.0% | 2.10 | 1.60 | 0.10 | 6 | 0 | 255.7 | 21.6% |
| rule_based_agent_0 | 40.0% | 30.0% | 2.10 | 2.10 | 0.00 | 7 | 3 | 176.5 | 0.2% |
| rule_based_agent_1 | 40.0% | 40.0% | 3.30 | 2.80 | 0.10 | 6 | 0 | 260.3 | 0.0% |
| rule_based_agent_2 | 30.0% | 70.0% | 3.40 | 2.40 | 0.20 | 2 | 1 | 283.3 | 0.1% |
| *Rule-Based Avg* | *36.7%* | *46.7%* | *2.93* | *2.43* | *0.10* | *5.0* | *1.3* | *240.0* | *0.1%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1638 | 163.8 | 64.1% |
| Bombs Placed | 355 | 35.5 | 13.9% |
| Waited (WAIT) | 553 | 55.3 | 21.6% |
| Invalid Actions | 11 | 1.1 | 0.4% |
| Crates Destroyed | 202 | 20.2 | - |
| **Total Actions** | **2557** | **255.7** | **100.0%** |
