# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_02-21-03_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_02-21-03_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 10:46:31

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **40.0%** | 40.0% | `+0.0%` |
| **Survival Rate** | **20.0%** | 40.0% | `-20.0%` |
| **Score / Round** | **2.60 ± 1.5** | 3.40 | `-0.80` |
| **Coins / Round** | **1.60 ± 1.2** | 2.40 | `-0.80` |
| **Coin Share** | **18.2%** | 27.3% | `-9.1%` |
| **Kills / Round** | **0.20 ± 0.4** | 0.20 | `-0.00` |
| **Kill / Death Ratio (KDR)** | **0.25** | 0.28 | `-0.03` |
| **Steps Survived** | **211.4 ± 112** | 211.5 | `-0.1` |
| **Suicides / Round** | **0.60** | 0.40 | `+0.20` |
| **Wait % of Actions** | **19.5%** | 0.4% | `+19.1%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **40.0%** | 20.0% | 2.60 | 1.60 | 0.20 | 3 | 1 | 211.4 | 19.5% |
| rule_based_agent_0 | 60.0% | 40.0% | 3.40 | 2.40 | 0.20 | 2 | 1 | 216.8 | 0.4% |
| rule_based_agent_1 | 20.0% | 20.0% | 5.00 | 3.00 | 0.40 | 3 | 1 | 206.8 | 0.0% |
| rule_based_agent_2 | 40.0% | 60.0% | 1.80 | 1.80 | 0.00 | 1 | 1 | 210.8 | 0.8% |
| *Rule-Based Avg* | *40.0%* | *40.0%* | *3.40* | *2.40* | *0.20* | *2.0* | *1.0* | *211.5* | *0.4%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 700 | 140.0 | 66.2% |
| Bombs Placed | 147 | 29.4 | 13.9% |
| Waited (WAIT) | 206 | 41.2 | 19.5% |
| Invalid Actions | 5 | 1.0 | 0.5% |
| Crates Destroyed | 95 | 19.0 | - |
| **Total Actions** | **1058** | **211.6** | **100.0%** |
