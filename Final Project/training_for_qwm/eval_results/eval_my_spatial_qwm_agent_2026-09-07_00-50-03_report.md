# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-06_23-46-34_spatial_qwm/best_model_s2_classic_alone.pt`
- **Source:** Run run_2026-09-06_23-46-34_spatial_qwm (best_model_s2_classic_alone.pt)
- **Scenario:** `classic` | **Rounds:** 2
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 00:50:09

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **0.0%** | 50.0% | `-50.0%` |
| **Survival Rate** | **0.0%** | 50.0% | `-50.0%` |
| **Score / Round** | **2.00 ± 2.0** | 4.00 | `-2.00` |
| **Coins / Round** | **2.00 ± 2.0** | 2.33 | `-0.33` |
| **Coin Share** | **22.2%** | 25.9% | `-3.7%` |
| **Kills / Round** | **0.00 ± 0.0** | 0.33 | `-0.33` |
| **Kill / Death Ratio (KDR)** | **0.00** | 0.67 | `-0.67` |
| **Steps Survived** | **80.5 ± 30** | 248.5 | `-168.0` |
| **Suicides / Round** | **0.50** | 0.50 | `+0.00` |
| **Wait % of Actions** | **14.3%** | 0.0% | `+14.2%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **0.0%** | 0.0% | 2.00 | 2.00 | 0.00 | 1 | 2 | 80.5 | 14.3% |
| rule_based_agent_0 | 50.0% | 100.0% | 4.00 | 4.00 | 0.00 | 0 | 0 | 345.0 | 0.1% |
| rule_based_agent_1 | 0.0% | 0.0% | 1.50 | 1.50 | 0.00 | 2 | 0 | 113.0 | 0.0% |
| rule_based_agent_2 | 100.0% | 50.0% | 6.50 | 1.50 | 1.00 | 1 | 0 | 287.5 | 0.0% |
| *Rule-Based Avg* | *50.0%* | *50.0%* | *4.00* | *2.33* | *0.33* | *1.0* | *0.0* | *248.5* | *0.0%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 120 | 60.0 | 74.5% |
| Bombs Placed | 17 | 8.5 | 10.6% |
| Waited (WAIT) | 23 | 11.5 | 14.3% |
| Invalid Actions | 1 | 0.5 | 0.6% |
| Crates Destroyed | 36 | 18.0 | - |
| **Total Actions** | **161** | **80.5** | **100.0%** |
