# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_00-50-35_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_00-50-35_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 01:16:16

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **50.0%** | 30.0% | `+20.0%` |
| **Survival Rate** | **10.0%** | 36.7% | `-26.7%` |
| **Score / Round** | **4.50 ± 3.6** | 2.50 | `+2.00` |
| **Coins / Round** | **2.50 ± 1.0** | 2.17 | `+0.33` |
| **Coin Share** | **27.8%** | 24.1% | `+3.7%` |
| **Kills / Round** | **0.40 ± 0.7** | 0.07 | `+0.33` |
| **Kill / Death Ratio (KDR)** | **0.44** | 0.10 | `+0.34` |
| **Steps Survived** | **201.5 ± 83** | 216.6 | `-15.1` |
| **Suicides / Round** | **0.80** | 0.47 | `+0.33` |
| **Wait % of Actions** | **3.9%** | 0.6% | `+3.3%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **50.0%** | 10.0% | 4.50 | 2.50 | 0.40 | 8 | 1 | 201.5 | 3.9% |
| rule_based_agent_0 | 30.0% | 40.0% | 2.20 | 2.20 | 0.00 | 3 | 3 | 212.2 | 0.7% |
| rule_based_agent_1 | 30.0% | 40.0% | 2.20 | 1.70 | 0.10 | 6 | 0 | 200.2 | 0.5% |
| rule_based_agent_2 | 30.0% | 30.0% | 3.10 | 2.60 | 0.10 | 5 | 2 | 237.4 | 0.5% |
| *Rule-Based Avg* | *30.0%* | *36.7%* | *2.50* | *2.17* | *0.07* | *4.7* | *1.7* | *216.6* | *0.6%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1728 | 172.8 | 85.8% |
| Bombs Placed | 195 | 19.5 | 9.7% |
| Waited (WAIT) | 78 | 7.8 | 3.9% |
| Invalid Actions | 14 | 1.4 | 0.7% |
| Crates Destroyed | 333 | 33.3 | - |
| **Total Actions** | **2015** | **201.5** | **100.0%** |
