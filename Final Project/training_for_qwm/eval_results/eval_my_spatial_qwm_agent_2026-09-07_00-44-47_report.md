# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-06_23-17-44_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-06_23-17-44_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 00:44:54

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **0.0%** | 53.3% | `-53.3%` |
| **Survival Rate** | **0.0%** | 46.7% | `-46.7%` |
| **Score / Round** | **0.00 ± 0.0** | 3.00 | `-3.00` |
| **Coins / Round** | **0.00 ± 0.0** | 3.00 | `-3.00` |
| **Coin Share** | **0.0%** | 33.3% | `-33.3%` |
| **Kills / Round** | **0.00 ± 0.0** | 0.00 | `+0.00` |
| **Kill / Death Ratio (KDR)** | **0.00** | 0.00 | `+0.00` |
| **Steps Survived** | **12.4 ± 12** | 234.2 | `-221.8` |
| **Suicides / Round** | **1.00** | 0.53 | `+0.47` |
| **Wait % of Actions** | **12.9%** | 0.4% | `+12.5%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **0.0%** | 0.0% | 0.00 | 0.00 | 0.00 | 5 | 0 | 12.4 | 12.9% |
| rule_based_agent_0 | 80.0% | 60.0% | 3.00 | 3.00 | 0.00 | 2 | 0 | 269.2 | 0.3% |
| rule_based_agent_1 | 40.0% | 60.0% | 2.80 | 2.80 | 0.00 | 2 | 0 | 259.4 | 0.3% |
| rule_based_agent_2 | 40.0% | 20.0% | 3.20 | 3.20 | 0.00 | 4 | 0 | 174.0 | 0.5% |
| *Rule-Based Avg* | *53.3%* | *46.7%* | *3.00* | *3.00* | *0.00* | *2.7* | *0.0* | *234.2* | *0.4%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 45 | 9.0 | 72.6% |
| Bombs Placed | 9 | 1.8 | 14.5% |
| Waited (WAIT) | 8 | 1.6 | 12.9% |
| Invalid Actions | 0 | 0.0 | 0.0% |
| Crates Destroyed | 18 | 3.6 | - |
| **Total Actions** | **62** | **12.4** | **100.0%** |
