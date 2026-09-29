# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/best_model_s4_vs_rule_based.pt`
- **Source:** Explicit file: agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/best_model_s4_vs_rule_based.pt
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 00:47:08

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **0.0%** | 33.3% | `-33.3%` |
| **Survival Rate** | **0.0%** | 40.0% | `-40.0%` |
| **Score / Round** | **0.00 ± 0.0** | 3.67 | `-3.67` |
| **Coins / Round** | **0.00 ± 0.0** | 3.00 | `-3.00` |
| **Coin Share** | **0.0%** | 33.3% | `-33.3%` |
| **Kills / Round** | **0.00 ± 0.0** | 0.13 | `-0.13` |
| **Kill / Death Ratio (KDR)** | **0.00** | 0.42 | `-0.42` |
| **Steps Survived** | **5.2 ± 0** | 204.5 | `-199.3` |
| **Suicides / Round** | **1.00** | 0.53 | `+0.47` |
| **Wait % of Actions** | **50.0%** | 0.4% | `+49.6%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **0.0%** | 0.0% | 0.00 | 0.00 | 0.00 | 5 | 0 | 5.2 | 50.0% |
| rule_based_agent_0 | 80.0% | 80.0% | 5.80 | 4.80 | 0.20 | 0 | 1 | 228.8 | 1.1% |
| rule_based_agent_1 | 0.0% | 20.0% | 2.80 | 2.80 | 0.00 | 4 | 1 | 173.4 | 0.0% |
| rule_based_agent_2 | 20.0% | 20.0% | 2.40 | 1.40 | 0.20 | 4 | 0 | 211.4 | 0.0% |
| *Rule-Based Avg* | *33.3%* | *40.0%* | *3.67* | *3.00* | *0.13* | *2.7* | *0.7* | *204.5* | *0.4%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 8 | 1.6 | 30.8% |
| Bombs Placed | 5 | 1.0 | 19.2% |
| Waited (WAIT) | 13 | 2.6 | 50.0% |
| Invalid Actions | 0 | 0.0 | 0.0% |
| Crates Destroyed | 15 | 3.0 | - |
| **Total Actions** | **26** | **5.2** | **100.0%** |
