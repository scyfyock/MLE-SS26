# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_02-21-03_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_02-21-03_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 10:50:56

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **0.0%** | 46.7% | `-46.7%` |
| **Survival Rate** | **20.0%** | 40.0% | `-20.0%` |
| **Score / Round** | **1.20 ± 0.7** | 4.27 | `-3.07` |
| **Coins / Round** | **1.20 ± 0.7** | 2.60 | `-1.40` |
| **Coin Share** | **13.3%** | 28.9% | `-15.6%` |
| **Kills / Round** | **0.00 ± 0.0** | 0.33 | `-0.33` |
| **Kill / Death Ratio (KDR)** | **0.00** | 0.47 | `-0.47` |
| **Steps Survived** | **208.4 ± 123** | 217.5 | `-9.1` |
| **Suicides / Round** | **0.60** | 0.40 | `+0.20` |
| **Wait % of Actions** | **17.4%** | 0.4% | `+17.0%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **0.0%** | 20.0% | 1.20 | 1.20 | 0.00 | 3 | 1 | 208.4 | 17.4% |
| rule_based_agent_0 | 20.0% | 40.0% | 2.60 | 1.60 | 0.20 | 1 | 2 | 217.4 | 0.4% |
| rule_based_agent_1 | 40.0% | 60.0% | 2.80 | 1.80 | 0.20 | 1 | 2 | 243.8 | 0.3% |
| rule_based_agent_2 | 80.0% | 20.0% | 7.40 | 4.40 | 0.60 | 4 | 0 | 191.2 | 0.4% |
| *Rule-Based Avg* | *46.7%* | *40.0%* | *4.27* | *2.60* | *0.33* | *2.0* | *1.3* | *217.5* | *0.4%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 711 | 142.2 | 68.2% |
| Bombs Placed | 147 | 29.4 | 14.1% |
| Waited (WAIT) | 181 | 36.2 | 17.4% |
| Invalid Actions | 4 | 0.8 | 0.4% |
| Crates Destroyed | 69 | 13.8 | - |
| **Total Actions** | **1043** | **208.6** | **100.0%** |
