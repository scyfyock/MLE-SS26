# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-06_19-56-41_spatial_qwm/best_model_s0_wm_warmup.pt`
- **Source:** Run run_2026-09-06_19-56-41_spatial_qwm (best_model_s0_wm_warmup.pt)
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-06 19:57:02

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **0.0%** | 33.3% | `-33.3%` |
| **Survival Rate** | **0.0%** | 53.3% | `-53.3%` |
| **Score / Round** | **0.00 ± 0.0** | 3.93 | `-3.93` |
| **Coins / Round** | **0.00 ± 0.0** | 2.93 | `-2.93` |
| **Coin Share** | **0.0%** | 33.3% | `-33.3%` |
| **Kills / Round** | **0.00 ± 0.0** | 0.20 | `-0.20` |
| **Kill / Death Ratio (KDR)** | **0.00** | 0.78 | `-0.78` |
| **Steps Survived** | **5.2 ± 0** | 264.2 | `-259.0` |
| **Suicides / Round** | **1.00** | 0.33 | `+0.67` |
| **Wait % of Actions** | **61.5%** | 0.1% | `+61.4%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **0.0%** | 0.0% | 0.00 | 0.00 | 0.00 | 5 | 0 | 5.2 | 61.5% |
| rule_based_agent_0 | 20.0% | 40.0% | 1.60 | 1.60 | 0.00 | 1 | 3 | 224.6 | 0.0% |
| rule_based_agent_1 | 60.0% | 80.0% | 6.60 | 4.60 | 0.40 | 1 | 0 | 331.4 | 0.3% |
| rule_based_agent_2 | 20.0% | 40.0% | 3.60 | 2.60 | 0.20 | 3 | 0 | 236.6 | 0.0% |
| *Rule-Based Avg* | *33.3%* | *53.3%* | *3.93* | *2.93* | *0.20* | *1.7* | *1.0* | *264.2* | *0.1%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 5 | 1.0 | 19.2% |
| Bombs Placed | 5 | 1.0 | 19.2% |
| Waited (WAIT) | 16 | 3.2 | 61.5% |
| Invalid Actions | 0 | 0.0 | 0.0% |
| Crates Destroyed | 15 | 3.0 | - |
| **Total Actions** | **26** | **5.2** | **100.0%** |
