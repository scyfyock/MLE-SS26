# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-06_19-56-41_spatial_qwm/best_model_s0_wm_warmup.pt`
- **Source:** Run run_2026-09-06_19-56-41_spatial_qwm (best_model_s0_wm_warmup.pt)
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-06 19:57:58

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **0.0%** | 33.3% | `-33.3%` |
| **Survival Rate** | **0.0%** | 40.0% | `-40.0%` |
| **Score / Round** | **0.00 ± 0.0** | 3.27 | `-3.27` |
| **Coins / Round** | **0.00 ± 0.0** | 2.93 | `-2.93` |
| **Coin Share** | **0.0%** | 33.3% | `-33.3%` |
| **Kills / Round** | **0.00 ± 0.0** | 0.07 | `-0.07` |
| **Kill / Death Ratio (KDR)** | **0.00** | 0.17 | `-0.17` |
| **Steps Survived** | **5.2 ± 0** | 187.0 | `-181.8` |
| **Suicides / Round** | **1.00** | 0.53 | `+0.47` |
| **Wait % of Actions** | **50.0%** | 0.4% | `+49.6%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **0.0%** | 0.0% | 0.00 | 0.00 | 0.00 | 5 | 0 | 5.2 | 50.0% |
| rule_based_agent_0 | 0.0% | 0.0% | 2.00 | 2.00 | 0.00 | 4 | 1 | 122.4 | 0.0% |
| rule_based_agent_1 | 60.0% | 60.0% | 4.20 | 4.20 | 0.00 | 2 | 0 | 231.4 | 0.3% |
| rule_based_agent_2 | 40.0% | 60.0% | 3.60 | 2.60 | 0.20 | 2 | 0 | 207.2 | 0.8% |
| *Rule-Based Avg* | *33.3%* | *40.0%* | *3.27* | *2.93* | *0.07* | *2.7* | *0.3* | *187.0* | *0.4%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 8 | 1.6 | 30.8% |
| Bombs Placed | 5 | 1.0 | 19.2% |
| Waited (WAIT) | 13 | 2.6 | 50.0% |
| Invalid Actions | 0 | 0.0 | 0.0% |
| Crates Destroyed | 18 | 3.6 | - |
| **Total Actions** | **26** | **5.2** | **100.0%** |
