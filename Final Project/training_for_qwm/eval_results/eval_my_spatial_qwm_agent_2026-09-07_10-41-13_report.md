# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_02-21-03_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_02-21-03_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 10:41:46

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **60.0%** | 46.7% | `+13.3%` |
| **Survival Rate** | **40.0%** | 40.0% | `+0.0%` |
| **Score / Round** | **3.80 ± 1.3** | 2.73 | `+1.07` |
| **Coins / Round** | **3.80 ± 1.3** | 1.73 | `+2.07` |
| **Coin Share** | **42.2%** | 19.3% | `+23.0%` |
| **Kills / Round** | **0.00 ± 0.0** | 0.20 | `-0.20` |
| **Kill / Death Ratio (KDR)** | **0.00** | 0.26 | `-0.26` |
| **Steps Survived** | **270.2 ± 108** | 200.3 | `+69.9` |
| **Suicides / Round** | **0.60** | 0.60 | `+0.00` |
| **Wait % of Actions** | **21.2%** | 0.4% | `+20.8%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **60.0%** | 40.0% | 3.80 | 3.80 | 0.00 | 3 | 0 | 270.2 | 21.2% |
| rule_based_agent_0 | 40.0% | 40.0% | 2.60 | 1.60 | 0.20 | 3 | 1 | 213.2 | 0.0% |
| rule_based_agent_1 | 60.0% | 40.0% | 3.60 | 2.60 | 0.20 | 3 | 2 | 214.4 | 0.7% |
| rule_based_agent_2 | 40.0% | 40.0% | 2.00 | 1.00 | 0.20 | 3 | 0 | 173.2 | 0.5% |
| *Rule-Based Avg* | *46.7%* | *40.0%* | *2.73* | *1.73* | *0.20* | *3.0* | *1.0* | *200.3* | *0.4%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 879 | 175.8 | 65.1% |
| Bombs Placed | 185 | 37.0 | 13.7% |
| Waited (WAIT) | 286 | 57.2 | 21.2% |
| Invalid Actions | 1 | 0.2 | 0.1% |
| Crates Destroyed | 150 | 30.0 | - |
| **Total Actions** | **1351** | **270.2** | **100.0%** |
