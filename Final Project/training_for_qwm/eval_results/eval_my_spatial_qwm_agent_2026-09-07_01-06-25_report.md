# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_00-50-35_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_00-50-35_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 01:06:48

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **30.0%** | 46.7% | `-16.7%` |
| **Survival Rate** | **0.0%** | 40.0% | `-40.0%` |
| **Score / Round** | **3.30 ± 2.1** | 2.90 | `+0.40` |
| **Coins / Round** | **2.30 ± 0.9** | 2.23 | `+0.07` |
| **Coin Share** | **25.6%** | 24.8% | `+0.7%` |
| **Kills / Round** | **0.20 ± 0.4** | 0.13 | `+0.07` |
| **Kill / Death Ratio (KDR)** | **0.20** | 0.22 | `-0.02` |
| **Steps Survived** | **136.0 ± 30** | 182.3 | `-46.3` |
| **Suicides / Round** | **1.00** | 0.53 | `+0.47` |
| **Wait % of Actions** | **7.3%** | 0.6% | `+6.7%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **30.0%** | 0.0% | 3.30 | 2.30 | 0.20 | 10 | 0 | 136.0 | 7.3% |
| rule_based_agent_0 | 40.0% | 40.0% | 2.80 | 1.80 | 0.20 | 6 | 0 | 174.1 | 0.5% |
| rule_based_agent_1 | 50.0% | 40.0% | 2.50 | 2.50 | 0.00 | 6 | 4 | 187.2 | 0.6% |
| rule_based_agent_2 | 50.0% | 40.0% | 3.40 | 2.40 | 0.20 | 4 | 2 | 185.5 | 0.6% |
| *Rule-Based Avg* | *46.7%* | *40.0%* | *2.90* | *2.23* | *0.13* | *5.3* | *2.0* | *182.3* | *0.6%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1076 | 107.6 | 79.1% |
| Bombs Placed | 163 | 16.3 | 12.0% |
| Waited (WAIT) | 99 | 9.9 | 7.3% |
| Invalid Actions | 22 | 2.2 | 1.6% |
| Crates Destroyed | 237 | 23.7 | - |
| **Total Actions** | **1360** | **136.0** | **100.0%** |
