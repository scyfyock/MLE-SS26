# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_00-50-35_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_00-50-35_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 01:09:27

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **30.0%** | 40.0% | `-10.0%` |
| **Survival Rate** | **0.0%** | 43.3% | `-43.3%` |
| **Score / Round** | **3.20 ± 3.2** | 3.60 | `-0.40` |
| **Coins / Round** | **2.20 ± 1.4** | 2.27 | `-0.07` |
| **Coin Share** | **24.4%** | 25.2% | `-0.7%` |
| **Kills / Round** | **0.20 ± 0.6** | 0.27 | `-0.07` |
| **Kill / Death Ratio (KDR)** | **0.20** | 0.43 | `-0.23` |
| **Steps Survived** | **111.8 ± 39** | 203.8 | `-92.0` |
| **Suicides / Round** | **0.80** | 0.43 | `+0.37` |
| **Wait % of Actions** | **5.5%** | 0.5% | `+5.1%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **30.0%** | 0.0% | 3.20 | 2.20 | 0.20 | 8 | 2 | 111.8 | 5.5% |
| rule_based_agent_0 | 40.0% | 40.0% | 2.80 | 2.30 | 0.10 | 6 | 3 | 184.9 | 0.6% |
| rule_based_agent_1 | 50.0% | 50.0% | 3.10 | 1.60 | 0.30 | 2 | 3 | 213.3 | 0.6% |
| rule_based_agent_2 | 30.0% | 40.0% | 4.90 | 2.90 | 0.40 | 5 | 2 | 213.2 | 0.2% |
| *Rule-Based Avg* | *40.0%* | *43.3%* | *3.60* | *2.27* | *0.27* | *4.3* | *2.7* | *203.8* | *0.5%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 913 | 91.3 | 81.7% |
| Bombs Placed | 124 | 12.4 | 11.1% |
| Waited (WAIT) | 62 | 6.2 | 5.5% |
| Invalid Actions | 19 | 1.9 | 1.7% |
| Crates Destroyed | 229 | 22.9 | - |
| **Total Actions** | **1118** | **111.8** | **100.0%** |
