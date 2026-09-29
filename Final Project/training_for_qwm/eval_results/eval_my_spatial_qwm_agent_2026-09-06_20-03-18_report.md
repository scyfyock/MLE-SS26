# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-06_20-02-08_spatial_qwm/best_model_s0_wm_warmup.pt`
- **Source:** Run run_2026-09-06_20-02-08_spatial_qwm (best_model_s0_wm_warmup.pt)
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-06 20:03:44

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **30.0%** | 46.7% | `-16.7%` |
| **Survival Rate** | **0.0%** | 43.3% | `-43.3%` |
| **Score / Round** | **4.10 ± 3.0** | 3.13 | `+0.97` |
| **Coins / Round** | **2.60 ± 1.6** | 2.13 | `+0.47` |
| **Coin Share** | **28.9%** | 23.7% | `+5.2%` |
| **Kills / Round** | **0.30 ± 0.5** | 0.20 | `+0.10` |
| **Kill / Death Ratio (KDR)** | **0.25** | 0.38 | `-0.13` |
| **Steps Survived** | **130.5 ± 47** | 217.2 | `-86.7` |
| **Suicides / Round** | **1.00** | 0.33 | `+0.67` |
| **Wait % of Actions** | **8.5%** | 0.4% | `+8.1%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **30.0%** | 0.0% | 4.10 | 2.60 | 0.30 | 10 | 2 | 130.5 | 8.5% |
| rule_based_agent_0 | 50.0% | 60.0% | 2.60 | 1.60 | 0.20 | 3 | 1 | 225.5 | 0.8% |
| rule_based_agent_1 | 30.0% | 20.0% | 3.30 | 2.30 | 0.20 | 5 | 3 | 211.6 | 0.1% |
| rule_based_agent_2 | 60.0% | 50.0% | 3.50 | 2.50 | 0.20 | 2 | 3 | 214.5 | 0.4% |
| *Rule-Based Avg* | *46.7%* | *43.3%* | *3.13* | *2.13* | *0.20* | *3.3* | *2.3* | *217.2* | *0.4%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1039 | 103.9 | 79.6% |
| Bombs Placed | 145 | 14.5 | 11.1% |
| Waited (WAIT) | 111 | 11.1 | 8.5% |
| Invalid Actions | 10 | 1.0 | 0.8% |
| Crates Destroyed | 289 | 28.9 | - |
| **Total Actions** | **1305** | **130.5** | **100.0%** |
