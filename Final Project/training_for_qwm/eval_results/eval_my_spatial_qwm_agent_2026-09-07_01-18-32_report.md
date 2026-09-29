# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_00-50-35_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_00-50-35_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 01:19:02

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **40.0%** | 40.0% | `+0.0%` |
| **Survival Rate** | **40.0%** | 40.0% | `+0.0%` |
| **Score / Round** | **4.10 ± 4.2** | 2.97 | `+1.13` |
| **Coins / Round** | **2.60 ± 1.4** | 2.13 | `+0.47` |
| **Coin Share** | **28.9%** | 23.7% | `+5.2%` |
| **Kills / Round** | **0.30 ± 0.6** | 0.17 | `+0.13` |
| **Kill / Death Ratio (KDR)** | **0.50** | 0.24 | `+0.26` |
| **Steps Survived** | **249.4 ± 129** | 250.3 | `-0.9` |
| **Suicides / Round** | **0.60** | 0.47 | `+0.13` |
| **Wait % of Actions** | **8.3%** | 0.2% | `+8.0%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **40.0%** | 40.0% | 4.10 | 2.60 | 0.30 | 6 | 0 | 249.4 | 8.3% |
| rule_based_agent_0 | 40.0% | 10.0% | 3.20 | 2.20 | 0.20 | 7 | 4 | 191.2 | 0.1% |
| rule_based_agent_1 | 40.0% | 50.0% | 2.80 | 1.80 | 0.20 | 4 | 3 | 280.9 | 0.2% |
| rule_based_agent_2 | 40.0% | 60.0% | 2.90 | 2.40 | 0.10 | 3 | 1 | 278.8 | 0.5% |
| *Rule-Based Avg* | *40.0%* | *40.0%* | *2.97* | *2.13* | *0.17* | *4.7* | *2.7* | *250.3* | *0.2%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 2023 | 202.3 | 81.1% |
| Bombs Placed | 242 | 24.2 | 9.7% |
| Waited (WAIT) | 206 | 20.6 | 8.3% |
| Invalid Actions | 23 | 2.3 | 0.9% |
| Crates Destroyed | 326 | 32.6 | - |
| **Total Actions** | **2494** | **249.4** | **100.0%** |
