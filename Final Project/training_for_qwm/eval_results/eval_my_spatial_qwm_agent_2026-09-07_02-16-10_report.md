# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_01-37-50_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_01-37-50_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `my_spatial_dqn_agent`, `my_wm_agent`, `rule_based_agent`
- **Timestamp:** 2026-09-07 02:20:05

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **50.0%** | 32.0% | `+18.0%` |
| **Survival Rate** | **64.0%** | 27.3% | `+36.7%` |
| **Score / Round** | **3.96 ± 2.6** | 3.11 | `+0.85` |
| **Coins / Round** | **2.56 ± 1.2** | 2.15 | `+0.41` |
| **Coin Share** | **28.4%** | 23.9% | `+4.6%` |
| **Kills / Round** | **0.28 ± 0.5** | 0.19 | `+0.09` |
| **Kill / Death Ratio (KDR)** | **0.64** | 0.25 | `+0.38` |
| **Steps Survived** | **271.3 ± 132** | 191.8 | `+79.5` |
| **Suicides / Round** | **0.20** | 0.60 | `-0.40` |
| **Wait % of Actions** | **9.6%** | 4.8% | `+4.8%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **50.0%** | 64.0% | 3.96 | 2.56 | 0.28 | 10 | 12 | 271.3 | 9.6% |
| my_spatial_dqn_agent | 50.0% | 36.0% | 4.44 | 2.94 | 0.30 | 30 | 5 | 202.2 | 7.5% |
| my_wm_agent | 24.0% | 4.0% | 2.30 | 1.50 | 0.16 | 40 | 13 | 138.4 | 6.8% |
| rule_based_agent | 22.0% | 42.0% | 2.60 | 2.00 | 0.12 | 20 | 13 | 234.7 | 0.1% |
| *Rule-Based Avg* | *32.0%* | *27.3%* | *3.11* | *2.15* | *0.19* | *30.0* | *10.3* | *191.8* | *4.8%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 11114 | 222.3 | 81.9% |
| Bombs Placed | 1019 | 20.4 | 7.5% |
| Waited (WAIT) | 1304 | 26.1 | 9.6% |
| Invalid Actions | 127 | 2.5 | 0.9% |
| Crates Destroyed | 1609 | 32.2 | - |
| **Total Actions** | **13564** | **271.3** | **100.0%** |
