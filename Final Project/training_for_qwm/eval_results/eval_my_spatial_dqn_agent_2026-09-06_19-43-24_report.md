# BombeRLe Evaluation Report: `my_spatial_dqn_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-06_18-24-53_spatial_dqn (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-06 19:44:12

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **30.0%** | 41.3% | `-11.3%` |
| **Survival Rate** | **6.0%** | 44.7% | `-38.7%` |
| **Score / Round** | **3.02 ± 2.0** | 2.85 | `+0.17` |
| **Coins / Round** | **2.82 ± 1.6** | 2.05 | `+0.77` |
| **Coin Share** | **31.5%** | 22.8% | `+8.6%` |
| **Kills / Round** | **0.04 ± 0.2** | 0.16 | `-0.12` |
| **Kill / Death Ratio (KDR)** | **0.04** | 0.26 | `-0.21` |
| **Steps Survived** | **115.8 ± 63** | 233.2 | `-117.4` |
| **Suicides / Round** | **0.88** | 0.50 | `+0.38` |
| **Wait % of Actions** | **8.8%** | 0.3% | `+8.5%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_dqn_agent** | **30.0%** | 6.0% | 3.02 | 2.82 | 0.04 | 44 | 3 | 115.8 | 8.8% |
| rule_based_agent_0 | 40.0% | 34.0% | 3.30 | 1.90 | 0.28 | 30 | 8 | 213.6 | 0.2% |
| rule_based_agent_1 | 56.0% | 58.0% | 3.36 | 2.46 | 0.18 | 17 | 7 | 254.2 | 0.4% |
| rule_based_agent_2 | 28.0% | 42.0% | 1.88 | 1.78 | 0.02 | 28 | 8 | 231.6 | 0.4% |
| *Rule-Based Avg* | *41.3%* | *44.7%* | *2.85* | *2.05* | *0.16* | *25.0* | *7.7* | *233.2* | *0.3%* |

## Action Profile for `my_spatial_dqn_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 4534 | 90.7 | 78.3% |
| Bombs Placed | 669 | 13.4 | 11.6% |
| Waited (WAIT) | 511 | 10.2 | 8.8% |
| Invalid Actions | 75 | 1.5 | 1.3% |
| Crates Destroyed | 1324 | 26.5 | - |
| **Total Actions** | **5789** | **115.8** | **100.0%** |
