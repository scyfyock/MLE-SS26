# BombeRLe Evaluation Report: `my_spatial_dqn_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-06_18-24-53_spatial_dqn (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 5
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 00:45:26

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **40.0%** | 40.0% | `+0.0%` |
| **Survival Rate** | **20.0%** | 40.0% | `-20.0%` |
| **Score / Round** | **3.60 ± 2.0** | 3.13 | `+0.47` |
| **Coins / Round** | **2.60 ± 1.0** | 2.13 | `+0.47` |
| **Coin Share** | **28.9%** | 23.7% | `+5.2%` |
| **Kills / Round** | **0.20 ± 0.4** | 0.20 | `+0.00` |
| **Kill / Death Ratio (KDR)** | **0.25** | 0.25 | `+0.00` |
| **Steps Survived** | **164.8 ± 121** | 222.9 | `-58.1` |
| **Suicides / Round** | **0.80** | 0.47 | `+0.33` |
| **Wait % of Actions** | **9.6%** | 0.6% | `+9.0%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_dqn_agent** | **40.0%** | 20.0% | 3.60 | 2.60 | 0.20 | 4 | 0 | 164.8 | 9.6% |
| rule_based_agent_0 | 20.0% | 40.0% | 2.00 | 2.00 | 0.00 | 2 | 1 | 204.0 | 1.4% |
| rule_based_agent_1 | 80.0% | 40.0% | 5.80 | 2.80 | 0.60 | 3 | 1 | 243.6 | 0.4% |
| rule_based_agent_2 | 20.0% | 40.0% | 1.60 | 1.60 | 0.00 | 2 | 2 | 221.2 | 0.1% |
| *Rule-Based Avg* | *40.0%* | *40.0%* | *3.13* | *2.13* | *0.20* | *2.3* | *1.3* | *222.9* | *0.6%* |

## Action Profile for `my_spatial_dqn_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 633 | 126.6 | 76.8% |
| Bombs Placed | 103 | 20.6 | 12.5% |
| Waited (WAIT) | 79 | 15.8 | 9.6% |
| Invalid Actions | 9 | 1.8 | 1.1% |
| Crates Destroyed | 136 | 27.2 | - |
| **Total Actions** | **824** | **164.8** | **100.0%** |
