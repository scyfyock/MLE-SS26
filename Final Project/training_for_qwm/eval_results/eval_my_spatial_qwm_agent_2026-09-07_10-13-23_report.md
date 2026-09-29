# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_02-21-03_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_02-21-03_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 10:16:25

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **16.0%** | 36.0% | `-20.0%` |
| **Survival Rate** | **62.0%** | 48.7% | `+13.3%` |
| **Score / Round** | **2.52 ± 1.4** | 3.25 | `-0.73` |
| **Coins / Round** | **2.52 ± 1.4** | 2.15 | `+0.37` |
| **Coin Share** | **28.1%** | 24.0% | `+4.1%` |
| **Kills / Round** | **0.00 ± 0.0** | 0.22 | `-0.22` |
| **Kill / Death Ratio (KDR)** | **0.00** | 0.40 | `-0.40` |
| **Steps Survived** | **305.8 ± 131** | 259.1 | `+46.7` |
| **Suicides / Round** | **0.06** | 0.47 | `-0.41` |
| **Wait % of Actions** | **7.3%** | 0.1% | `+7.2%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **16.0%** | 62.0% | 2.52 | 2.52 | 0.00 | 3 | 18 | 305.8 | 7.3% |
| rule_based_agent_0 | 44.0% | 56.0% | 3.46 | 2.36 | 0.22 | 21 | 4 | 278.6 | 0.1% |
| rule_based_agent_1 | 24.0% | 42.0% | 2.76 | 1.86 | 0.18 | 27 | 6 | 227.5 | 0.1% |
| rule_based_agent_2 | 40.0% | 48.0% | 3.54 | 2.24 | 0.26 | 22 | 5 | 271.3 | 0.1% |
| *Rule-Based Avg* | *36.0%* | *48.7%* | *3.25* | *2.15* | *0.22* | *23.3* | *5.0* | *259.1* | *0.1%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 13381 | 267.6 | 87.5% |
| Bombs Placed | 711 | 14.2 | 4.6% |
| Waited (WAIT) | 1110 | 22.2 | 7.3% |
| Invalid Actions | 90 | 1.8 | 0.6% |
| Crates Destroyed | 1417 | 28.3 | - |
| **Total Actions** | **15292** | **305.8** | **100.0%** |
