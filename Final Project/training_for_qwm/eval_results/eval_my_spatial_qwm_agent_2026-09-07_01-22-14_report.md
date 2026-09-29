# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_00-50-35_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_00-50-35_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 20
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 01:23:31

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **60.0%** | 20.0% | `+40.0%` |
| **Survival Rate** | **85.0%** | 38.3% | `+46.7%` |
| **Score / Round** | **5.05 ± 2.5** | 2.48 | `+2.57` |
| **Coins / Round** | **3.05 ± 1.1** | 1.98 | `+1.07` |
| **Coin Share** | **33.9%** | 22.0% | `+11.9%` |
| **Kills / Round** | **0.40 ± 0.5** | 0.10 | `+0.30` |
| **Kill / Death Ratio (KDR)** | **2.67** | 0.15 | `+2.52` |
| **Steps Survived** | **375.7 ± 49** | 248.8 | `+126.8` |
| **Suicides / Round** | **0.10** | 0.45 | `-0.35` |
| **Wait % of Actions** | **9.2%** | 0.0% | `+9.2%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **60.0%** | 85.0% | 5.05 | 3.05 | 0.40 | 2 | 1 | 375.7 | 9.2% |
| rule_based_agent_0 | 20.0% | 25.0% | 2.20 | 1.70 | 0.10 | 9 | 9 | 223.8 | 0.0% |
| rule_based_agent_1 | 30.0% | 35.0% | 3.00 | 2.25 | 0.15 | 10 | 3 | 228.8 | 0.1% |
| rule_based_agent_2 | 10.0% | 55.0% | 2.25 | 2.00 | 0.05 | 8 | 1 | 294.0 | 0.0% |
| *Rule-Based Avg* | *20.0%* | *38.3%* | *2.48* | *1.98* | *0.10* | *9.0* | *4.3* | *248.8* | *0.0%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 6260 | 313.0 | 83.3% |
| Bombs Placed | 485 | 24.2 | 6.5% |
| Waited (WAIT) | 692 | 34.6 | 9.2% |
| Invalid Actions | 77 | 3.9 | 1.0% |
| Crates Destroyed | 671 | 33.5 | - |
| **Total Actions** | **7514** | **375.7** | **100.0%** |
