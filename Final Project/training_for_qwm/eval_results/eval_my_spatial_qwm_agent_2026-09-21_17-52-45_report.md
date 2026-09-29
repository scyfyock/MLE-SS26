# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-11_23-46-21_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-11_23-46-21_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-21 17:53:28

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **20.0%** | 43.3% | `-23.3%` |
| **Survival Rate** | **30.0%** | 40.0% | `-10.0%` |
| **Score / Round** | **4.10 ± 4.0** | 3.57 | `+0.53` |
| **Coins / Round** | **2.10 ± 1.0** | 2.23 | `-0.13` |
| **Coin Share** | **23.9%** | 25.4% | `-1.5%` |
| **Kills / Round** | **0.40 ± 0.7** | 0.27 | `+0.13` |
| **Kill / Death Ratio (KDR)** | **0.50** | 0.39 | `+0.11` |
| **Steps Survived** | **252.7 ± 134** | 234.2 | `+18.5` |
| **Suicides / Round** | **0.60** | 0.37 | `+0.23` |
| **Wait % of Actions** | **6.3%** | 0.5% | `+5.8%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **20.0%** | 30.0% | 4.10 | 2.10 | 0.40 | 6 | 2 | 252.7 | 6.3% |
| rule_based_agent_0 | 40.0% | 30.0% | 4.50 | 3.00 | 0.30 | 4 | 4 | 235.6 | 0.3% |
| rule_based_agent_1 | 50.0% | 50.0% | 3.80 | 1.80 | 0.40 | 3 | 3 | 234.0 | 0.6% |
| rule_based_agent_2 | 40.0% | 40.0% | 2.40 | 1.90 | 0.10 | 4 | 3 | 233.0 | 0.6% |
| *Rule-Based Avg* | *43.3%* | *40.0%* | *3.57* | *2.23* | *0.27* | *3.7* | *3.3* | *234.2* | *0.5%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 2125 | 212.5 | 84.1% |
| Bombs Placed | 220 | 22.0 | 8.7% |
| Waited (WAIT) | 159 | 15.9 | 6.3% |
| Invalid Actions | 23 | 2.3 | 0.9% |
| Crates Destroyed | 236 | 23.6 | - |
| **Total Actions** | **2527** | **252.7** | **100.0%** |
