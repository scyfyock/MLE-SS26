# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_00-50-35_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-07_00-50-35_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 20
- **Opponents:** 3x `rule_based_agent`
- **Timestamp:** 2026-09-07 01:08:05

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **40.0%** | 36.7% | `+3.3%` |
| **Survival Rate** | **5.0%** | 40.0% | `-35.0%` |
| **Score / Round** | **3.15 ± 3.7** | 2.60 | `+0.55` |
| **Coins / Round** | **1.90 ± 1.6** | 2.35 | `-0.45` |
| **Coin Share** | **21.2%** | 26.3% | `-5.0%` |
| **Kills / Round** | **0.25 ± 0.5** | 0.05 | `+0.20` |
| **Kill / Death Ratio (KDR)** | **0.26** | 0.09 | `+0.17` |
| **Steps Survived** | **112.2 ± 65** | 208.5 | `-96.3` |
| **Suicides / Round** | **0.95** | 0.53 | `+0.42` |
| **Wait % of Actions** | **8.3%** | 0.4% | `+7.9%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **40.0%** | 5.0% | 3.15 | 1.90 | 0.25 | 19 | 0 | 112.2 | 8.3% |
| rule_based_agent_0 | 35.0% | 30.0% | 2.65 | 2.40 | 0.05 | 11 | 6 | 194.2 | 0.3% |
| rule_based_agent_1 | 50.0% | 55.0% | 3.25 | 2.75 | 0.10 | 9 | 0 | 241.1 | 0.6% |
| rule_based_agent_2 | 25.0% | 35.0% | 1.90 | 1.90 | 0.00 | 12 | 2 | 190.2 | 0.4% |
| *Rule-Based Avg* | *36.7%* | *40.0%* | *2.60* | *2.35* | *0.05* | *10.7* | *2.7* | *208.5* | *0.4%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1767 | 88.3 | 78.7% |
| Bombs Placed | 267 | 13.3 | 11.9% |
| Waited (WAIT) | 186 | 9.3 | 8.3% |
| Invalid Actions | 24 | 1.2 | 1.1% |
| Crates Destroyed | 503 | 25.1 | - |
| **Total Actions** | **2244** | **112.2** | **100.0%** |
