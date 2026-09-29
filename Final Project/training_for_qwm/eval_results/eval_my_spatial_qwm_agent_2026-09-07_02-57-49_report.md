# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_02-21-03_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Explicit file: agent_code/my_spatial_qwm_agent/runs/run_2026-09-07_02-21-03_spatial_qwm/best_model_s4_vs_rule_based.pt
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-07 02:59:08

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **30.0%** | 30.0% | `+0.0%` |
| **Survival Rate** | **60.0%** | 50.0% | `+10.0%` |
| **Score / Round** | **3.10 ± 1.9** | 2.80 | `+0.30` |
| **Coins / Round** | **2.60 ± 1.0** | 2.13 | `+0.47` |
| **Coin Share** | **28.9%** | 23.7% | `+5.2%` |
| **Kills / Round** | **0.10 ± 0.3** | 0.13 | `-0.03` |
| **Kill / Death Ratio (KDR)** | **0.25** | 0.27 | `-0.02` |
| **Steps Survived** | **291.4 ± 143** | 255.2 | `+36.2` |
| **Suicides / Round** | **0.30** | 0.37 | `-0.07` |
| **Wait % of Actions** | **5.6%** | 0.1% | `+5.5%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **30.0%** | 60.0% | 3.10 | 2.60 | 0.10 | 3 | 1 | 291.4 | 5.6% |
| rule_based_agent_0 | 50.0% | 50.0% | 3.90 | 2.40 | 0.30 | 4 | 1 | 255.5 | 0.2% |
| rule_based_agent_1 | 20.0% | 50.0% | 1.60 | 1.60 | 0.00 | 3 | 2 | 243.4 | 0.1% |
| rule_based_agent_2 | 20.0% | 50.0% | 2.90 | 2.40 | 0.10 | 4 | 1 | 266.8 | 0.0% |
| *Rule-Based Avg* | *30.0%* | *50.0%* | *2.80* | *2.13* | *0.13* | *3.7* | *1.3* | *255.2* | *0.1%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 2540 | 254.0 | 87.1% |
| Bombs Placed | 199 | 19.9 | 6.8% |
| Waited (WAIT) | 163 | 16.3 | 5.6% |
| Invalid Actions | 13 | 1.3 | 0.4% |
| Crates Destroyed | 327 | 32.7 | - |
| **Total Actions** | **2915** | **291.5** | **100.0%** |
