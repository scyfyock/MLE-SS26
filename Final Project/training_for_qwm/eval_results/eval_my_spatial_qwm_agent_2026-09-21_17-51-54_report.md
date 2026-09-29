# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/runs/run_2026-09-11_23-46-21_spatial_qwm/best_model_s4_vs_rule_based.pt`
- **Source:** Run run_2026-09-11_23-46-21_spatial_qwm (best Stage 4 vs rule_based)
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-21 17:52:32

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **30.0%** | 33.3% | `-3.3%` |
| **Survival Rate** | **40.0%** | 30.0% | `+10.0%` |
| **Score / Round** | **3.10 ± 4.1** | 3.97 | `-0.87` |
| **Coins / Round** | **1.60 ± 1.3** | 2.47 | `-0.87` |
| **Coin Share** | **17.8%** | 27.4% | `-9.6%` |
| **Kills / Round** | **0.30 ± 0.9** | 0.30 | `+0.00` |
| **Kill / Death Ratio (KDR)** | **0.50** | 0.41 | `+0.09` |
| **Steps Survived** | **219.1 ± 122** | 217.7 | `+1.4` |
| **Suicides / Round** | **0.40** | 0.53 | `-0.13` |
| **Wait % of Actions** | **4.6%** | 0.4% | `+4.2%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **30.0%** | 40.0% | 3.10 | 1.60 | 0.30 | 4 | 2 | 219.1 | 4.6% |
| rule_based_agent_0 | 50.0% | 50.0% | 4.80 | 2.30 | 0.50 | 4 | 2 | 254.8 | 0.7% |
| rule_based_agent_1 | 10.0% | 20.0% | 3.60 | 2.10 | 0.30 | 6 | 4 | 183.1 | 0.4% |
| rule_based_agent_2 | 40.0% | 20.0% | 3.50 | 3.00 | 0.10 | 6 | 4 | 215.3 | 0.1% |
| *Rule-Based Avg* | *33.3%* | *30.0%* | *3.97* | *2.47* | *0.30* | *5.3* | *3.3* | *217.7* | *0.4%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 1859 | 185.9 | 84.8% |
| Bombs Placed | 203 | 20.3 | 9.3% |
| Waited (WAIT) | 101 | 10.1 | 4.6% |
| Invalid Actions | 28 | 2.8 | 1.3% |
| Crates Destroyed | 212 | 21.2 | - |
| **Total Actions** | **2191** | **219.1** | **100.0%** |
