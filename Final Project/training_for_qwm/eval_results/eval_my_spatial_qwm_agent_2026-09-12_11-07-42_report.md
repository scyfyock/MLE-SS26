# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-12 11:11:02

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **50.0%** | 23.3% | `+26.7%` |
| **Survival Rate** | **70.0%** | 36.7% | `+33.3%` |
| **Score / Round** | **3.50 ± 2.2** | 3.00 | `+0.50` |
| **Coins / Round** | **2.50 ± 0.8** | 2.17 | `+0.33` |
| **Coin Share** | **27.8%** | 24.1% | `+3.7%` |
| **Kills / Round** | **0.20 ± 0.4** | 0.17 | `+0.03` |
| **Kill / Death Ratio (KDR)** | **0.67** | 0.33 | `+0.34` |
| **Steps Survived** | **327.1 ± 82** | 228.5 | `+98.6` |
| **Suicides / Round** | **0.20** | 0.53 | `-0.33` |
| **Wait % of Actions** | **2.5%** | 0.1% | `+2.4%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **50.0%** | 70.0% | 3.50 | 2.50 | 0.20 | 2 | 1 | 327.1 | 2.5% |
| rule_based_agent_0 | 40.0% | 60.0% | 3.70 | 2.20 | 0.30 | 4 | 0 | 262.4 | 0.2% |
| rule_based_agent_1 | 0.0% | 30.0% | 2.50 | 2.00 | 0.10 | 4 | 4 | 229.8 | 0.0% |
| rule_based_agent_2 | 30.0% | 20.0% | 2.80 | 2.30 | 0.10 | 8 | 2 | 193.2 | 0.1% |
| *Rule-Based Avg* | *23.3%* | *36.7%* | *3.00* | *2.17* | *0.17* | *5.3* | *2.0* | *228.5* | *0.1%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 2877 | 287.7 | 88.0% |
| Bombs Placed | 282 | 28.2 | 8.6% |
| Waited (WAIT) | 83 | 8.3 | 2.5% |
| Invalid Actions | 29 | 2.9 | 0.9% |
| Crates Destroyed | 304 | 30.4 | - |
| **Total Actions** | **3271** | **327.1** | **100.0%** |
