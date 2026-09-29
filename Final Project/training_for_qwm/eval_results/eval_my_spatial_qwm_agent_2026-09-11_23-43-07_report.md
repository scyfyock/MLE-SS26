# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-11 23:44:24

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **50.0%** | 23.3% | `+26.7%` |
| **Survival Rate** | **60.0%** | 43.3% | `+16.7%` |
| **Score / Round** | **3.40 ± 2.0** | 3.00 | `+0.40` |
| **Coins / Round** | **2.90 ± 1.3** | 2.00 | `+0.90` |
| **Coin Share** | **32.6%** | 22.5% | `+10.1%` |
| **Kills / Round** | **0.10 ± 0.3** | 0.20 | `-0.10` |
| **Kill / Death Ratio (KDR)** | **0.25** | 0.28 | `-0.03` |
| **Steps Survived** | **321.3 ± 124** | 259.6 | `+61.7` |
| **Suicides / Round** | **0.30** | 0.53 | `-0.23` |
| **Wait % of Actions** | **1.6%** | 0.2% | `+1.4%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **50.0%** | 60.0% | 3.40 | 2.90 | 0.10 | 3 | 1 | 321.3 | 1.6% |
| rule_based_agent_0 | 30.0% | 40.0% | 3.40 | 2.40 | 0.20 | 6 | 1 | 253.7 | 0.3% |
| rule_based_agent_1 | 30.0% | 50.0% | 3.20 | 2.20 | 0.20 | 4 | 2 | 283.0 | 0.1% |
| rule_based_agent_2 | 10.0% | 40.0% | 2.40 | 1.40 | 0.20 | 6 | 3 | 242.2 | 0.0% |
| *Rule-Based Avg* | *23.3%* | *43.3%* | *3.00* | *2.00* | *0.20* | *5.3* | *2.0* | *259.6* | *0.2%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 2858 | 285.8 | 89.0% |
| Bombs Placed | 272 | 27.2 | 8.5% |
| Waited (WAIT) | 50 | 5.0 | 1.6% |
| Invalid Actions | 33 | 3.3 | 1.0% |
| Crates Destroyed | 275 | 27.5 | - |
| **Total Actions** | **3213** | **321.3** | **100.0%** |
