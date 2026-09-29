# BombeRLe Evaluation Report: `my_spatial_qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-11 08:55:39

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **46.0%** | 35.3% | `+10.7%` |
| **Survival Rate** | **60.0%** | 36.0% | `+24.0%` |
| **Score / Round** | **3.10 ± 2.6** | 3.17 | `-0.07` |
| **Coins / Round** | **2.30 ± 1.4** | 2.23 | `+0.07` |
| **Coin Share** | **25.6%** | 24.8% | `+0.7%` |
| **Kills / Round** | **0.16 ± 0.4** | 0.19 | `-0.03` |
| **Kill / Death Ratio (KDR)** | **0.38** | 0.26 | `+0.12` |
| **Steps Survived** | **280.0 ± 143** | 230.2 | `+49.8` |
| **Suicides / Round** | **0.28** | 0.51 | `-0.23` |
| **Wait % of Actions** | **2.2%** | 0.2% | `+2.0%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **my_spatial_qwm_agent** | **46.0%** | 60.0% | 3.10 | 2.30 | 0.16 | 14 | 7 | 280.0 | 2.2% |
| rule_based_agent_0 | 30.0% | 34.0% | 2.98 | 2.18 | 0.16 | 27 | 10 | 226.7 | 0.2% |
| rule_based_agent_1 | 34.0% | 34.0% | 3.52 | 2.32 | 0.24 | 25 | 12 | 235.7 | 0.1% |
| rule_based_agent_2 | 42.0% | 40.0% | 3.00 | 2.20 | 0.16 | 25 | 7 | 228.2 | 0.3% |
| *Rule-Based Avg* | *35.3%* | *36.0%* | *3.17* | *2.23* | *0.19* | *25.7* | *9.7* | *230.2* | *0.2%* |

## Action Profile for `my_spatial_qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 12369 | 247.4 | 88.3% |
| Bombs Placed | 1204 | 24.1 | 8.6% |
| Waited (WAIT) | 314 | 6.3 | 2.2% |
| Invalid Actions | 115 | 2.3 | 0.8% |
| Crates Destroyed | 1271 | 25.4 | - |
| **Total Actions** | **14002** | **280.0** | **100.0%** |
