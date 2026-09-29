# BombeRLe Evaluation Report: `qwm_agent_veryclean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_veryclean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 10
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-21 17:55:46

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **40.0%** | 30.0% | `+10.0%` |
| **Survival Rate** | **60.0%** | 43.3% | `+16.7%` |
| **Score / Round** | **4.40 ± 3.2** | 2.87 | `+1.53` |
| **Coins / Round** | **2.90 ± 1.4** | 2.03 | `+0.87` |
| **Coin Share** | **32.2%** | 22.6% | `+9.6%` |
| **Kills / Round** | **0.30 ± 0.5** | 0.17 | `+0.13` |
| **Kill / Death Ratio (KDR)** | **0.75** | 0.35 | `+0.40` |
| **Steps Survived** | **325.3 ± 113** | 267.3 | `+58.0` |
| **Suicides / Round** | **0.30** | 0.37 | `-0.07` |
| **Wait % of Actions** | **5.8%** | 0.2% | `+5.7%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_veryclean** | **40.0%** | 60.0% | 4.40 | 2.90 | 0.30 | 3 | 1 | 325.3 | 5.8% |
| rule_based_agent_0 | 10.0% | 40.0% | 2.90 | 1.90 | 0.20 | 4 | 3 | 255.4 | 0.4% |
| rule_based_agent_1 | 20.0% | 30.0% | 2.00 | 2.00 | 0.00 | 4 | 3 | 220.8 | 0.0% |
| rule_based_agent_2 | 60.0% | 60.0% | 3.70 | 2.20 | 0.30 | 3 | 1 | 325.8 | 0.2% |
| *Rule-Based Avg* | *30.0%* | *43.3%* | *2.87* | *2.03* | *0.17* | *3.7* | *2.3* | *267.3* | *0.2%* |

## Action Profile for `qwm_agent_veryclean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 2702 | 270.2 | 83.1% |
| Bombs Placed | 325 | 32.5 | 10.0% |
| Waited (WAIT) | 190 | 19.0 | 5.8% |
| Invalid Actions | 36 | 3.6 | 1.1% |
| Crates Destroyed | 292 | 29.2 | - |
| **Total Actions** | **3253** | **325.3** | **100.0%** |
