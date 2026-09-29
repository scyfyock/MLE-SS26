# BombeRLe Evaluation Report: `qwm_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 50
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-12 22:40:25

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **40.0%** | 30.7% | `+9.3%` |
| **Survival Rate** | **68.0%** | 36.0% | `+32.0%` |
| **Score / Round** | **4.02 ± 3.0** | 2.85 | `+1.17` |
| **Coins / Round** | **2.82 ± 1.5** | 2.05 | `+0.77` |
| **Coin Share** | **31.4%** | 22.9% | `+8.5%` |
| **Kills / Round** | **0.24 ± 0.5** | 0.16 | `+0.08` |
| **Kill / Death Ratio (KDR)** | **0.63** | 0.22 | `+0.41` |
| **Steps Survived** | **299.2 ± 137** | 234.3 | `+64.9` |
| **Suicides / Round** | **0.22** | 0.53 | `-0.31` |
| **Wait % of Actions** | **2.8%** | 0.2% | `+2.7%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent** | **40.0%** | 68.0% | 4.02 | 2.82 | 0.24 | 11 | 8 | 299.2 | 2.8% |
| rule_based_agent_0 | 30.0% | 36.0% | 3.00 | 2.30 | 0.14 | 24 | 12 | 244.9 | 0.1% |
| rule_based_agent_1 | 38.0% | 42.0% | 3.00 | 2.30 | 0.14 | 27 | 6 | 237.2 | 0.3% |
| rule_based_agent_2 | 24.0% | 30.0% | 2.56 | 1.56 | 0.20 | 29 | 10 | 220.8 | 0.2% |
| *Rule-Based Avg* | *30.7%* | *36.0%* | *2.85* | *2.05* | *0.16* | *26.7* | *9.3* | *234.3* | *0.2%* |

## Action Profile for `qwm_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 12958 | 259.2 | 86.6% |
| Bombs Placed | 1401 | 28.0 | 9.4% |
| Waited (WAIT) | 426 | 8.5 | 2.8% |
| Invalid Actions | 173 | 3.5 | 1.2% |
| Crates Destroyed | 1361 | 27.2 | - |
| **Total Actions** | **14958** | **299.2** | **100.0%** |
