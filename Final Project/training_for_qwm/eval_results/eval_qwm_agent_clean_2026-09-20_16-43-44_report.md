# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 100
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-20 16:54:33

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **48.0%** | 25.3% | `+22.7%` |
| **Survival Rate** | **62.0%** | 39.0% | `+23.0%` |
| **Score / Round** | **4.38 ± 2.8** | 2.90 | `+1.48` |
| **Coins / Round** | **2.73 ± 1.4** | 2.08 | `+0.65` |
| **Coin Share** | **30.4%** | 23.2% | `+7.2%` |
| **Kills / Round** | **0.33 ± 0.5** | 0.16 | `+0.17` |
| **Kill / Death Ratio (KDR)** | **0.79** | 0.24 | `+0.55` |
| **Steps Survived** | **310.0 ± 130** | 240.0 | `+70.1` |
| **Suicides / Round** | **0.26** | 0.46 | `-0.20` |
| **Wait % of Actions** | **5.9%** | 0.3% | `+5.6%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **48.0%** | 62.0% | 4.38 | 2.73 | 0.33 | 26 | 16 | 310.0 | 5.9% |
| rule_based_agent_0 | 30.0% | 34.0% | 2.96 | 2.06 | 0.18 | 46 | 28 | 224.2 | 0.3% |
| rule_based_agent_1 | 26.0% | 39.0% | 2.79 | 2.04 | 0.15 | 46 | 22 | 240.5 | 0.3% |
| rule_based_agent_2 | 20.0% | 44.0% | 2.95 | 2.15 | 0.16 | 46 | 16 | 255.3 | 0.3% |
| *Rule-Based Avg* | *25.3%* | *39.0%* | *2.90* | *2.08* | *0.16* | *46.0* | *22.0* | *240.0* | *0.3%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 25794 | 257.9 | 83.2% |
| Bombs Placed | 3057 | 30.6 | 9.9% |
| Waited (WAIT) | 1825 | 18.2 | 5.9% |
| Invalid Actions | 328 | 3.3 | 1.1% |
| Crates Destroyed | 2938 | 29.4 | - |
| **Total Actions** | **31004** | **310.0** | **100.0%** |
