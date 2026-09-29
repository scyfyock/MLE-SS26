# BombeRLe Evaluation Report: `qwm_agent_clean`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/qwm_agent_clean/my-saved-model-spatial-qwm.pt`
- **Source:** Agent root saved model: my-saved-model-spatial-qwm.pt
- **Scenario:** `classic` | **Rounds:** 200
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-20 14:49:35

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **41.0%** | 27.5% | `+13.5%` |
| **Survival Rate** | **63.0%** | 40.3% | `+22.7%` |
| **Score / Round** | **3.98 ± 2.9** | 3.07 | `+0.91` |
| **Coins / Round** | **2.56 ± 1.5** | 2.15 | `+0.42` |
| **Coin Share** | **28.5%** | 23.8% | `+4.6%` |
| **Kills / Round** | **0.28 ± 0.5** | 0.18 | `+0.10` |
| **Kill / Death Ratio (KDR)** | **0.68** | 0.27 | `+0.41` |
| **Steps Survived** | **302.2 ± 132** | 252.9 | `+49.3` |
| **Suicides / Round** | **0.24** | 0.47 | `-0.23` |
| **Wait % of Actions** | **9.1%** | 0.3% | `+8.8%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwm_agent_clean** | **41.0%** | 63.0% | 3.98 | 2.56 | 0.28 | 48 | 36 | 302.2 | 9.1% |
| rule_based_agent_0 | 31.0% | 37.5% | 3.17 | 2.27 | 0.18 | 95 | 50 | 247.8 | 0.3% |
| rule_based_agent_1 | 23.0% | 44.0% | 2.67 | 1.87 | 0.16 | 90 | 39 | 261.4 | 0.4% |
| rule_based_agent_2 | 28.5% | 39.5% | 3.37 | 2.29 | 0.21 | 94 | 43 | 249.5 | 0.3% |
| *Rule-Based Avg* | *27.5%* | *40.3%* | *3.07* | *2.15* | *0.18* | *93.0* | *44.0* | *252.9* | *0.3%* |

## Action Profile for `qwm_agent_clean`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 48033 | 240.2 | 79.5% |
| Bombs Placed | 6295 | 31.5 | 10.4% |
| Waited (WAIT) | 5490 | 27.4 | 9.1% |
| Invalid Actions | 626 | 3.1 | 1.0% |
| Crates Destroyed | 5530 | 27.6 | - |
| **Total Actions** | **60444** | **302.2** | **100.0%** |
