# BombeRLe Evaluation Report: `dyna_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/dyna_agent/dyna-model.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/dyna_agent/dyna-model.pt
- **Scenario:** `classic` | **Rounds:** 100
- **Opponents:** `rule_based_agent_0`, `rule_based_agent_1`, `rule_based_agent_2`
- **Timestamp:** 2026-09-20 17:22:06

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **36.0%** | 30.3% | `+5.7%` |
| **Survival Rate** | **50.0%** | 40.0% | `+10.0%` |
| **Score / Round** | **3.68 ± 2.7** | 3.09 | `+0.59` |
| **Coins / Round** | **2.43 ± 1.4** | 2.19 | `+0.24` |
| **Coin Share** | **27.0%** | 24.3% | `+2.7%` |
| **Kills / Round** | **0.25 ± 0.5** | 0.18 | `+0.07` |
| **Kill / Death Ratio (KDR)** | **0.48** | 0.27 | `+0.21` |
| **Steps Survived** | **264.4 ± 149** | 240.4 | `+24.0` |
| **Suicides / Round** | **0.42** | 0.46 | `-0.04` |
| **Wait % of Actions** | **2.9%** | 0.2% | `+2.7%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **dyna_agent** | **36.0%** | 50.0% | 3.68 | 2.43 | 0.25 | 42 | 10 | 264.4 | 2.9% |
| rule_based_agent_0 | 29.0% | 48.0% | 3.19 | 2.14 | 0.21 | 40 | 17 | 256.8 | 0.2% |
| rule_based_agent_1 | 33.0% | 36.0% | 3.05 | 2.20 | 0.17 | 50 | 24 | 233.0 | 0.2% |
| rule_based_agent_2 | 29.0% | 36.0% | 3.02 | 2.22 | 0.16 | 48 | 28 | 231.4 | 0.2% |
| *Rule-Based Avg* | *30.3%* | *40.0%* | *3.09* | *2.19* | *0.18* | *46.0* | *23.0* | *240.4* | *0.2%* |

## Action Profile for `dyna_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 22535 | 225.3 | 85.2% |
| Bombs Placed | 3016 | 30.2 | 11.4% |
| Waited (WAIT) | 754 | 7.5 | 2.9% |
| Invalid Actions | 138 | 1.4 | 0.5% |
| Crates Destroyed | 3102 | 31.0 | - |
| **Total Actions** | **26443** | **264.4** | **100.0%** |
