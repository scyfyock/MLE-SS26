# BombeRLe Evaluation Report: `dyna_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/dyna_agent/dyna-model.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/dyna_agent/dyna-model.pt
- **Scenario:** `classic` | **Rounds:** 100
- **Opponents:** `qwm_agent_clean`
- **Timestamp:** 2026-09-20 17:33:45

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **50.0%** | 58.0% | `-8.0%` |
| **Survival Rate** | **66.0%** | 78.0% | `-12.0%` |
| **Score / Round** | **4.55 ± 2.4** | 4.56 | `-0.01` |
| **Coins / Round** | **4.20 ± 1.7** | 4.36 | `-0.16` |
| **Coin Share** | **49.1%** | 50.9% | `-1.9%` |
| **Kills / Round** | **0.07 ± 0.3** | 0.04 | `+0.03` |
| **Kill / Death Ratio (KDR)** | **0.20** | 0.17 | `+0.03` |
| **Steps Survived** | **321.9 ± 119** | 346.9 | `-25.0` |
| **Suicides / Round** | **0.31** | 0.16 | `+0.15` |
| **Wait % of Actions** | **3.3%** | 5.1% | `-1.8%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **dyna_agent** | **50.0%** | 66.0% | 4.55 | 4.20 | 0.07 | 31 | 4 | 321.9 | 3.3% |
| qwm_agent_clean | 58.0% | 78.0% | 4.56 | 4.36 | 0.04 | 16 | 7 | 346.9 | 5.1% |
| *Rule-Based Avg* | *58.0%* | *78.0%* | *4.56* | *4.36* | *0.04* | *16.0* | *7.0* | *346.9* | *5.1%* |

## Action Profile for `dyna_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 27396 | 274.0 | 85.1% |
| Bombs Placed | 3649 | 36.5 | 11.3% |
| Waited (WAIT) | 1060 | 10.6 | 3.3% |
| Invalid Actions | 88 | 0.9 | 0.3% |
| Crates Destroyed | 5911 | 59.1 | - |
| **Total Actions** | **32193** | **321.9** | **100.0%** |
