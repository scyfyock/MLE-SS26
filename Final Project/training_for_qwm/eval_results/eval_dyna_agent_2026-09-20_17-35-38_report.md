# BombeRLe Evaluation Report: `dyna_agent`

- **Evaluated Model:** `/home/tobitoyota/Desktop/bomberman_rl/agent_code/dyna_agent/dyna-model.pt`
- **Source:** Explicit relative to agent: /home/tobitoyota/Desktop/bomberman_rl/agent_code/dyna_agent/dyna-model.pt
- **Scenario:** `classic` | **Rounds:** 100
- **Opponents:** `qwm_agent_clean`, `rule_based_agent_0`, `rule_based_agent_1`
- **Timestamp:** 2026-09-20 17:45:14

## Executive Highlights

| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |
|:---|---:|---:|---:|
| **Win Rate** | **32.0%** | 32.0% | `+0.0%` |
| **Survival Rate** | **40.0%** | 48.3% | `-8.3%` |
| **Score / Round** | **3.34 ± 2.7** | 3.13 | `+0.21` |
| **Coins / Round** | **2.29 ± 1.4** | 2.23 | `+0.06` |
| **Coin Share** | **25.5%** | 24.8% | `+0.6%` |
| **Kills / Round** | **0.21 ± 0.4** | 0.18 | `+0.03` |
| **Kill / Death Ratio (KDR)** | **0.32** | 0.35 | `-0.03` |
| **Steps Survived** | **255.2 ± 146** | 266.9 | `-11.7` |
| **Suicides / Round** | **0.51** | 0.38 | `+0.13` |
| **Wait % of Actions** | **2.8%** | 2.7% | `+0.1%` |

## Head-to-Head Comparison

| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **dyna_agent** | **32.0%** | 40.0% | 3.34 | 2.29 | 0.21 | 51 | 14 | 255.2 | 2.8% |
| qwm_agent_clean | 45.0% | 64.0% | 3.95 | 2.65 | 0.26 | 27 | 14 | 296.8 | 7.5% |
| rule_based_agent_0 | 28.0% | 40.0% | 2.73 | 2.08 | 0.13 | 43 | 25 | 259.1 | 0.4% |
| rule_based_agent_1 | 23.0% | 41.0% | 2.72 | 1.97 | 0.15 | 43 | 22 | 244.8 | 0.2% |
| *Rule-Based Avg* | *32.0%* | *48.3%* | *3.13* | *2.23* | *0.18* | *37.7* | *20.3* | *266.9* | *2.7%* |

## Action Profile for `dyna_agent`

| Action Type | Total Count | Per Round | % of Actions |
|:---|---:|---:|---:|
| Moves (UP/DOWN/LEFT/RIGHT) | 21828 | 218.3 | 85.5% |
| Bombs Placed | 2809 | 28.1 | 11.0% |
| Waited (WAIT) | 704 | 7.0 | 2.8% |
| Invalid Actions | 180 | 1.8 | 0.7% |
| Crates Destroyed | 3079 | 30.8 | - |
| **Total Actions** | **25521** | **255.2** | **100.0%** |
