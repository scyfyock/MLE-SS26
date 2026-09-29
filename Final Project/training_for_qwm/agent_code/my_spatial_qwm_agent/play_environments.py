#!/usr/bin/env python3
"""
play_environments.py -- Evaluates my_spatial_qwm_agent in the target environments.

Target Environments:
1. Environment 6 (s4_vs_rule_based):
   Scenario: 'classic'
   Opponents: ['rule_based_agent', 'rule_based_agent', 'rule_based_agent']

2. Environment 7 (s7_vs_mixed_classic):
   Scenario: 'classic'
   Opponents: ['my_spatial_dqn_agent', 'my_spatial_qwm_agent', 'rule_based_agent']

Tracks comprehensive metrics (Win Rate, Survival Rate, Score, Coins, Crates, Kills, Suicides)
and saves telemetry JSON, CSV, and comparative plots to a dedicated run folder.
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Root directory resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

AGENT_DIR = Path(__file__).resolve().parent

import events as e
import settings as s
from environment import BombeRLeWorld, WorldArgs


TARGET_ENVIRONMENTS = {
    "s4_vs_rule_based": {
        "env_id": 6,
        "name": "s4_vs_rule_based",
        "scenario": "classic",
        "opponents": ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
        "description": "my_spatial_qwm_agent vs 3x rule_based_agent (Classic)",
    },
    "s7_vs_mixed_classic": {
        "env_id": 7,
        "name": "s7_vs_mixed_classic",
        "scenario": "classic",
        "opponents": ["my_spatial_dqn_agent", "my_spatial_qwm_agent", "rule_based_agent"],
        "description": "my_spatial_qwm_agent vs my_spatial_dqn_agent + my_spatial_qwm_agent clone + rule_based_agent (Classic)",
    },
}


def create_run_folder(base_runs_dir: str = None) -> str:
    runs_dir = base_runs_dir or os.path.join(str(AGENT_DIR), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(runs_dir, f"run_{timestamp}_play_eval")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def play_tournament_in_environment(
    env_key: str,
    n_rounds: int = 20,
    seed: int = 1000,
) -> Dict:
    """Runs n_rounds in the specified environment and collects detailed agent stats."""
    cfg = TARGET_ENVIRONMENTS[env_key]
    scenario = cfg["scenario"]
    opponents = cfg["opponents"]
    agent_specs = [("my_spatial_qwm_agent", False)] + [(opp, False) for opp in opponents]

    print("\n" + "=" * 80)
    print(f" PLAYING ENVIRONMENT: {cfg['name']} ({cfg['description']})")
    print(f" Scenario: {scenario} | Rounds: {n_rounds} | Seed: {seed}")
    print("=" * 80)

    args = WorldArgs(
        no_gui=True,
        fps=0,
        turn_based=False,
        update_interval=0.0,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir="logs",
        save_stats=False,
        match_name=None,
        seed=seed,
        silence_errors=True,
        scenario=scenario,
    )

    world = BombeRLeWorld(args, agent_specs)
    agent_names = [a.name for a in world.agents]
    print(f"Agents initialized in match: {agent_names}")

    # Track per-agent metrics
    agent_stats = {
        name: {
            "scores": [],
            "coins": [],
            "crates": [],
            "kills": [],
            "suicides": [],
            "survived": [],
            "won": [],
            "steps": [],
        }
        for name in agent_names
    }

    round_history = []
    start_time = time.time()

    print("-" * 80)
    print(f"{'Round':>5} | {'Agent 0 (QWM)':>18} | {'Agent 1':>18} | {'Agent 2':>18} | {'Agent 3':>18} | {'Steps':>5}")
    print(f"{'':>5} | {'Score (Won/Surv)':>18} | {'Score (Won/Surv)':>18} | {'Score (Won/Surv)':>18} | {'Score (Won/Surv)':>18} |")
    print("-" * 80)

    for r in range(1, n_rounds + 1):
        world.new_round()
        step_count = 0
        while world.running:
            world.do_step(None)
            step_count += 1

        # Determine winner
        living = [a for a in world.agents if not a.dead]
        max_score = max((a.score for a in world.agents), default=0)
        sole_survivor = living[0] if len(living) == 1 else None

        round_entry = {"round": r, "steps": step_count, "agents": {}}
        round_display = []

        for a in world.agents:
            survived = int(not a.dead)
            # Strict point win: must hold highest positive score
            won = int(a.score == max_score and a.score > 0)
            score = a.score
            coins = a.statistics.get("coins", 0)
            crates = a.statistics.get("crates", 0)
            kills = a.statistics.get("kills", 0)
            suicides = a.statistics.get("suicides", 0)

            agent_stats[a.name]["scores"].append(score)
            agent_stats[a.name]["coins"].append(coins)
            agent_stats[a.name]["crates"].append(crates)
            agent_stats[a.name]["kills"].append(kills)
            agent_stats[a.name]["suicides"].append(suicides)
            agent_stats[a.name]["survived"].append(survived)
            agent_stats[a.name]["won"].append(won)
            agent_stats[a.name]["steps"].append(step_count)

            round_entry["agents"][a.name] = {
                "score": score,
                "coins": coins,
                "crates": crates,
                "kills": kills,
                "suicides": suicides,
                "survived": survived,
                "won": won,
            }
            round_display.append(f"{score:>2d} (w={won}/s={survived})")

        round_history.append(round_entry)
        display_str = " | ".join(f"{s:>18}" for s in round_display)
        print(f"{r:>5d} | {display_str} | {step_count:>5d}")

    world.end()
    elapsed = time.time() - start_time

    # Compute aggregates per agent
    summary = {}
    for name in agent_names:
        st = agent_stats[name]
        n = len(st["scores"])
        summary[name] = {
            "rounds": n,
            "win_count": int(sum(st["won"])),
            "win_rate": float(np.mean(st["won"])) * 100.0 if n > 0 else 0.0,
            "survival_count": int(sum(st["survived"])),
            "survival_rate": float(np.mean(st["survived"])) * 100.0 if n > 0 else 0.0,
            "mean_score": float(np.mean(st["scores"])) if n > 0 else 0.0,
            "std_score": float(np.std(st["scores"])) if n > 0 else 0.0,
            "max_score": int(max(st["scores"])) if n > 0 else 0,
            "total_coins": int(sum(st["coins"])),
            "mean_coins": float(np.mean(st["coins"])) if n > 0 else 0.0,
            "total_crates": int(sum(st["crates"])),
            "mean_crates": float(np.mean(st["crates"])) if n > 0 else 0.0,
            "total_kills": int(sum(st["kills"])),
            "mean_kills": float(np.mean(st["kills"])) if n > 0 else 0.0,
            "total_suicides": int(sum(st["suicides"])),
        }

    print("-" * 80)
    print(f"Summary for {cfg['name']} over {n_rounds} rounds ({elapsed:.1f}s):")
    for name in agent_names:
        sm = summary[name]
        print(f"  [{name:<25}] Win: {sm['win_rate']:>5.1f}% ({sm['win_count']}/{n_rounds}) | "
              f"Surv: {sm['survival_rate']:>5.1f}% | Score: {sm['mean_score']:>4.2f} ± {sm['std_score']:>4.2f} | "
              f"Coins: {sm['total_coins']:>2d} | Kills: {sm['total_kills']:>2d} | Suicides: {sm['total_suicides']:>2d}")

    return {
        "env_key": env_key,
        "config": cfg,
        "n_rounds": n_rounds,
        "elapsed_s": elapsed,
        "agent_names": agent_names,
        "agent_stats": agent_stats,
        "summary": summary,
        "round_history": round_history,
    }


def plot_comparison(results_env6: Dict, results_env7: Dict, out_path: str):
    """Plots a 4-panel comparison figure between Environment 6 and Environment 7."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    fig.patch.set_facecolor('#0f111a')

    dark_bg = '#1a1c29'
    grid_col = '#2c3144'
    text_col = '#e0e6ed'
    accent_green = '#00e676'
    accent_blue = '#4fc3f7'
    accent_orange = '#ff9100'
    accent_purple = '#b388ff'
    accent_red = '#ff5252'

    fig.suptitle("Evaluation Results: my_spatial_qwm_agent across Target Environments",
                 fontsize=14, fontweight='bold', color=text_col, y=0.98)

    for ax in axes.flat:
        ax.set_facecolor(dark_bg)
        ax.grid(True, linestyle='--', color=grid_col, alpha=0.6)
        ax.tick_params(colors=text_col, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(grid_col)

    # ------------------------------------------------------------------------
    # Panel 1: Win Rate Comparison (Env 6 vs Env 7)
    # ------------------------------------------------------------------------
    ax1 = axes[0, 0]
    envs = [results_env6, results_env7]
    env_labels = ["Env 6: s4_vs_rule_based", "Env 7: s7_vs_mixed_classic"]

    x = np.arange(len(envs))
    width = 0.35

    our_win_rates = [e["summary"][e["agent_names"][0]]["win_rate"] for e in envs]
    our_surv_rates = [e["summary"][e["agent_names"][0]]["survival_rate"] for e in envs]

    bars1 = ax1.bar(x - width/2, our_win_rates, width, label='Win Rate %', color=accent_green)
    bars2 = ax1.bar(x + width/2, our_surv_rates, width, label='Survival Rate %', color=accent_blue)

    ax1.set_ylabel('Percentage (%)', color=text_col, fontsize=10)
    ax1.set_title('my_spatial_qwm_agent: Win & Survival Rates', color=text_col, fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(env_labels, color=text_col, fontsize=9)
    ax1.set_ylim([0, 105])
    ax1.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=text_col, fontsize=9)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', color=text_col, fontsize=9, fontweight='bold')
    for bar in bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', color=text_col, fontsize=9, fontweight='bold')

    # ------------------------------------------------------------------------
    # Panel 2: Mean Scores & Coins
    # ------------------------------------------------------------------------
    ax2 = axes[0, 1]
    our_scores = [e["summary"][e["agent_names"][0]]["mean_score"] for e in envs]
    our_coins = [e["summary"][e["agent_names"][0]]["mean_coins"] for e in envs]

    bars3 = ax2.bar(x - width/2, our_scores, width, label='Mean Score / Match', color=accent_orange)
    bars4 = ax2.bar(x + width/2, our_coins, width, label='Mean Coins / Match', color=accent_purple)

    ax2.set_ylabel('Count', color=text_col, fontsize=10)
    ax2.set_title('my_spatial_qwm_agent: Average Score & Coins', color=text_col, fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(env_labels, color=text_col, fontsize=9)
    max_y = max(max(our_scores), max(our_coins), 1.0) * 1.3
    ax2.set_ylim([0, max_y])
    ax2.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=text_col, fontsize=9)

    for bar in bars3:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', color=text_col, fontsize=9, fontweight='bold')
    for bar in bars4:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', color=text_col, fontsize=9, fontweight='bold')

    # ------------------------------------------------------------------------
    # Panel 3: Env 7 Full Match Breakdown by Agent
    # ------------------------------------------------------------------------
    ax3 = axes[1, 0]
    env7_names = results_env7["agent_names"]
    env7_wins = [results_env7["summary"][name]["win_rate"] for name in env7_names]
    env7_surv = [results_env7["summary"][name]["survival_rate"] for name in env7_names]

    x7 = np.arange(len(env7_names))
    short_names = [n.replace("my_spatial_", "").replace("_agent", "") for n in env7_names]

    b1 = ax3.bar(x7 - width/2, env7_wins, width, label='Win Rate %', color=accent_green)
    b2 = ax3.bar(x7 + width/2, env7_surv, width, label='Survival Rate %', color=accent_blue)
    ax3.set_ylabel('Percentage (%)', color=text_col, fontsize=10)
    ax3.set_title('Env 7 (Mixed Classic): Head-to-Head Comparison', color=text_col, fontsize=11, fontweight='bold')
    ax3.set_xticks(x7)
    ax3.set_xticklabels(short_names, color=text_col, fontsize=9)
    ax3.set_ylim([0, 105])
    ax3.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=text_col, fontsize=9)

    # ------------------------------------------------------------------------
    # Panel 4: Combat Effectiveness (Kills vs Suicides)
    # ------------------------------------------------------------------------
    ax4 = axes[1, 1]
    our_kills = [e["summary"][e["agent_names"][0]]["total_kills"] for e in envs]
    our_suicides = [e["summary"][e["agent_names"][0]]["total_suicides"] for e in envs]

    bars5 = ax4.bar(x - width/2, our_kills, width, label='Total Opponent Kills', color=accent_green)
    bars6 = ax4.bar(x + width/2, our_suicides, width, label='Total Self-Suicides', color=accent_red)

    ax4.set_ylabel('Total Count', color=text_col, fontsize=10)
    ax4.set_title('Tactical Safety & Combat: Kills vs Suicides', color=text_col, fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(env_labels, color=text_col, fontsize=9)
    max_k = max(max(our_kills), max(our_suicides), 1) * 1.35
    ax4.set_ylim([0, max_k])
    ax4.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=text_col, fontsize=9)

    for bar in bars5:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{int(yval)}", ha='center', va='bottom', color=text_col, fontsize=9, fontweight='bold')
    for bar in bars6:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{int(yval)}", ha='center', va='bottom', color=text_col, fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n[Plot Saved] Evaluation comparison saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate my_spatial_qwm_agent in s4_vs_rule_based and s7_vs_mixed_classic")
    parser.add_argument("--rounds", type=int, default=25, help="Number of rounds per environment")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for tournament reproducibility")
    parser.add_argument("--run-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_dir = args.run_dir or create_run_folder()
    print("=" * 80)
    print(f" TARGET ENVIRONMENT TOURNAMENT RUNNER")
    print(f" Directory: {run_dir}")
    print(f" Rounds per Environment: {args.rounds}")
    print("=" * 80)

    # 1. Run Environment 6: s4_vs_rule_based
    res_env6 = play_tournament_in_environment(
        env_key="s4_vs_rule_based",
        n_rounds=args.rounds,
        seed=args.seed,
    )

    # 2. Run Environment 7: s7_vs_mixed_classic
    res_env7 = play_tournament_in_environment(
        env_key="s7_vs_mixed_classic",
        n_rounds=args.rounds,
        seed=args.seed + 1,
    )

    # 3. Save full JSON results
    json_path = os.path.join(run_dir, "play_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "env_6": res_env6,
            "env_7": res_env7,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # 4. Save CSV summary
    csv_path = os.path.join(run_dir, "play_summary.csv")
    csv_fields = ["environment", "agent", "rounds", "win_rate", "win_count", "survival_rate", "survival_count",
                  "mean_score", "std_score", "max_score", "total_coins", "total_crates", "total_kills", "total_suicides"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for env_res in [res_env6, res_env7]:
            for name in env_res["agent_names"]:
                sm = env_res["summary"][name]
                writer.writerow({
                    "environment": env_res["config"]["name"],
                    "agent": name,
                    "rounds": sm["rounds"],
                    "win_rate": f"{sm['win_rate']:.1f}",
                    "win_count": sm["win_count"],
                    "survival_rate": f"{sm['survival_rate']:.1f}",
                    "survival_count": sm["survival_count"],
                    "mean_score": f"{sm['mean_score']:.2f}",
                    "std_score": f"{sm['std_score']:.2f}",
                    "max_score": sm["max_score"],
                    "total_coins": sm["total_coins"],
                    "total_crates": sm["total_crates"],
                    "total_kills": sm["total_kills"],
                    "total_suicides": sm["total_suicides"],
                })

    # 5. Generate plot
    plot_path = os.path.join(run_dir, "tournament_comparison.png")
    plot_comparison(res_env6, res_env7, plot_path)

    # Copy to agent root and brain artifact directory
    shutil.copyfile(plot_path, os.path.join(str(AGENT_DIR), "tournament_comparison.png"))
    brain_dir = "/home/tobitoyota/.gemini/antigravity/brain/6efa50a8-ca8d-41f8-8dd9-0f458b0af786"
    if os.path.isdir(brain_dir):
        shutil.copyfile(plot_path, os.path.join(brain_dir, "tournament_comparison.png"))

    print("\n" + "=" * 80)
    print(" TOURNAMENT EVALUATION COMPLETE!")
    print(f" Run Directory: {run_dir}")
    print(f" JSON Telemetry: {json_path}")
    print(f" CSV Summary:    {csv_path}")
    print(f" Curves Plot:    {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

