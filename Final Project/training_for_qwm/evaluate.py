#!/usr/bin/env python3
"""
evaluate.py -- Comprehensive Evaluation Benchmark for BombeRLe Agents.

Tests any trained agent checkpoint against 3 rule_based_agents and generates
in-depth performance metrics:
  - Win rate (overall win %, sole survivor %, highest score %)
  - Coin rate (mean coins/round, coin share %, collection efficiency)
  - Kill rate & combat (kills/round, suicides, got killed, KDR)
  - Longevity & survival (mean steps survived, survival rate %)
  - Action profile (moves, bombs placed, waited actions, invalid actions)
  - Head-to-head comparison vs individual opponents & rule-based average

Exports:
  - Rich ANSI/ASCII terminal tables
  - JSON results
  - CSV per-round breakdown
  - Markdown executive report
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Formatting & Terminal Output Utilities
# -----------------------------------------------------------------------------

import re

ANSI_REGEX = re.compile(r'\x1b\[[0-9;]*m')


def visible_len(text: Any) -> int:
    """Returns character length excluding ANSI escape sequences."""
    return len(ANSI_REGEX.sub('', str(text)))


class Colors:
    """Terminal ANSI colors with automatic TTY detection."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.stdout.isatty()

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str: return self._c("1", text)
    def dim(self, text: str) -> str: return self._c("2", text)
    def green(self, text: str) -> str: return self._c("92", text)
    def red(self, text: str) -> str: return self._c("91", text)
    def yellow(self, text: str) -> str: return self._c("93", text)
    def cyan(self, text: str) -> str: return self._c("96", text)
    def magenta(self, text: str) -> str: return self._c("95", text)
    def blue(self, text: str) -> str: return self._c("94", text)


def render_table(headers: List[str], rows: List[List[Any]], aligns: Optional[List[str]] = None,
                 title: Optional[str] = None, use_utf8: bool = True) -> str:
    """Renders a formatted table using box-drawing characters."""
    if not rows:
        return ""

    if aligns is None:
        aligns = ['left'] + ['right'] * (len(headers) - 1)

    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]

    col_widths = [visible_len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], visible_len(cell))

    col_widths = [w + 2 for w in col_widths]

    if use_utf8:
        tl, tm, tr = "┌", "┬", "┐"
        ml, mm, mr = "├", "┼", "┤"
        bl, bm, br = "└", "┴", "┘"
        h_line = "─"
        v_line = "│"
    else:
        tl = tm = tr = ml = mm = mr = bl = bm = br = "+"
        h_line = "-"
        v_line = "|"

    lines = []
    if title:
        total_w = sum(col_widths) + len(col_widths) - 1
        lines.append(f" {title.upper()} ".center(total_w, "="))

    top_border = tl + tm.join(h_line * w for w in col_widths) + tr
    mid_border = ml + mm.join(h_line * w for w in col_widths) + mr
    bot_border = bl + bm.join(h_line * w for w in col_widths) + br

    lines.append(top_border)

    def pad_cell(text: str, width: int, align: str) -> str:
        vis_len = visible_len(text)
        avail = max(0, width - 2)
        pad_needed = max(0, avail - vis_len)
        if align == 'right':
            return " " + " " * pad_needed + text + " "
        elif align == 'center':
            left_pad = pad_needed // 2
            right_pad = pad_needed - left_pad
            return " " + " " * left_pad + text + " " * right_pad + " "
        else:
            return " " + text + " " * pad_needed + " "

    header_cells = [pad_cell(h, col_widths[i], aligns[i]) for i, h in enumerate(str_headers)]
    lines.append(v_line + v_line.join(header_cells) + v_line)
    lines.append(mid_border)

    for row in str_rows:
        row_cells = [pad_cell(cell, col_widths[i], aligns[i]) for i, cell in enumerate(row)]
        lines.append(v_line + v_line.join(row_cells) + v_line)

    lines.append(bot_border)
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Checkpoint & Model Resolution
# -----------------------------------------------------------------------------

def resolve_model_path(agent_name: str, requested_file: Optional[str] = None) -> Tuple[Path, str]:
    """
    Finds the model file to evaluate.
    Returns (Path, source_description).
    """
    agent_dir = ROOT_DIR / "agent_code" / agent_name
    if not agent_dir.is_dir():
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")

    # 1. Explicit user request
    if requested_file:
        cand = Path(requested_file)
        if cand.is_file():
            return cand.resolve(), f"Explicit file: {cand}"
        cand2 = ROOT_DIR / requested_file
        if cand2.is_file():
            return cand2.resolve(), f"Explicit relative to root: {cand2}"
        cand3 = agent_dir / requested_file
        if cand3.is_file():
            return cand3.resolve(), f"Explicit relative to agent: {cand3}"
        raise FileNotFoundError(f"Requested model file not found: {requested_file}")

    # 2. Check run directories for best Stage 4 vs rule_based
    runs_dir = agent_dir / "runs"
    if runs_dir.is_dir():
        run_folders = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )
        for rf in run_folders:
            s4_best = rf / "best_model_s4_vs_rule_based.pt"
            if s4_best.is_file():
                return s4_best.resolve(), f"Run {rf.name} (best Stage 4 vs rule_based)"

    # 3. Agent root best checkpoint or saved model
    s4_root = agent_dir / "best_model_s4_vs_rule_based.pt"
    if s4_root.is_file():
        return s4_root.resolve(), "Agent root best Stage 4 checkpoint"

    for fname in ["my-saved-model-spatial-qwm.pt", "my-saved-model-spatial-dqn.pt", "my-saved-model-dqn.pt", "my-saved-model.pt"]:
        model_p = agent_dir / fname
        if model_p.is_file():
            return model_p.resolve(), f"Agent root saved model: {fname}"

    # 4. Check run directories for other checkpoints
    if runs_dir.is_dir():
        for rf in run_folders:
            best_pts = sorted(
                list(rf.glob("best_model_*.pt")),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if best_pts:
                return best_pts[0].resolve(), f"Run {rf.name} ({best_pts[0].name})"

        for rf in run_folders:
            for fname in ["my-saved-model-spatial-qwm.pt", "my-saved-model-spatial-dqn.pt", "my-saved-model-dqn.pt", "my-saved-model.pt"]:
                saved_model = rf / fname
                if saved_model.is_file():
                    return saved_model.resolve(), f"Run {rf.name} ({fname})"

    # 5. Config fallback
    cfg_p = agent_dir / "config.json"
    if cfg_p.is_file():
        with open(cfg_p) as fh:
            cfg = json.load(fh)
        mf = cfg.get("model_file")
        if mf and (agent_dir / mf).is_file():
            return (agent_dir / mf).resolve(), f"Agent config.json default: {mf}"

    raise FileNotFoundError(
        f"No model checkpoint (.pt) found for agent '{agent_name}'. "
        f"Please train the model first or pass --model-file <path>."
    )


# -----------------------------------------------------------------------------
# Metric Computation Engine
# -----------------------------------------------------------------------------

def compute_metrics(raw_stats: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """
    Computes aggregated KPIs, distributions, action profiles, and comparative metrics.
    """
    by_round = raw_stats.get("by_round", {})
    by_agent_lifetime = raw_stats.get("by_agent", {})

    round_ids = sorted(by_round.keys())
    n_rounds = len(round_ids)
    if n_rounds == 0:
        raise ValueError("No rounds found in statistics data.")

    # Identify all participating agents
    all_agent_names = list(by_agent_lifetime.keys())
    eval_agent_name = agent_name
    opponents = [a for a in all_agent_names if a != eval_agent_name]

    # Collect per-round metrics for all agents
    agent_round_records: Dict[str, List[Dict[str, Any]]] = {a: [] for a in all_agent_names}
    round_summary_rows = []

    total_coins_in_game = 0

    for r_idx, r_id in enumerate(round_ids, 1):
        r_data = by_round[r_id]
        r_agents = r_data.get("by_agent", {})
        r_steps = r_data.get("steps", 0)

        r_total_coins = sum(r_agents.get(a, {}).get("coins", 0) for a in all_agent_names)
        total_coins_in_game += r_total_coins

        eval_data = r_agents.get(eval_agent_name, {})
        round_summary_rows.append({
            "round": r_idx,
            "round_id": r_id,
            "steps": r_steps,
            "eval_score": eval_data.get("score", 0),
            "eval_coins": eval_data.get("coins", 0),
            "eval_kills": eval_data.get("kills", 0),
            "eval_suicides": eval_data.get("suicides", 0),
            "eval_got_killed": eval_data.get("got_killed", 0),
            "eval_steps": eval_data.get("steps", 0),
            "eval_survived": eval_data.get("survived", 0),
            "eval_won": eval_data.get("won", 0),
            "eval_bombs": eval_data.get("bombs", 0),
            "eval_waited": eval_data.get("waited", 0),
        })

        for a in all_agent_names:
            ad = r_agents.get(a, {})
            agent_round_records[a].append({
                "round": r_idx,
                "score": ad.get("score", 0),
                "coins": ad.get("coins", 0),
                "kills": ad.get("kills", 0),
                "suicides": ad.get("suicides", 0),
                "got_killed": ad.get("got_killed", 0),
                "bombs": ad.get("bombs", 0),
                "waited": ad.get("waited", 0),
                "moves": ad.get("moves", 0),
                "crates": ad.get("crates", 0),
                "invalid": ad.get("invalid", 0),
                "steps": ad.get("steps", 0),
                "survived": ad.get("survived", 0),
                "won": ad.get("won", 0),
                "dead": ad.get("dead", False),
            })

    def summarize_agent(records: List[Dict[str, Any]], life_dict: Dict[str, Any]) -> Dict[str, Any]:
        n = len(records)
        scores = np.array([r["score"] for r in records], dtype=float)
        coins = np.array([r["coins"] for r in records], dtype=float)
        kills = np.array([r["kills"] for r in records], dtype=float)
        suicides = np.array([r["suicides"] for r in records], dtype=float)
        got_killed = np.array([r["got_killed"] for r in records], dtype=float)
        bombs = np.array([r["bombs"] for r in records], dtype=float)
        waited = np.array([r["waited"] for r in records], dtype=float)
        moves = np.array([r["moves"] for r in records], dtype=float)
        crates = np.array([r["crates"] for r in records], dtype=float)
        invalid = np.array([r["invalid"] for r in records], dtype=float)
        steps = np.array([r["steps"] for r in records], dtype=float)
        survived = np.array([r["survived"] for r in records], dtype=float)
        won = np.array([r["won"] for r in records], dtype=float)

        total_actions = np.sum(moves) + np.sum(waited) + np.sum(bombs) + np.sum(invalid)
        total_deaths = np.sum(suicides) + np.sum(got_killed)

        return {
            "rounds": n,
            "wins_total": int(np.sum(won)),
            "win_rate_pct": float(np.mean(won) * 100),
            "survival_wins": int(life_dict.get("survival_wins", 0)),
            "score_wins": int(life_dict.get("score_wins", 0)),
            "survived_rounds": int(np.sum(survived)),
            "survival_rate_pct": float(np.mean(survived) * 100),
            "total_score": int(np.sum(scores)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "median_score": float(np.median(scores)),
            "min_score": int(np.min(scores)),
            "max_score": int(np.max(scores)),
            "total_coins": int(np.sum(coins)),
            "coins_per_round": float(np.mean(coins)),
            "coins_std": float(np.std(coins)),
            "coin_share_pct": float((np.sum(coins) / max(1, total_coins_in_game)) * 100),
            "total_kills": int(np.sum(kills)),
            "kills_per_round": float(np.mean(kills)),
            "kills_std": float(np.std(kills)),
            "total_suicides": int(np.sum(suicides)),
            "suicides_per_round": float(np.mean(suicides)),
            "total_got_killed": int(np.sum(got_killed)),
            "got_killed_per_round": float(np.mean(got_killed)),
            "total_deaths": int(total_deaths),
            "kdr": float(np.sum(kills) / max(1, total_deaths)),
            "total_steps": int(np.sum(steps)),
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "min_steps": int(np.min(steps)),
            "max_steps": int(np.max(steps)),
            "total_actions": int(total_actions),
            "moves_total": int(np.sum(moves)),
            "moves_per_round": float(np.mean(moves)),
            "moves_pct": float((np.sum(moves) / max(1, total_actions)) * 100),
            "waited_total": int(np.sum(waited)),
            "waited_per_round": float(np.mean(waited)),
            "waited_pct": float((np.sum(waited) / max(1, total_actions)) * 100),
            "bombs_total": int(np.sum(bombs)),
            "bombs_per_round": float(np.mean(bombs)),
            "bombs_pct": float((np.sum(bombs) / max(1, total_actions)) * 100),
            "bombs_per_100_steps": float((np.sum(bombs) / max(1, np.sum(steps))) * 100),
            "crates_total": int(np.sum(crates)),
            "crates_per_round": float(np.mean(crates)),
            "invalid_total": int(np.sum(invalid)),
            "invalid_per_round": float(np.mean(invalid)),
            "invalid_pct": float((np.sum(invalid) / max(1, total_actions)) * 100),
        }

    agent_summaries = {
        a: summarize_agent(agent_round_records[a], by_agent_lifetime.get(a, {}))
        for a in all_agent_names
    }

    # Compute Rule-Based Composite Average
    rb_avg = {}
    if opponents:
        for key in agent_summaries[eval_agent_name].keys():
            vals = [agent_summaries[op][key] for op in opponents]
            rb_avg[key] = float(np.mean(vals))
    else:
        rb_avg = agent_summaries[eval_agent_name]

    # Advantage / Delta
    eval_summary = agent_summaries[eval_agent_name]
    diffs = {
        "win_rate_delta": eval_summary["win_rate_pct"] - rb_avg["win_rate_pct"],
        "survival_rate_delta": eval_summary["survival_rate_pct"] - rb_avg["survival_rate_pct"],
        "mean_score_delta": eval_summary["mean_score"] - rb_avg["mean_score"],
        "coins_per_round_delta": eval_summary["coins_per_round"] - rb_avg["coins_per_round"],
        "kills_per_round_delta": eval_summary["kills_per_round"] - rb_avg["kills_per_round"],
        "kdr_delta": eval_summary["kdr"] - rb_avg["kdr"],
        "mean_steps_delta": eval_summary["mean_steps"] - rb_avg["mean_steps"],
        "suicides_per_round_delta": eval_summary["suicides_per_round"] - rb_avg["suicides_per_round"],
        "waited_pct_delta": eval_summary["waited_pct"] - rb_avg["waited_pct"],
    }

    return {
        "n_rounds": n_rounds,
        "total_coins_in_game": total_coins_in_game,
        "evaluated_agent_name": eval_agent_name,
        "opponents": opponents,
        "evaluated_agent": eval_summary,
        "opponents_summary": {op: agent_summaries[op] for op in opponents},
        "rule_based_avg": rb_avg,
        "deltas": diffs,
        "per_round": round_summary_rows,
        "agent_round_records": agent_round_records,
    }


# -----------------------------------------------------------------------------
# Display & Reporting
# -----------------------------------------------------------------------------

def print_evaluation_report(metrics: Dict[str, Any], model_path: Path, scenario: str,
                            source_desc: str, verbose: bool = False):
    """Prints a beautiful formatted evaluation report to stdout."""
    c = Colors(enabled=True)
    ea = metrics["evaluated_agent"]
    rb = metrics["rule_based_avg"]
    d = metrics["deltas"]
    eval_name = metrics["evaluated_agent_name"]
    n_rounds = metrics["n_rounds"]

    print("\n" + c.bold(c.cyan("=" * 78)))
    print(c.bold(c.cyan(f" BOMBERLE MODEL EVALUATION REPORT ")).center(78))
    print(c.bold(c.cyan("=" * 78)))
    print(f" {c.bold('Agent:')}        {eval_name}")
    print(f" {c.bold('Model File:')}   {model_path}")
    print(f" {c.bold('Source:')}       {source_desc}")
    opponents = metrics.get("opponents")
    opponents_str = ", ".join(opponents) if opponents else "3x rule_based_agent"
    print(f" {c.bold('Scenario:')}     {scenario}  |  {c.bold('Rounds:')} {n_rounds}  |  {c.bold('Opponents:')} {opponents_str}")
    print(f" {c.bold('Evaluated At:')} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(c.cyan("-" * 78))

    # Executive Highlights
    print("\n" + c.bold("★ EXECUTIVE PERFORMANCE SUMMARY ★"))
    kpi_rows = [
        ["Win Rate (%)", f"{ea['win_rate_pct']:.1f}%", f"{rb['win_rate_pct']:.1f}%",
         c.green(f"+{d['win_rate_delta']:.1f}%") if d['win_rate_delta'] >= 0 else c.red(f"{d['win_rate_delta']:.1f}%")],
        ["Survival Rate (%)", f"{ea['survival_rate_pct']:.1f}%", f"{rb['survival_rate_pct']:.1f}%",
         c.green(f"+{d['survival_rate_delta']:.1f}%") if d['survival_rate_delta'] >= 0 else c.red(f"{d['survival_rate_delta']:.1f}%")],
        ["Score / Round", f"{ea['mean_score']:.2f} ± {ea['std_score']:.1f}", f"{rb['mean_score']:.2f}",
         c.green(f"+{d['mean_score_delta']:.2f}") if d['mean_score_delta'] >= 0 else c.red(f"{d['mean_score_delta']:.2f}")],
        ["Coins / Round", f"{ea['coins_per_round']:.2f} ± {ea['coins_std']:.1f}", f"{rb['coins_per_round']:.2f}",
         c.green(f"+{d['coins_per_round_delta']:.2f}") if d['coins_per_round_delta'] >= 0 else c.red(f"{d['coins_per_round_delta']:.2f}")],
        ["Coin Share (%)", f"{ea['coin_share_pct']:.1f}%", f"{rb['coin_share_pct']:.1f}%",
         c.green(f"+{ea['coin_share_pct'] - rb['coin_share_pct']:.1f}%") if ea['coin_share_pct'] >= rb['coin_share_pct'] else c.red(f"{ea['coin_share_pct'] - rb['coin_share_pct']:.1f}%")],
        ["Kills / Round", f"{ea['kills_per_round']:.2f} ± {ea['kills_std']:.1f}", f"{rb['kills_per_round']:.2f}",
         c.green(f"+{d['kills_per_round_delta']:.2f}") if d['kills_per_round_delta'] >= 0 else c.red(f"{d['kills_per_round_delta']:.2f}")],
        ["Kill / Death Ratio", f"{ea['kdr']:.2f}", f"{rb['kdr']:.2f}",
         c.green(f"+{d['kdr_delta']:.2f}") if d['kdr_delta'] >= 0 else c.red(f"{d['kdr_delta']:.2f}")],
        ["Steps Survived", f"{ea['mean_steps']:.1f} ± {ea['std_steps']:.0f}", f"{rb['mean_steps']:.1f}",
         c.green(f"+{d['mean_steps_delta']:.1f}") if d['mean_steps_delta'] >= 0 else c.red(f"{d['mean_steps_delta']:.1f}")],
        ["Suicides / Round", f"{ea['suicides_per_round']:.2f}", f"{rb['suicides_per_round']:.2f}",
         c.green(f"{d['suicides_per_round_delta']:.2f}") if d['suicides_per_round_delta'] <= 0 else c.red(f"+{d['suicides_per_round_delta']:.2f}")],
        ["Wait % of Actions", f"{ea['waited_pct']:.1f}%", f"{rb['waited_pct']:.1f}%",
         c.green(f"{d['waited_pct_delta']:.1f}%") if d['waited_pct_delta'] <= 0 else c.yellow(f"+{d['waited_pct_delta']:.1f}%")],
    ]
    print(render_table(
        headers=["Metric", eval_name, "Rule-Based Avg", "Delta (vs RB)"],
        rows=kpi_rows,
        aligns=['left', 'right', 'right', 'right']
    ))

    # Head-to-Head Table
    print("\n" + c.bold("★ HEAD-TO-HEAD COMPARISON (ALL AGENTS) ★"))
    h2h_headers = ["Agent", "Win %", "Surv %", "Score", "Coins", "Kills", "Suicides", "Got Killed", "Avg Steps", "Wait %"]
    h2h_rows = []

    # Evaluated agent
    h2h_rows.append([
        c.bold(c.cyan(eval_name)),
        f"{ea['win_rate_pct']:.1f}%",
        f"{ea['survival_rate_pct']:.1f}%",
        f"{ea['mean_score']:.2f}",
        f"{ea['coins_per_round']:.2f}",
        f"{ea['kills_per_round']:.2f}",
        str(ea['total_suicides']),
        str(ea['total_got_killed']),
        f"{ea['mean_steps']:.1f}",
        f"{ea['waited_pct']:.1f}%"
    ])

    # Opponents
    for op in metrics["opponents"]:
        od = metrics["opponents_summary"][op]
        h2h_rows.append([
            op,
            f"{od['win_rate_pct']:.1f}%",
            f"{od['survival_rate_pct']:.1f}%",
            f"{od['mean_score']:.2f}",
            f"{od['coins_per_round']:.2f}",
            f"{od['kills_per_round']:.2f}",
            str(od['total_suicides']),
            str(od['total_got_killed']),
            f"{od['mean_steps']:.1f}",
            f"{od['waited_pct']:.1f}%"
        ])

    # Rule-based Average
    h2h_rows.append([
        c.dim("Rule-Based Avg"),
        f"{rb['win_rate_pct']:.1f}%",
        f"{rb['survival_rate_pct']:.1f}%",
        f"{rb['mean_score']:.2f}",
        f"{rb['coins_per_round']:.2f}",
        f"{rb['kills_per_round']:.2f}",
        f"{rb['total_suicides']:.1f}",
        f"{rb['total_got_killed']:.1f}",
        f"{rb['mean_steps']:.1f}",
        f"{rb['waited_pct']:.1f}%"
    ])

    print(render_table(
        headers=h2h_headers,
        rows=h2h_rows,
        aligns=['left', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
    ))

    # Action Profile Table
    print("\n" + c.bold(f"★ ACTION PROFILE FOR {eval_name.upper()} ★"))
    act_headers = ["Action Type", "Total Count", "Per Round", "% of Actions"]
    act_rows = [
        ["Moves (UP/DOWN/LEFT/RIGHT)", f"{ea['moves_total']}", f"{ea['moves_per_round']:.1f}", f"{ea['moves_pct']:.1f}%"],
        ["Bombs Placed", f"{ea['bombs_total']}", f"{ea['bombs_per_round']:.1f}", f"{ea['bombs_pct']:.1f}%"],
        ["Waited (WAIT)", f"{ea['waited_total']}", f"{ea['waited_per_round']:.1f}", f"{ea['waited_pct']:.1f}%"],
        ["Invalid Actions", f"{ea['invalid_total']}", f"{ea['invalid_per_round']:.1f}", f"{ea['invalid_pct']:.1f}%"],
        ["Crates Destroyed", f"{ea['crates_total']}", f"{ea['crates_per_round']:.1f}", "-"],
        [c.bold("Total Actions Taken"), c.bold(str(ea['total_actions'])), c.bold(f"{ea['total_actions'] / n_rounds:.1f}"), c.bold("100.0%")],
    ]
    print(render_table(
        headers=act_headers,
        rows=act_rows,
        aligns=['left', 'right', 'right', 'right']
    ))

    # Verbose per-round table
    if verbose:
        print("\n" + c.bold("★ PER-ROUND BREAKDOWN ★"))
        pr_headers = ["Rnd", "Steps", "Score", "Coins", "Kills", "Suicides", "Got Killed", "Bombs", "Waited", "Won"]
        pr_rows = []
        for r in metrics["per_round"]:
            won_str = c.green("WON") if r["eval_won"] else c.dim("lost")
            pr_rows.append([
                f"#{r['round']:02d}",
                str(r["steps"]),
                str(r["eval_score"]),
                str(r["eval_coins"]),
                str(r["eval_kills"]),
                str(r["eval_suicides"]),
                str(r["eval_got_killed"]),
                str(r["eval_bombs"]),
                str(r["eval_waited"]),
                won_str,
            ])
        print(render_table(
            headers=pr_headers,
            rows=pr_rows,
            aligns=['right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'center']
        ))


def export_csv(metrics: Dict[str, Any], output_file: Path):
    """Exports a granular per-round CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round", "agent", "won", "survived", "score", "coins", "kills",
            "suicides", "got_killed", "steps", "bombs", "waited", "moves",
            "crates", "invalid"
        ])
        records = metrics["agent_round_records"]
        n_rounds = metrics["n_rounds"]
        for r_idx in range(n_rounds):
            for agent_name, recs in records.items():
                r = recs[r_idx]
                writer.writerow([
                    r["round"],
                    agent_name,
                    r["won"],
                    r["survived"],
                    r["score"],
                    r["coins"],
                    r["kills"],
                    r["suicides"],
                    r["got_killed"],
                    r["steps"],
                    r["bombs"],
                    r["waited"],
                    r["moves"],
                    r["crates"],
                    r["invalid"],
                ])


def export_markdown_report(metrics: Dict[str, Any], model_path: Path, scenario: str,
                           source_desc: str, output_file: Path):
    """Generates a complete GitHub-flavored Markdown evaluation report."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ea = metrics["evaluated_agent"]
    rb = metrics["rule_based_avg"]
    d = metrics["deltas"]
    eval_name = metrics["evaluated_agent_name"]
    n_rounds = metrics["n_rounds"]

    md = []
    md.append(f"# BombeRLe Evaluation Report: `{eval_name}`\n")
    md.append(f"- **Evaluated Model:** `{model_path}`")
    md.append(f"- **Source:** {source_desc}")
    md.append(f"- **Scenario:** `{scenario}` | **Rounds:** {n_rounds}")
    opponents = metrics.get("opponents")
    opponents_display = ", ".join(f"`{op}`" for op in opponents) if opponents else "3x `rule_based_agent`"
    md.append(f"- **Opponents:** {opponents_display}")
    md.append(f"- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    md.append("## Executive Highlights\n")
    md.append("| Metric | Evaluated Agent | Rule-Based Avg | Advantage / Delta |")
    md.append("|:---|---:|---:|---:|")
    md.append(f"| **Win Rate** | **{ea['win_rate_pct']:.1f}%** | {rb['win_rate_pct']:.1f}% | `{d['win_rate_delta']:+.1f}%` |")
    md.append(f"| **Survival Rate** | **{ea['survival_rate_pct']:.1f}%** | {rb['survival_rate_pct']:.1f}% | `{d['survival_rate_delta']:+.1f}%` |")
    md.append(f"| **Score / Round** | **{ea['mean_score']:.2f} ± {ea['std_score']:.1f}** | {rb['mean_score']:.2f} | `{d['mean_score_delta']:+.2f}` |")
    md.append(f"| **Coins / Round** | **{ea['coins_per_round']:.2f} ± {ea['coins_std']:.1f}** | {rb['coins_per_round']:.2f} | `{d['coins_per_round_delta']:+.2f}` |")
    md.append(f"| **Coin Share** | **{ea['coin_share_pct']:.1f}%** | {rb['coin_share_pct']:.1f}% | `{ea['coin_share_pct'] - rb['coin_share_pct']:+.1f}%` |")
    md.append(f"| **Kills / Round** | **{ea['kills_per_round']:.2f} ± {ea['kills_std']:.1f}** | {rb['kills_per_round']:.2f} | `{d['kills_per_round_delta']:+.2f}` |")
    md.append(f"| **Kill / Death Ratio (KDR)** | **{ea['kdr']:.2f}** | {rb['kdr']:.2f} | `{d['kdr_delta']:+.2f}` |")
    md.append(f"| **Steps Survived** | **{ea['mean_steps']:.1f} ± {ea['std_steps']:.0f}** | {rb['mean_steps']:.1f} | `{d['mean_steps_delta']:+.1f}` |")
    md.append(f"| **Suicides / Round** | **{ea['suicides_per_round']:.2f}** | {rb['suicides_per_round']:.2f} | `{d['suicides_per_round_delta']:+.2f}` |")
    md.append(f"| **Wait % of Actions** | **{ea['waited_pct']:.1f}%** | {rb['waited_pct']:.1f}% | `{d['waited_pct_delta']:+.1f}%` |\n")

    md.append("## Head-to-Head Comparison\n")
    md.append("| Agent | Win % | Surv % | Score | Coins | Kills | Suicides | Got Killed | Avg Steps | Wait % |")
    md.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    md.append(f"| **{eval_name}** | **{ea['win_rate_pct']:.1f}%** | {ea['survival_rate_pct']:.1f}% | {ea['mean_score']:.2f} | {ea['coins_per_round']:.2f} | {ea['kills_per_round']:.2f} | {ea['total_suicides']} | {ea['total_got_killed']} | {ea['mean_steps']:.1f} | {ea['waited_pct']:.1f}% |")

    for op in metrics["opponents"]:
        od = metrics["opponents_summary"][op]
        md.append(f"| {op} | {od['win_rate_pct']:.1f}% | {od['survival_rate_pct']:.1f}% | {od['mean_score']:.2f} | {od['coins_per_round']:.2f} | {od['kills_per_round']:.2f} | {od['total_suicides']} | {od['total_got_killed']} | {od['mean_steps']:.1f} | {od['waited_pct']:.1f}% |")

    md.append(f"| *Rule-Based Avg* | *{rb['win_rate_pct']:.1f}%* | *{rb['survival_rate_pct']:.1f}%* | *{rb['mean_score']:.2f}* | *{rb['coins_per_round']:.2f}* | *{rb['kills_per_round']:.2f}* | *{rb['total_suicides']:.1f}* | *{rb['total_got_killed']:.1f}* | *{rb['mean_steps']:.1f}* | *{rb['waited_pct']:.1f}%* |\n")

    md.append(f"## Action Profile for `{eval_name}`\n")
    md.append("| Action Type | Total Count | Per Round | % of Actions |")
    md.append("|:---|---:|---:|---:|")
    md.append(f"| Moves (UP/DOWN/LEFT/RIGHT) | {ea['moves_total']} | {ea['moves_per_round']:.1f} | {ea['moves_pct']:.1f}% |")
    md.append(f"| Bombs Placed | {ea['bombs_total']} | {ea['bombs_per_round']:.1f} | {ea['bombs_pct']:.1f}% |")
    md.append(f"| Waited (WAIT) | {ea['waited_total']} | {ea['waited_per_round']:.1f} | {ea['waited_pct']:.1f}% |")
    md.append(f"| Invalid Actions | {ea['invalid_total']} | {ea['invalid_per_round']:.1f} | {ea['invalid_pct']:.1f}% |")
    md.append(f"| Crates Destroyed | {ea['crates_total']} | {ea['crates_per_round']:.1f} | - |")
    md.append(f"| **Total Actions** | **{ea['total_actions']}** | **{ea['total_actions'] / n_rounds:.1f}** | **100.0%** |\n")

    with open(output_file, "w") as f:
        f.write("\n".join(md))


# -----------------------------------------------------------------------------
# Main Evaluation Runner
# -----------------------------------------------------------------------------

def evaluate_model(
    agent_name: str = "my_spatial_dqn_agent",
    model_file: Optional[str] = None,
    n_rounds: int = 50,
    scenario: str = "classic",
    opponents: Optional[List[str]] = None,
    seed: Optional[int] = None,
    gui: bool = False,
    update_interval: float = 0.05,
    save_json: Optional[str] = None,
    save_csv: Optional[str] = None,
    save_markdown: Optional[str] = None,
    verbose: bool = False,
    silence_errors: bool = True,
    no_save: bool = False,
) -> Dict[str, Any]:
    """
    Executes an evaluation match and returns calculated metrics.
    """
    model_path, source_desc = resolve_model_path(agent_name, model_file)

    if opponents is None:
        opponents = ["rule_based_agent"] * 3

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = ROOT_DIR / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(save_json) if save_json else results_dir / f"eval_{agent_name}_{timestamp}.json"
    csv_path = Path(save_csv) if save_csv else results_dir / f"eval_{agent_name}_{timestamp}_rounds.csv"
    md_path = Path(save_markdown) if save_markdown else results_dir / f"eval_{agent_name}_{timestamp}_report.md"

    # Setup environment variables for clean evaluation
    env = os.environ.copy()
    if "qwm" in agent_name:
        env["MY_QWM_MODEL_FILE"] = str(model_path)
        env["MY_QWM_EXPLORE_UNSAFE"] = "0.0"
        env["MY_QWM_EPSILON"] = "0.0"
        env.pop("MY_WM_MODEL_FILE", None)
    else:
        env["MY_WM_MODEL_FILE"] = str(model_path)
        env["MY_WM_EXPLORE_UNSAFE"] = "0.0"
        env["MY_WM_EPSILON"] = "0.0"

    # Construct main.py play command
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_stats:
        temp_stats_file = Path(tmp_stats.name)

    cmd = [
        sys.executable,
        str(ROOT_DIR / "main.py"),
        "play",
    ]
    if opponents and opponents != ["rule_based_agent"] * 3:
        cmd.extend(["--agents", agent_name, *opponents])
    else:
        cmd.extend(["--my-agent", agent_name])

    cmd.extend([
        "--n-rounds", str(n_rounds),
        "--scenario", scenario,
        "--save-stats", str(temp_stats_file),
    ])

    if not gui:
        cmd.append("--no-gui")
    else:
        cmd.extend(["--update-interval", str(update_interval)])

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    if silence_errors:
        cmd.append("--silence-errors")

    opp_names = ", ".join(opponents) if opponents else "3 rule_based_agents"
    print(f"\n[evaluate.py] Running {n_rounds} rounds against {opp_names}...")
    print(f"[evaluate.py] Model: {model_path.name} ({source_desc})")
    print(f"[evaluate.py] Scenario: {scenario}")
    sys.stdout.flush()

    try:
        proc = subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        if temp_stats_file.exists():
            temp_stats_file.unlink()
        raise RuntimeError(f"Match execution failed with exit code {e.returncode}") from e

    # Read the match stats JSON
    if not temp_stats_file.exists() or temp_stats_file.stat().st_size == 0:
        raise RuntimeError("No statistics file generated by BombeRLe match.")

    with open(temp_stats_file) as f:
        raw_stats = json.load(f)

    temp_stats_file.unlink(missing_ok=True)

    # Compute comprehensive metrics
    metrics = compute_metrics(raw_stats, agent_name)

    # Print report
    print_evaluation_report(metrics, model_path, scenario, source_desc, verbose=verbose)

    # Save outputs if not disabled
    if not no_save:
        # Save JSON
        full_json_data = {
            "metadata": {
                "agent": agent_name,
                "model_path": str(model_path),
                "source_desc": source_desc,
                "scenario": scenario,
                "n_rounds": n_rounds,
                "timestamp": timestamp,
            },
            "metrics": {
                "evaluated_agent": metrics["evaluated_agent"],
                "rule_based_avg": metrics["rule_based_avg"],
                "opponents": metrics["opponents_summary"],
                "deltas": metrics["deltas"],
            },
            "per_round": metrics["per_round"],
            "raw_stats": raw_stats,
        }
        with open(json_path, "w") as f:
            json.dump(full_json_data, f, indent=2)

        # Save CSV
        export_csv(metrics, csv_path)

        # Save Markdown
        export_markdown_report(metrics, model_path, scenario, source_desc, md_path)

        c = Colors(enabled=True)
        print("\n" + c.bold("★ EXPORTED ARTIFACTS ★"))
        print(f"  • JSON:     {json_path}")
        print(f"  • CSV:      {csv_path}")
        print(f"  • Markdown: {md_path}\n")

    return metrics


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate BombeRLe agents against 3 rule_based_agents with rich performance metrics."
    )
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default="my_spatial_dqn_agent",
        help="Agent code directory to evaluate (default: my_spatial_dqn_agent)"
    )
    parser.add_argument(
        "--model-file", "-m",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (default: auto-detect best/latest checkpoint)"
    )
    parser.add_argument(
        "--n-rounds", "-n",
        type=int,
        default=50,
        help="Number of evaluation rounds (default: 50)"
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="classic",
        choices=["classic", "coin-heaven", "loot-crate", "empty"],
        help="Game scenario to evaluate in (default: classic)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible evaluation"
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=None,
        help="Opponent agents to play against (default: 3x rule_based_agent)"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Render GUI during evaluation"
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=0.05,
        help="GUI step interval in seconds (default: 0.05)"
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Custom path for JSON metrics output"
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Custom path for per-round CSV output"
    )
    parser.add_argument(
        "--save-markdown", "--markdown",
        type=str,
        default=None,
        help="Custom path for Markdown report output"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Disable writing results to disk"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show per-round details in console"
    )
    parser.add_argument(
        "--silence-errors",
        action="store_true",
        default=True,
        help="Silence agent exceptions so evaluation can complete"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    evaluate_model(
        agent_name=args.agent,
        model_file=args.model_file,
        n_rounds=args.n_rounds,
        scenario=args.scenario,
        opponents=args.opponents,
        seed=args.seed,
        gui=args.gui,
        update_interval=args.update_interval,
        save_json=args.save_json,
        save_csv=args.save_csv,
        save_markdown=args.save_markdown,
        verbose=args.verbose,
        silence_errors=args.silence_errors,
        no_save=args.no_save,
    )


if __name__ == "__main__":
    main()
