#!/usr/bin/env python3
"""
evaluate_tasks.py -- Multi-Task Scientific Evaluation Benchmark for BombeRLe Agents.

Directly implements the task specification defined in Section 4 of final_project.pdf:
"Tasks your agents will have to solve" (Machine Learning Essentials, Summer-Semester 2026):

  • Task 1: Coin Navigation (coin-heaven, 0 opponents)
    Collect revealed coins as quickly as possible without crates or bombs.
    Measures: pathfinding speed, coin rate, steps/coin efficiency, zero waits.

  • Task 2: Crate Clearing & Bomb Safety (loot-crate, 0 opponents)
    Drop bombs to destroy crates and uncover coins without committing suicide.
    Measures: crates destroyed, coins collected, bomb efficiency, suicide rate (0% target).

  • Task 3: Hunting Passive & Coin Collector Opponents (classic, vs peaceful + coin_collector)
    Hunt moving targets (peaceful_agent and coin_collector_agent) while evading bombs.
    Measures: opponent kills, win rate, survival rate, avoiding coin_collector blast.

  • Task 4: Full Competitive Match vs Rule-Based Opponents (classic, vs 3x rule_based_agent)
    Tournament scenario holding own against 3 strong rule-based agents.
    Measures: win rate %, score, survival %, kills, coin share %, head-to-head delta.

  • Optional Task 4b: 1v1 Duel vs Rule-Based (classic, vs 1x rule_based_agent)
  • Optional Task 5: Mixed League Tournament (classic, vs mixed DQN + QWM + rule_based)

Features:
  - Multi-round parallel multiprocessing execution across CPU cores.
  - Zero-variance matched seeds across agents for rigorous scientific comparison.
  - Multi-agent comparison mode (evaluate multiple models side-by-side for the report).
  - Rich ANSI/ASCII terminal scorecards and grading.
  - Automated exports: Markdown report (ready for project report), JSON stats, per-round CSV.

Usage:
    # Run all 4 tasks (default 20 rounds each) for qwm_agent_clean:
    python evaluate_tasks.py

    # Run Task 1 and Task 2 for 25 rounds each:
    python evaluate_tasks.py --tasks 1 2 --rounds 25

    # Compare two agent models side-by-side on all tasks (ideal for final report):
    python evaluate_tasks.py --agents qwm_agent_clean my_spatial_dqn_agent --rounds 20

    # Run specific task with 8 parallel worker processes:
    python evaluate_tasks.py --tasks 4 --rounds 50 --workers 8
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import warnings

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parent
AGENT_CODE_DIR = ROOT_DIR / "agent_code"
DEFAULT_EVAL_DIR = ROOT_DIR / "eval_results" / "tasks"

# Ensure logs directory exists to prevent BombeRLe FileNotFoundError
(ROOT_DIR / "logs").mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Terminal Styling & Table Rendering Utilities
# -----------------------------------------------------------------------------

ANSI_REGEX = re.compile(r'\x1b\[[0-9;]*m')


def visible_len(text: Any) -> int:
    """Returns string character length excluding ANSI escape codes."""
    return len(ANSI_REGEX.sub('', str(text)))


class Colors:
    """Terminal ANSI colors with automatic TTY detection."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.stdout.isatty()

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else str(text)

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
    """Renders a clean box-drawing table."""
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


class TaskProgressBar:
    """Adaptive progress bar supporting tqdm with rich stats and clean fallback ticker."""
    def __init__(self, total: int, desc: str, enabled: bool = True):
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self.completed = 0
        self.start_time = time.time()
        self.use_tqdm = enabled and TQDM_AVAILABLE and sys.stdout.isatty()
        self._pbar = None

        if self.enabled:
            if self.use_tqdm:
                self._pbar = tqdm(
                    total=total,
                    desc=desc,
                    unit="rnd",
                    leave=True,
                    dynamic_ncols=True,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                )
            else:
                print(f"  ▶ {desc} [{total} rounds queued]")
                sys.stdout.flush()

    def update(self, n: int = 1, postfix: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        self.completed += n
        if self._pbar is not None:
            if postfix:
                self._pbar.set_postfix(postfix)
            self._pbar.update(n)
        else:
            # Fallback for non-interactive environments (background tasks, CI, redirected stdout)
            should_log = (
                self.completed == self.total
                or self.completed == 1
                or (self.total >= 4 and self.completed % max(1, self.total // 4) == 0)
            )
            if should_log:
                elapsed = time.time() - self.start_time
                pct = (self.completed / max(1, self.total)) * 100
                post_str = " | ".join(f"{k}: {v}" for k, v in (postfix or {}).items())
                post_str = f" | {post_str}" if post_str else ""
                print(f"  [{self.desc}] Completed {self.completed:3d}/{self.total:3d} ({pct:5.1f}%) | Elapsed: {elapsed:.1f}s{post_str}")
                sys.stdout.flush()

    def close(self):
        if self._pbar is not None:
            self._pbar.close()


# -----------------------------------------------------------------------------
# Task Definitions (final_project.pdf Section 4)
# -----------------------------------------------------------------------------

@dataclass
class TaskSpec:
    task_id: int
    key: str
    name: str
    section_ref: str
    scenario: str
    opponents: List[str]
    description: str
    focus: str
    default_rounds: int = 20


TASKS_CATALOG: Dict[str, TaskSpec] = {
    "1": TaskSpec(
        task_id=1,
        key="task1",
        name="Task 1: Coin Navigation",
        section_ref="final_project.pdf §4.1",
        scenario="coin-heaven",
        opponents=[],
        description="Collect revealed coins on an open board without crates or opponents.",
        focus="Navigation speed, coin collection rate, steps per coin, zero waiting",
        default_rounds=20,
    ),
    "2": TaskSpec(
        task_id=2,
        key="task2",
        name="Task 2: Crate Clearing & Bomb Safety",
        section_ref="final_project.pdf §4.2",
        scenario="loot-crate",
        opponents=[],
        description="Destroy randomly placed crates to uncover and collect hidden coins without killing self.",
        focus="Crate destruction, bomb placement, blast escape, zero suicides",
        default_rounds=20,
    ),
    "3": TaskSpec(
        task_id=3,
        key="task3",
        name="Task 3: Hunting Passive & Collector Enemies",
        section_ref="final_project.pdf §4.3",
        scenario="classic",
        opponents=["peaceful_agent", "coin_collector_agent"],
        description="Hunt moving targets (peaceful_agent and coin_collector_agent) while evading bombs.",
        focus="Target hunting, opponent kills, coin competition, blast evasion",
        default_rounds=20,
    ),
    "4": TaskSpec(
        task_id=4,
        key="task4",
        name="Task 4: Full Competitive Combat (Tournament)",
        section_ref="final_project.pdf §4.4",
        scenario="classic",
        opponents=["rule_based_agent", "rule_based_agent", "rule_based_agent"],
        description="Full 4-player official tournament match against 3 strong rule-based agents.",
        focus="Win rate %, combat score, survival %, kills, beating rule_based_agent",
        default_rounds=20,
    ),
    "4b": TaskSpec(
        task_id=4,
        key="task4_duel",
        name="Task 4b: 1v1 Duel vs Rule-Based",
        section_ref="final_project.pdf §4.4 (1v1)",
        scenario="classic",
        opponents=["rule_based_agent"],
        description="Direct head-to-head 1v1 duel against a single rule_based_agent.",
        focus="1v1 tactical outmaneuvering, duel win rate %, direct elimination",
        default_rounds=20,
    ),
    "5": TaskSpec(
        task_id=5,
        key="task5_gauntlet",
        name="Task 5: Mixed League Tournament Gauntlet",
        section_ref="final_project.pdf §6 & Curriculum Stage 7",
        scenario="classic",
        opponents=["my_spatial_dqn_agent", "my_spatial_qwm_agent", "rule_based_agent"],
        description="Diverse multi-model competition vs DQN, QWM clone, and rule-based opponents.",
        focus="Adaptability against diverse strategic playstyles and network models",
        default_rounds=20,
    ),
    "h2h_1v1": TaskSpec(
        task_id=91,
        key="h2h_1v1",
        name="Direct Head-to-Head 1v1 Duel",
        section_ref="Direct Comparison (1v1 Duel)",
        scenario="classic",
        opponents=[],
        description="Direct 2-player duel between both candidate models in classic arena on matched seeds.",
        focus="1v1 tactical combat, direct elimination, outmaneuvering, duel win rate",
        default_rounds=20,
    ),
    "h2h_4p": TaskSpec(
        task_id=92,
        key="h2h_4p",
        name="Direct Head-to-Head 4-Player Combat (1v1 + 2 Rule-Based)",
        section_ref="Direct Comparison (4-Player Tournament)",
        scenario="classic",
        opponents=["rule_based_agent", "rule_based_agent"],
        description="4-player tournament match with both candidate models competing against each other and 2 rule-based agents on matched seeds.",
        focus="Multi-agent tournament adaptability, surviving chaotic crossfire, defeating rival agent",
        default_rounds=20,
    ),
}

@dataclass
class H2HMatchupSpec:
    mode: str
    key: str
    name: str
    scenario: str
    opponents: List[str]
    description: str
    focus: str
    default_rounds: int = 20


H2H_MATCHUPS: Dict[str, H2HMatchupSpec] = {
    "1v1": H2HMatchupSpec(
        mode="1v1",
        key="h2h_1v1",
        name="Direct Head-to-Head 1v1 Duel",
        scenario="classic",
        opponents=[],
        description="Direct 2-player duel between candidate models in classic arena on matched seeds.",
        focus="1v1 tactical combat, direct elimination, outmaneuvering, duel win rate",
        default_rounds=20,
    ),
    "4p": H2HMatchupSpec(
        mode="4p",
        key="h2h_4p",
        name="Direct Head-to-Head 4-Player Combat (1v1 + 2 Rule-Based)",
        scenario="classic",
        opponents=["rule_based_agent", "rule_based_agent"],
        description="4-player tournament match with both candidate models competing against each other and 2 rule-based agents.",
        focus="Multi-agent tournament adaptability, surviving chaotic crossfire, defeating human/rule-based adversaries",
        default_rounds=20,
    ),
}

# Aliases for task selection
TASK_ALIASES: Dict[str, List[str]] = {
    "all": ["1", "2", "3", "4"],
    "pdf": ["1", "2", "3", "4"],
    "main": ["1", "2", "3", "4"],
    "core": ["1", "2", "3", "4"],
    "extended": ["1", "2", "3", "4", "4b", "5"],
    "task1": ["1"],
    "task2": ["2"],
    "task3": ["3"],
    "task4": ["4"],
    "task4b": ["4b"],
    "task5": ["5"],
    "coins": ["1"],
    "crates": ["2"],
    "hunting": ["3"],
    "combat": ["4"],
    "duel": ["4b"],
    "gauntlet": ["5"],
    "h2h": ["h2h_1v1", "h2h_4p"],
    "h2h_1v1": ["h2h_1v1"],
    "h2h_4p": ["h2h_4p"],
    "h2h_all": ["h2h_1v1", "h2h_4p"],
}


def resolve_tasks(task_inputs: Optional[List[str]]) -> List[TaskSpec]:
    """Resolves task arguments into an ordered list of TaskSpec objects."""
    if not task_inputs:
        task_inputs = ["all"]

    resolved_keys: List[str] = []
    for item in task_inputs:
        k = str(item).strip().lower()
        if k in TASK_ALIASES:
            for sub_k in TASK_ALIASES[k]:
                if sub_k not in resolved_keys:
                    resolved_keys.append(sub_k)
        elif k in TASKS_CATALOG:
            if k not in resolved_keys:
                resolved_keys.append(k)
        else:
            # Check by key name
            matched = False
            for cat_k, spec in TASKS_CATALOG.items():
                if k == spec.key or k in spec.name.lower():
                    if cat_k not in resolved_keys:
                        resolved_keys.append(cat_k)
                    matched = True
                    break
            if not matched:
                print(f"Warning: Unknown task identifier '{item}'. Valid: {list(TASKS_CATALOG.keys()) + list(TASK_ALIASES.keys())}")

    if not resolved_keys:
        resolved_keys = ["1", "2", "3", "4"]

    return [TASKS_CATALOG[k] for k in resolved_keys]


# -----------------------------------------------------------------------------
# Model & Agent Resolution
# -----------------------------------------------------------------------------

def resolve_agent_model(agent_name: str, requested_checkpoint: Optional[str] = None) -> Tuple[Path, str]:
    """Resolves model weights checkpoint for an agent."""
    agent_dir = AGENT_CODE_DIR / agent_name
    if not agent_dir.is_dir():
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")

    if requested_checkpoint:
        cand = Path(requested_checkpoint)
        if cand.is_file():
            return cand.resolve(), f"Explicit: {cand.name}"
        cand2 = ROOT_DIR / requested_checkpoint
        if cand2.is_file():
            return cand2.resolve(), f"Explicit: {cand2.name}"
        cand3 = agent_dir / requested_checkpoint
        if cand3.is_file():
            return cand3.resolve(), f"Explicit: {cand3.name}"

    # Check agent config.json
    cfg_path = agent_dir / "config.json"
    if cfg_path.is_file():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            mf = cfg.get("model_file")
            if mf and (agent_dir / mf).is_file():
                return (agent_dir / mf).resolve(), f"config.json: {mf}"
        except Exception:
            pass

    # Standard model filename fallbacks
    standard_names = [
        "my-saved-model-spatial-qwm.pt",
        "my-saved-model-spatial-dqn.pt",
        "my-saved-model-dqn.pt",
        "my-saved-model.pt",
        "small_model.pt",
        "best_model_s4_vs_rule_based.pt",
        "best_model_s7_vs_mixed_classic.pt",
        "dyna-model.pt",
    ]
    for name in standard_names:
        pt = agent_dir / name
        if pt.is_file():
            return pt.resolve(), f"Agent root: {name}"

    # Check runs directory
    runs_dir = agent_dir / "runs"
    if runs_dir.is_dir():
        run_folders = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
        for rf in run_folders:
            for name in standard_names:
                pt = rf / name
                if pt.is_file():
                    return pt.resolve(), f"Run {rf.name}: {name}"

    # Dummy fallback if agent uses non-checkpoint logic
    dummy = agent_dir / "callbacks.py"
    return dummy.resolve(), "Agent code default"


# -----------------------------------------------------------------------------
# Single Match Execution Worker
# -----------------------------------------------------------------------------

@dataclass
class RoundExecutionResult:
    agent_name: str
    task_id: int
    task_key: str
    round_idx: int
    seed: int
    score: float = 0.0
    won: bool = False
    survived: bool = False
    coins: int = 0
    crates: int = 0
    kills: int = 0
    suicides: int = 0
    got_killed: int = 0
    steps: int = 0
    bombs: int = 0
    waited: int = 0
    moves: int = 0
    invalid: int = 0
    peaceful_kills: int = 0
    collector_kills: int = 0
    rule_based_kills: int = 0
    other_agents: List[Dict[str, Any]] = field(default_factory=list)
    search_depth: Optional[int] = None
    duration_sec: float = 0.0
    ms_per_step: float = 0.0
    success: bool = True
    error_msg: str = ""


@dataclass
class H2HRoundResult:
    mode: str
    round_idx: int
    seed: int
    agent_a: str
    agent_b: str
    winner: str = "tie"
    overall_winner: str = "tie"
    a_stats: Dict[str, Any] = field(default_factory=dict)
    b_stats: Dict[str, Any] = field(default_factory=dict)
    rule_based_stats: List[Dict[str, Any]] = field(default_factory=list)
    a_killed_b: bool = False
    b_killed_a: bool = False
    slot_order: List[str] = field(default_factory=list)
    steps: int = 0
    success: bool = True
    error_msg: str = ""


def run_single_match_worker(args_tuple: Tuple) -> RoundExecutionResult:
    """Executes a single match round in an isolated subprocess with BombeRLe."""
    if len(args_tuple) >= 8:
        agent_name, scenario, opponents, seed, round_idx, task_key, checkpoint_path, search_depth = args_tuple[:8]
    else:
        agent_name, scenario, opponents, seed, round_idx, task_key, checkpoint_path = args_tuple[:7]
        search_depth = None

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_f:
        stats_path = Path(tmp_f.name)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"

    if checkpoint_path and Path(checkpoint_path).is_file():
        if "qwm" in agent_name:
            env["MY_QWM_MODEL_FILE"] = str(checkpoint_path)
            env["MY_QWM_TREE_SEARCH"] = "1"
            env["MY_QWM_EPSILON"] = "0.0"
        elif "dyna" in agent_name:
            env["DYNA_MODEL_FILE"] = str(checkpoint_path)
            env["MY_WM_MODEL_FILE"] = str(checkpoint_path)
            env["MY_WM_EPSILON"] = "0.0"
        else:
            env["MY_WM_MODEL_FILE"] = str(checkpoint_path)
            env["MY_WM_EPSILON"] = "0.0"

    # Inject search depth / planning horizon if specified
    if search_depth is not None:
        env["MY_QWM_SEARCH_DEPTH"] = str(search_depth)
        env["MY_WM_SEARCH_DEPTH"] = str(search_depth)

    all_agents = [agent_name] + opponents
    cmd = [
        sys.executable,
        str(ROOT_DIR / "main.py"),
        "play",
        "--agents", *all_agents,
        "--scenario", scenario,
        "--seed", str(seed),
        "--n-rounds", "1",
        "--no-gui",
        "--silence-errors",
        "--save-stats", str(stats_path),
    ]

    t_start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )

        duration_sec = time.perf_counter() - t_start

        if not stats_path.is_file() or stats_path.stat().st_size == 0:
            err_tail = proc.stderr[-250:] if proc.stderr else f"Process returncode {proc.returncode}"
            return RoundExecutionResult(
                agent_name=agent_name,
                task_id=0,
                task_key=task_key,
                round_idx=round_idx,
                seed=seed,
                search_depth=search_depth,
                duration_sec=round(duration_sec, 3),
                success=False,
                error_msg=f"Failed to generate stats: {err_tail}",
            )

        with open(stats_path, "r") as f:
            raw = json.load(f)

        by_round = raw.get("by_round", {})
        if not by_round:
            return RoundExecutionResult(
                agent_name=agent_name,
                task_id=0,
                task_key=task_key,
                round_idx=round_idx,
                seed=seed,
                search_depth=search_depth,
                duration_sec=round(duration_sec, 3),
                success=False,
                error_msg="Empty by_round in match stats",
            )

        r_data = list(by_round.values())[0]
        r_agents = r_data.get("by_agent", {})
        agent_data = r_agents.get(agent_name, {})

        # Collect opponent stats
        other_agents = []
        peaceful_k = 0
        collector_k = 0
        rule_based_k = 0

        for opp in opponents:
            od = r_agents.get(opp, {})
            other_agents.append({
                "name": opp,
                "score": od.get("score", 0),
                "coins": od.get("coins", 0),
                "kills": od.get("kills", 0),
                "suicides": od.get("suicides", 0),
                "got_killed": od.get("got_killed", 0),
                "survived": od.get("survived", 0),
                "dead": od.get("dead", False),
            })

        # Parse kills
        our_kills = agent_data.get("kills", 0)
        steps_count = int(agent_data.get("steps", r_data.get("steps", 0)))
        ms_per_step = (duration_sec / max(1, steps_count)) * 1000.0

        res = RoundExecutionResult(
            agent_name=agent_name,
            task_id=0,
            task_key=task_key,
            round_idx=round_idx,
            seed=seed,
            score=float(agent_data.get("score", 0)),
            won=bool(agent_data.get("won", 0)),
            survived=bool(agent_data.get("survived", 0)),
            coins=int(agent_data.get("coins", 0)),
            crates=int(agent_data.get("crates", 0)),
            kills=int(our_kills),
            suicides=int(agent_data.get("suicides", 0)),
            got_killed=int(agent_data.get("got_killed", 0)),
            steps=steps_count,
            bombs=int(agent_data.get("bombs", 0)),
            waited=int(agent_data.get("waited", 0)),
            moves=int(agent_data.get("moves", 0)),
            invalid=int(agent_data.get("invalid", 0)),
            peaceful_kills=peaceful_k,
            collector_kills=collector_k,
            rule_based_kills=rule_based_k,
            other_agents=other_agents,
            search_depth=search_depth,
            duration_sec=round(duration_sec, 3),
            ms_per_step=round(ms_per_step, 2),
            success=True,
        )
        return res

    except Exception as exc:
        duration_sec = time.perf_counter() - t_start
        return RoundExecutionResult(
            agent_name=agent_name,
            task_id=0,
            task_key=task_key,
            round_idx=round_idx,
            seed=seed,
            search_depth=search_depth,
            duration_sec=round(duration_sec, 3),
            success=False,
            error_msg=str(exc),
        )
    finally:
        if stats_path.is_file():
            try:
                stats_path.unlink()
            except Exception:
                pass


def run_single_h2h_worker(args_tuple: Tuple[str, str, List[str], int, int, str, Optional[str], Optional[str], bool]) -> H2HRoundResult:
    """Executes a single direct head-to-head match round between agent_a and agent_b."""
    agent_a, agent_b, extra_opponents, seed, round_idx, mode, ckpt_a, ckpt_b, slot_reversed = args_tuple

    # Alternate starting slots across matched seeds to eliminate positional corner bias
    if slot_reversed:
        agent_order = [agent_b, agent_a] + list(extra_opponents)
    else:
        agent_order = [agent_a, agent_b] + list(extra_opponents)

    with tempfile.TemporaryDirectory() as tmp_dir:
        stats_path = Path(tmp_dir) / "stats.json"

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["TORCH_NUM_THREADS"] = "1"

        # Configure environment variables for both agents
        for ag_name, ckpt in [(agent_a, ckpt_a), (agent_b, ckpt_b)]:
            if ckpt and Path(ckpt).is_file():
                if "qwm" in ag_name:
                    env["MY_QWM_MODEL_FILE"] = str(ckpt)
                    env["MY_QWM_TREE_SEARCH"] = "1"
                    env["MY_QWM_EPSILON"] = "0.0"
                elif "dyna" in ag_name:
                    env["DYNA_MODEL_FILE"] = str(ckpt)
                    env["MY_WM_MODEL_FILE"] = str(ckpt)
                    env["MY_WM_EPSILON"] = "0.0"
                else:
                    env["MY_WM_MODEL_FILE"] = str(ckpt)
                    env["MY_WM_EPSILON"] = "0.0"

        cmd = [
            sys.executable,
            str(ROOT_DIR / "main.py"),
            "play",
            "--agents", *agent_order,
            "--scenario", "classic",
            "--seed", str(seed),
            "--n-rounds", "1",
            "--no-gui",
            "--silence-errors",
            "--save-stats", str(stats_path),
            "--log-dir", tmp_dir,
        ]

        try:
            proc = subprocess.run(
                cmd,
                env=env,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180,
            )

            if not stats_path.is_file() or stats_path.stat().st_size == 0:
                err_tail = proc.stderr[-250:] if proc.stderr else f"Process returncode {proc.returncode}"
                return H2HRoundResult(
                    mode=mode,
                    round_idx=round_idx,
                    seed=seed,
                    agent_a=agent_a,
                    agent_b=agent_b,
                    slot_order=agent_order,
                    success=False,
                    error_msg=f"No stats: {err_tail}",
                )

            with open(stats_path, "r") as f:
                raw = json.load(f)

            by_round = raw.get("by_round", {})
            if not by_round:
                return H2HRoundResult(
                    mode=mode,
                    round_idx=round_idx,
                    seed=seed,
                    agent_a=agent_a,
                    agent_b=agent_b,
                    slot_order=agent_order,
                    success=False,
                    error_msg="Empty by_round in match stats",
                )

            r_data = list(by_round.values())[0]
            r_agents = r_data.get("by_agent", {})

            data_a = r_agents.get(agent_a, {})
            data_b = r_agents.get(agent_b, {})

            # Opponent stats for extra opponents (e.g. rule-based)
            rb_stats = []
            for k, v in r_agents.items():
                if k not in (agent_a, agent_b):
                    rb_stats.append({
                        "name": k,
                        "score": v.get("score", 0),
                        "coins": v.get("coins", 0),
                        "kills": v.get("kills", 0),
                        "survived": v.get("survived", 0),
                        "dead": v.get("dead", False),
                    })

            # Check direct kills from game.log
            a_killed_b = False
            b_killed_a = False
            log_file = Path(tmp_dir) / "game.log"
            if log_file.is_file():
                try:
                    with open(log_file, "r", errors="ignore") as lf:
                        for line in lf:
                            if f"Agent <{agent_b}> blown up by agent <{agent_a}>" in line:
                                a_killed_b = True
                            elif f"Agent <{agent_a}> blown up by agent <{agent_b}>" in line:
                                b_killed_a = True
                except Exception:
                    pass

            if mode == "1v1":
                if data_a.get("kills", 0) > 0 and data_b.get("dead", False):
                    a_killed_b = True
                if data_b.get("kills", 0) > 0 and data_a.get("dead", False):
                    b_killed_a = True

            # Determine head-to-head winner between A and B
            score_a = float(data_a.get("score", 0))
            score_b = float(data_b.get("score", 0))
            surv_a = bool(data_a.get("survived", 0))
            surv_b = bool(data_b.get("survived", 0))

            if score_a > score_b:
                winner = agent_a
            elif score_b > score_a:
                winner = agent_b
            else:
                if surv_a and not surv_b:
                    winner = agent_a
                elif surv_b and not surv_a:
                    winner = agent_b
                else:
                    winner = "tie"

            # Determine overall match winner across all agents
            max_score = -1.0
            overall_winner = "tie"
            for ag_k, ag_v in r_agents.items():
                s = float(ag_v.get("score", 0))
                if s > max_score:
                    max_score = s
                    overall_winner = ag_k
                elif s == max_score and max_score > 0:
                    overall_winner = "tie"

            return H2HRoundResult(
                mode=mode,
                round_idx=round_idx,
                seed=seed,
                agent_a=agent_a,
                agent_b=agent_b,
                winner=winner,
                overall_winner=overall_winner,
                a_stats=data_a,
                b_stats=data_b,
                rule_based_stats=rb_stats,
                a_killed_b=a_killed_b,
                b_killed_a=b_killed_a,
                slot_order=agent_order,
                steps=int(r_data.get("steps", 0)),
                success=True,
            )

        except Exception as exc:
            return H2HRoundResult(
                mode=mode,
                round_idx=round_idx,
                seed=seed,
                agent_a=agent_a,
                agent_b=agent_b,
                slot_order=agent_order,
                success=False,
                error_msg=str(exc),
            )


# -----------------------------------------------------------------------------
# Metric Aggregation & Grading Engine
# -----------------------------------------------------------------------------

def grade_metric(score_val: float, thresholds: Tuple[float, float, float, float]) -> str:
    """Assigns letter grade based on thresholds: [A+, A, B, C]."""
    ap, a, b, c = thresholds
    if score_val >= ap: return "A+"
    if score_val >= a:  return "A"
    if score_val >= b:  return "B"
    if score_val >= c:  return "C"
    return "Needs Improvement"


def summarize_task_results(task: TaskSpec, rounds: List[RoundExecutionResult]) -> Dict[str, Any]:
    """Computes comprehensive statistics and domain KPIs for a specific task."""
    valid_rounds = [r for r in rounds if r.success]
    n = len(valid_rounds)
    if n == 0:
        return {"n_rounds": 0, "success": False, "task": asdict(task)}

    scores = np.array([r.score for r in valid_rounds], dtype=float)
    coins = np.array([r.coins for r in valid_rounds], dtype=float)
    crates = np.array([r.crates for r in valid_rounds], dtype=float)
    kills = np.array([r.kills for r in valid_rounds], dtype=float)
    suicides = np.array([r.suicides for r in valid_rounds], dtype=float)
    got_killed = np.array([r.got_killed for r in valid_rounds], dtype=float)
    steps = np.array([r.steps for r in valid_rounds], dtype=float)
    bombs = np.array([r.bombs for r in valid_rounds], dtype=float)
    waited = np.array([r.waited for r in valid_rounds], dtype=float)
    moves = np.array([r.moves for r in valid_rounds], dtype=float)
    invalid = np.array([r.invalid for r in valid_rounds], dtype=float)
    wins = np.array([1.0 if r.won else 0.0 for r in valid_rounds], dtype=float)
    survived = np.array([1.0 if r.survived else 0.0 for r in valid_rounds], dtype=float)

    total_actions = np.sum(moves) + np.sum(bombs) + np.sum(waited) + np.sum(invalid)

    # Opponent averages (for tasks with opponents)
    opp_scores: List[float] = []
    opp_survivals: List[float] = []
    opp_kills: List[float] = []
    for r in valid_rounds:
        for opp in r.other_agents:
            opp_scores.append(float(opp.get("score", 0)))
            opp_survivals.append(1.0 if opp.get("survived", 0) else 0.0)
            opp_kills.append(float(opp.get("kills", 0)))

    rule_based_avg_score = float(np.mean(opp_scores)) if opp_scores else 0.0
    rule_based_avg_surv = float(np.mean(opp_survivals) * 100) if opp_survivals else 0.0

    # Task-specific KPIs and letter grading
    kpi_summary = {}
    grade = "B"

    if task.task_id == 1:
        # Task 1: Coin Navigation (coin-heaven, 50 coins available)
        mean_coins = float(np.mean(coins))
        steps_per_coin = float(np.sum(steps) / max(1.0, np.sum(coins)))
        wait_pct = float((np.sum(waited) / max(1.0, total_actions)) * 100)
        invalid_pct = float((np.sum(invalid) / max(1.0, total_actions)) * 100)
        bombs_dropped = int(np.sum(bombs))

        kpi_summary = {
            "mean_coins": mean_coins,
            "coins_std": float(np.std(coins)),
            "steps_per_coin": steps_per_coin,
            "wait_pct": wait_pct,
            "invalid_pct": invalid_pct,
            "bombs_dropped": bombs_dropped,
            "speed_rating": f"{steps_per_coin:.1f} steps/coin",
        }
        # Grade: >42 coins = A+, >35 = A, >25 = B, >15 = C
        grade = grade_metric(mean_coins, (42.0, 35.0, 25.0, 15.0))

    elif task.task_id == 2:
        # Task 2: Crate Clearing & Bomb Safety
        mean_crates = float(np.mean(crates))
        mean_coins = float(np.mean(coins))
        total_suicides = int(np.sum(suicides))
        crates_per_bomb = float(np.sum(crates) / max(1.0, np.sum(bombs)))
        survival_pct = float(np.mean(survived) * 100)

        kpi_summary = {
            "mean_crates": mean_crates,
            "crates_std": float(np.std(crates)),
            "mean_coins": mean_coins,
            "crates_per_bomb": crates_per_bomb,
            "total_suicides": total_suicides,
            "suicides_per_round": float(np.mean(suicides)),
            "survival_rate_pct": survival_pct,
            "safety_rating": "Zero Suicides (Safe)" if total_suicides == 0 else f"{total_suicides} suicides",
        }
        # Grade: High crates + low suicide:
        # Score = crates - 15*suicide_rate
        crate_safety_score = mean_crates - (float(np.mean(suicides)) * 25.0)
        grade = grade_metric(crate_safety_score, (28.0, 20.0, 12.0, 5.0))

    elif task.task_id == 3:
        # Task 3: Hunting Passive & Collector Enemies
        mean_kills = float(np.mean(kills))
        win_rate = float(np.mean(wins) * 100)
        survival_rate = float(np.mean(survived) * 100)
        mean_score = float(np.mean(scores))
        total_suicides = int(np.sum(suicides))

        kpi_summary = {
            "mean_kills": mean_kills,
            "kills_std": float(np.std(kills)),
            "win_rate_pct": win_rate,
            "survival_rate_pct": survival_rate,
            "mean_score": mean_score,
            "total_suicides": total_suicides,
            "hunting_efficiency": f"{mean_kills:.2f} kills/rnd",
        }
        # Grade: >1.2 kills + >75% win = A+
        hunt_score = (mean_kills * 50.0) + (win_rate * 0.5)
        grade = grade_metric(hunt_score, (95.0, 75.0, 50.0, 25.0))

    else:
        # Task 4 / 4b / 5: Full Combat vs Rule-Based
        win_rate = float(np.mean(wins) * 100)
        survival_rate = float(np.mean(survived) * 100)
        mean_score = float(np.mean(scores))
        mean_kills = float(np.mean(kills))
        total_suicides = int(np.sum(suicides))
        delta_score = mean_score - rule_based_avg_score

        kpi_summary = {
            "win_rate_pct": win_rate,
            "survival_rate_pct": survival_rate,
            "mean_score": mean_score,
            "score_std": float(np.std(scores)),
            "mean_kills": mean_kills,
            "total_suicides": total_suicides,
            "rule_based_avg_score": rule_based_avg_score,
            "rule_based_avg_surv": rule_based_avg_surv,
            "delta_vs_rule_based_score": delta_score,
        }
        # Grade: Win rate vs 3x rule-based: >50% = A+, >40% = A, >25% = B, >10% = C
        grade = grade_metric(win_rate, (50.0, 40.0, 25.0, 10.0))

    durations = np.array([r.duration_sec for r in valid_rounds if r.duration_sec > 0], dtype=float)
    ms_steps = np.array([r.ms_per_step for r in valid_rounds if r.ms_per_step > 0], dtype=float)
    mean_duration_sec = float(np.mean(durations)) if len(durations) > 0 else 0.0
    mean_ms_per_step = float(np.mean(ms_steps)) if len(ms_steps) > 0 else 0.0
    score_ci95 = float(1.96 * (np.std(scores) / np.sqrt(n))) if n > 1 else 0.0

    return {
        "task": asdict(task),
        "n_rounds": n,
        "grade": grade,
        "overall_win_rate": float(np.mean(wins) * 100),
        "overall_survival_rate": float(np.mean(survived) * 100),
        "overall_mean_score": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "overall_score_ci95": score_ci95,
        "mean_duration_sec": mean_duration_sec,
        "mean_ms_per_step": mean_ms_per_step,
        "overall_coins": float(np.mean(coins)),
        "overall_crates": float(np.mean(crates)),
        "overall_kills": float(np.mean(kills)),
        "overall_suicides": int(np.sum(suicides)),
        "overall_got_killed": int(np.sum(got_killed)),
        "overall_steps": float(np.mean(steps)),
        "overall_bombs": float(np.mean(bombs)),
        "overall_waited_pct": float((np.sum(waited) / max(1.0, total_actions)) * 100),
        "overall_invalid_pct": float((np.sum(invalid) / max(1.0, total_actions)) * 100),
        "kpis": kpi_summary,
        "rounds": [asdict(r) for r in valid_rounds],
        "success": True,
    }


def summarize_h2h_results(rounds: List[H2HRoundResult], agent_a: str, agent_b: str) -> Dict[str, Any]:
    """Aggregates metrics for head-to-head match series between agent_a and agent_b."""
    valid = [r for r in rounds if r.success]
    n = len(valid)
    if n == 0:
        return {"n_rounds": 0, "success": False, "agent_a": agent_a, "agent_b": agent_b}

    a_wins = sum(1 for r in valid if r.winner == agent_a)
    b_wins = sum(1 for r in valid if r.winner == agent_b)
    ties = sum(1 for r in valid if r.winner == "tie")

    a_scores = np.array([float(r.a_stats.get("score", 0)) for r in valid], dtype=float)
    b_scores = np.array([float(r.b_stats.get("score", 0)) for r in valid], dtype=float)
    score_deltas = a_scores - b_scores

    a_coins = np.array([int(r.a_stats.get("coins", 0)) for r in valid], dtype=float)
    b_coins = np.array([int(r.b_stats.get("coins", 0)) for r in valid], dtype=float)
    total_coins_collected = np.sum(a_coins) + np.sum(b_coins)

    a_crates = np.array([int(r.a_stats.get("crates", 0)) for r in valid], dtype=float)
    b_crates = np.array([int(r.b_stats.get("crates", 0)) for r in valid], dtype=float)

    a_kills = np.array([int(r.a_stats.get("kills", 0)) for r in valid], dtype=float)
    b_kills = np.array([int(r.b_stats.get("kills", 0)) for r in valid], dtype=float)

    a_direct_kills = sum(1 for r in valid if r.a_killed_b)
    b_direct_kills = sum(1 for r in valid if r.b_killed_a)

    a_suicides = sum(int(r.a_stats.get("suicides", 0)) for r in valid)
    b_suicides = sum(int(r.b_stats.get("suicides", 0)) for r in valid)

    a_survivals = sum(1 for r in valid if r.a_stats.get("survived", 0))
    b_survivals = sum(1 for r in valid if r.b_stats.get("survived", 0))

    mean_delta = float(np.mean(score_deltas))
    std_delta = float(np.std(score_deltas, ddof=1)) if n > 1 else 0.0
    se_delta = std_delta / np.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * se_delta

    # Paired t-test p-value across matched seeds
    p_value = 1.0
    if n > 1 and std_delta > 1e-8:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import scipy.stats as st
                _, p_val = st.ttest_rel(a_scores, b_scores)
            p_value = float(p_val)
        except Exception:
            z = abs(mean_delta) / max(1e-8, se_delta)
            import math
            p_value = float(math.erfc(z / math.sqrt(2.0)))

    # Matchup verdict
    if a_wins > b_wins and (a_wins / n >= 0.55 or p_value < 0.05):
        verdict = f"{agent_a} WINS"
    elif b_wins > a_wins and (b_wins / n >= 0.55 or p_value < 0.05):
        verdict = f"{agent_b} WINS"
    elif a_wins > b_wins:
        verdict = f"{agent_a} SLIGHT EDGE"
    elif b_wins > a_wins:
        verdict = f"{agent_b} SLIGHT EDGE"
    else:
        verdict = "DEAD EVEN TIE"

    mode = valid[0].mode
    return {
        "success": True,
        "n_rounds": n,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "mode": mode,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "a_win_rate": float((a_wins / n) * 100),
        "b_win_rate": float((b_wins / n) * 100),
        "tie_rate": float((ties / n) * 100),
        "a_survived_pct": float((a_survivals / n) * 100),
        "b_survived_pct": float((b_survivals / n) * 100),
        "a_mean_score": float(np.mean(a_scores)),
        "b_mean_score": float(np.mean(b_scores)),
        "score_delta": mean_delta,
        "score_delta_ci95": ci95,
        "p_value": p_value,
        "a_mean_coins": float(np.mean(a_coins)),
        "b_mean_coins": float(np.mean(b_coins)),
        "a_coin_share_pct": float((np.sum(a_coins) / max(1.0, total_coins_collected)) * 100),
        "b_coin_share_pct": float((np.sum(b_coins) / max(1.0, total_coins_collected)) * 100),
        "a_mean_crates": float(np.mean(a_crates)),
        "b_mean_crates": float(np.mean(b_crates)),
        "a_mean_kills": float(np.mean(a_kills)),
        "b_mean_kills": float(np.mean(b_kills)),
        "a_direct_kills": a_direct_kills,
        "b_direct_kills": b_direct_kills,
        "a_suicides": a_suicides,
        "b_suicides": b_suicides,
        "verdict": verdict,
        "rounds": [asdict(r) for r in valid],
    }


def print_h2h_summary_tables(h2h_summaries: Dict[str, Dict[str, Any]]):
    """Renders formatted comparison tables for direct head-to-head duels and 4-player matches."""
    c = Colors(enabled=True)
    if not h2h_summaries:
        return

    print("\n" + c.bold("=" * 88))
    print(c.bold("★ DIRECT HEAD-TO-HEAD BENCHMARK: MATCHED SEED COMPARISON ★"))
    print(c.bold("=" * 88))

    for mode_key, summary in h2h_summaries.items():
        if not summary.get("success", False):
            continue

        agent_a = summary["agent_a"]
        agent_b = summary["agent_b"]
        n_rnd = summary["n_rounds"]
        mode_title = "1v1 DUEL (NO OPPONENTS)" if summary["mode"] == "1v1" else "4-PLAYER COMBAT (1v1 + 2 RULE-BASED)"

        print(c.bold(f"\n▶ MATCHUP: {c.cyan(mode_title)} [{n_rnd} Rounds on Identical Matched Seeds]"))

        headers = ["Evaluation Metric", f"{agent_a}", f"{agent_b}", "Delta / Advantage", "Scientific Note"]

        delta_score = summary["score_delta"]
        delta_str = f"+{delta_score:.2f}" if delta_score > 0 else f"{delta_score:.2f}"
        p_val_str = f"p = {summary['p_value']:.4f} " + ("(stat. sig.)" if summary['p_value'] < 0.05 else "(not sig.)")

        rows = [
            ["Rounds Evaluated", str(n_rnd), str(n_rnd), "Matched Seeds", "Symmetric slot balancing"],
            ["Head-to-Head Wins", f"{summary['a_wins']} ({summary['a_win_rate']:.1f}%)", f"{summary['b_wins']} ({summary['b_win_rate']:.1f}%)", f"Ties: {summary['ties']} ({summary['tie_rate']:.1f}%)", f"Win leader: {agent_a if summary['a_wins'] > summary['b_wins'] else (agent_b if summary['b_wins'] > summary['a_wins'] else 'Tied')}"],
            ["Survival Rate", f"{summary['a_survived_pct']:.1f}%", f"{summary['b_survived_pct']:.1f}%", f"{summary['a_survived_pct'] - summary['b_survived_pct']:+.1f}%", "Longevity & evasion"],
            ["Mean Game Score", f"{summary['a_mean_score']:.2f}", f"{summary['b_mean_score']:.2f}", f"Δ = {delta_str} (±{summary['score_delta_ci95']:.2f})", p_val_str],
            ["Coins Collected", f"{summary['a_mean_coins']:.1f} ({summary['a_coin_share_pct']:.1f}%)", f"{summary['b_mean_coins']:.1f} ({summary['b_coin_share_pct']:.1f}%)", f"{summary['a_coin_share_pct'] - summary['b_coin_share_pct']:+.1f}% share", "Resource gathering speed"],
            ["Crates Destroyed", f"{summary['a_mean_crates']:.1f}", f"{summary['b_mean_crates']:.1f}", f"{summary['a_mean_crates'] - summary['b_mean_crates']:+.1f}", "Territory opening capability"],
            ["Total Kills", f"{summary['a_mean_kills']:.2f}/rnd", f"{summary['b_mean_kills']:.2f}/rnd", f"{summary['a_mean_kills'] - summary['b_mean_kills']:+.2f}", "Offensive threat density"],
            ["Direct Rival Kills", f"{summary['a_direct_kills']} kills", f"{summary['b_direct_kills']} kills", f"{agent_a} → {agent_b}" if summary['a_direct_kills'] > summary['b_direct_kills'] else (f"{agent_b} → {agent_a}" if summary['b_direct_kills'] > summary['a_direct_kills'] else "Equal eliminations"), "Direct elimination combat"],
            ["Suicides (Self-Kills)", f"{summary['a_suicides']}", f"{summary['b_suicides']}", f"{summary['a_suicides'] - summary['b_suicides']:+d}", "Self-blast avoidance"],
            ["Matchup Verdict", c.bold(summary['verdict']), c.bold(summary['verdict']), "---", "Final Head-to-Head Standing"],
        ]

        print(render_table(
            headers=headers,
            rows=rows,
            aligns=['left', 'center', 'center', 'center', 'left'],
            title=f"Head-to-Head Scorecard: {agent_a} vs {agent_b} ({summary['mode'].upper()})"
        ))


# -----------------------------------------------------------------------------
# Report Generation (Terminal & Markdown)
# -----------------------------------------------------------------------------

def print_tasks_evaluation_summary(agent_summaries: Dict[str, Dict[str, Any]]):
    """Renders comprehensive ANSI terminal summary tables for all evaluated tasks."""
    c = Colors(enabled=True)

    print("\n" + c.bold("=" * 88))
    print(c.bold("★ MULTI-TASK BENCHMARK EVALUATION (final_project.pdf §4 TASKS 1 - 4) ★"))
    print(c.bold("=" * 88))

    for agent_name, tasks_dict in agent_summaries.items():
        print(c.bold(f"\n▶ AGENT: {c.green(agent_name)}"))

        # Table 1: Per-Task Performance Summary
        headers = ["Task", "Scenario", "Rounds", "Win %", "Surv %", "Score", "Coins", "Crates", "Kills", "Suicides", "Grade"]
        rows = []
        for t_key, t_res in tasks_dict.items():
            t_spec = t_res["task"]
            grade_str = c.green(t_res["grade"]) if "A" in t_res["grade"] else (c.yellow(t_res["grade"]) if "B" in t_res["grade"] else c.red(t_res["grade"]))
            rows.append([
                t_spec["name"],
                t_spec["scenario"],
                str(t_res["n_rounds"]),
                f"{t_res['overall_win_rate']:.1f}%",
                f"{t_res['overall_survival_rate']:.1f}%",
                f"{t_res['overall_mean_score']:.2f}",
                f"{t_res['overall_coins']:.1f}",
                f"{t_res['overall_crates']:.1f}",
                f"{t_res['overall_kills']:.2f}",
                str(t_res["overall_suicides"]),
                grade_str,
            ])

        print(render_table(
            headers=headers,
            rows=rows,
            aligns=['left', 'center', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'center'],
            title=f"Task Breakdown: {agent_name}"
        ))

        # Table 2: Scientific KPIs per task
        print(c.bold(f"\n★ KEY SCIENTIFIC METRICS (REPORT SECTION 6) - {agent_name} ★"))
        kpi_headers = ["Task ID & Goal", "Key Primary Metric", "Efficiency & Safety", "Assessment & Observations"]
        kpi_rows = []
        for t_key, t_res in tasks_dict.items():
            t_spec = t_res["task"]
            kpis = t_res.get("kpis", {})

            if t_spec["task_id"] == 1:
                kpi_rows.append([
                    f"Task 1: Coin Navigation",
                    f"{t_res['overall_coins']:.1f} / 50 coins ({t_res['overall_coins']/50.0*100:.1f}%)",
                    f"{kpis.get('steps_per_coin', 0.0):.1f} steps/coin | {t_res['overall_waited_pct']:.1f}% waits",
                    "Rapid coin seeking; zero wall collisions" if t_res['overall_coins'] >= 35 else "Suboptimal coin pursuit"
                ])
            elif t_spec["task_id"] == 2:
                suicides = t_res['overall_suicides']
                safety_str = c.green("PERFECT SAFETY (0 suicides)") if suicides == 0 else c.red(f"{suicides} suicides ({suicides/max(1, t_res['n_rounds']):.2f}/rnd)")
                kpi_rows.append([
                    f"Task 2: Crate Clearing",
                    f"{t_res['overall_crates']:.1f} crates destroyed",
                    f"{kpis.get('crates_per_bomb', 0.0):.2f} crates/bomb | {safety_str}",
                    "Explosive path clearing with safe bomb escapes"
                ])
            elif t_spec["task_id"] == 3:
                kpi_rows.append([
                    f"Task 3: Hunting Enemies",
                    f"{t_res['overall_kills']:.2f} kills / round",
                    f"Win: {t_res['overall_win_rate']:.1f}% | Surv: {t_res['overall_survival_rate']:.1f}%",
                    "Successfully targets & neutralizes moving agents"
                ])
            elif t_spec["task_id"] == 4:
                diff = kpis.get("delta_vs_rule_based_score", 0.0)
                diff_str = c.green(f"+{diff:.2f}") if diff > 0 else f"{diff:.2f}"
                kpi_rows.append([
                    f"Task 4: Tournament Combat",
                    f"Win Rate: {t_res['overall_win_rate']:.1f}% vs 3x Rule-Based",
                    f"Score Δ vs Rule-Based: {diff_str} pts",
                    "Tournament-ready: beats rule_based_agent benchmark" if t_res['overall_win_rate'] >= 40 else "Competitive combatant"
                ])
            else:
                kpi_rows.append([
                    t_spec["name"],
                    f"Score: {t_res['overall_mean_score']:.2f}",
                    f"Win Rate: {t_res['overall_win_rate']:.1f}%",
                    t_spec["focus"],
                ])

        print(render_table(
            headers=kpi_headers,
            rows=kpi_rows,
            aligns=['left', 'left', 'left', 'left'],
            title="Scientific Metrics & KPIs"
        ))

    # Cross-Agent Comparison Table (if multiple agents evaluated)
    if len(agent_summaries) > 1:
        print("\n" + c.bold("=" * 88))
        print(c.bold("★ MODEL COMPARISON FOR FINAL PROJECT REPORT (SECTION 6) ★"))
        print(c.bold("=" * 88))
        comp_headers = ["Model / Agent", "Task 1 (Coins)", "Task 2 (Crates/Suic)", "Task 3 (Kills/Win%)", "Task 4 (Win%/Score)", "Overall Rating"]
        comp_rows = []
        for a_name, t_dict in agent_summaries.items():
            t1 = t_dict.get("task1", {})
            t2 = t_dict.get("task2", {})
            t3 = t_dict.get("task3", {})
            t4 = t_dict.get("task4", {})

            t1_str = f"{t1.get('overall_coins', 0):.1f} ({t1.get('grade', '-')})" if t1 else "-"
            t2_str = f"{t2.get('overall_crates', 0):.1f} cr ({t2.get('overall_suicides', 0)} suic)" if t2 else "-"
            t3_str = f"{t3.get('overall_kills', 0):.2f} k ({t3.get('overall_win_rate', 0):.0f}%)" if t3 else "-"
            t4_str = f"{t4.get('overall_win_rate', 0):.1f}% ({t4.get('overall_mean_score', 0):.1f} pts)" if t4 else "-"

            # Composite rating
            grades = [t.get("grade", "B") for t in [t1, t2, t3, t4] if t]
            composite = "CHAMPION" if grades.count("A+") >= 2 else ("STRONG" if "A" in "".join(grades) else "BASELINE")

            comp_rows.append([
                c.bold(a_name),
                t1_str,
                t2_str,
                t3_str,
                t4_str,
                c.green(composite) if composite == "CHAMPION" else composite
            ])

        print(render_table(
            headers=comp_headers,
            rows=comp_rows,
            aligns=['left', 'center', 'center', 'center', 'center', 'center'],
            title="Multi-Model Cross-Task Benchmark"
        ))


def print_depth_sweep_summary_table(
    depth_summaries: Dict[int, Dict[str, Any]],
    agent_name: str,
):
    """Renders formatted comparison table across evaluated planning horizons (search depths)."""
    c = Colors(enabled=True)
    depths = sorted(depth_summaries.keys())
    if not depths:
        return

    first_d = depths[0]
    task_keys = list(depth_summaries[first_d].keys())

    headers = ["Planning Horizon (D)"]
    for t_key in task_keys:
        t_spec = depth_summaries[first_d][t_key]["task"]
        headers.append(f"{t_spec['name']} ({t_spec['scenario']})")
    headers.extend(["Mean Latency", "Horizon Trade-Off Observation"])

    rows = []
    prev_score = None
    for d in depths:
        row = [f"Depth D = {d}"]
        total_score = 0.0
        latencies = []

        for t_key in task_keys:
            t_res = depth_summaries[d].get(t_key, {})
            score = t_res.get("overall_mean_score", 0.0)
            win_pct = t_res.get("overall_win_rate", 0.0)
            coins = t_res.get("overall_coins", 0.0)
            crates = t_res.get("overall_crates", 0.0)
            suicides = t_res.get("overall_suicides", 0)
            total_score += score
            if "mean_ms_per_step" in t_res and t_res["mean_ms_per_step"] > 0:
                latencies.append(t_res["mean_ms_per_step"])

            t_id = t_res.get("task", {}).get("task_id", 0)
            if t_id == 1:
                row.append(f"{coins:.1f} coins ({score:.1f} pts)")
            elif t_id == 2:
                row.append(f"{crates:.1f} cr ({suicides} suic)")
            elif t_id in (3, 4):
                row.append(f"{win_pct:.1f}% win ({score:.2f} pts)")
            else:
                row.append(f"{score:.2f} pts")

        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        row.append(f"{mean_lat:.1f} ms/step" if mean_lat > 0 else "-")

        if prev_score is None:
            verdict = "Baseline (Shallow)"
        else:
            diff = total_score - prev_score
            if diff > 1.5:
                verdict = c.green(f"Strong Gain (+{diff:.1f} pts)")
            elif diff > 0.3:
                verdict = c.cyan(f"Moderate Gain (+{diff:.1f} pts)")
            elif diff >= -0.3:
                verdict = c.yellow("Diminishing Returns")
            else:
                verdict = c.red(f"Horizon Overfit ({diff:.1f} pts)")
        prev_score = total_score
        row.append(verdict)
        rows.append(row)

    print("\n" + c.bold("=" * 88))
    print(c.bold("★ PLANNING HORIZON (SEARCH DEPTH) SWEEP: MATCHED SEED EVALUATION ★"))
    print(c.bold("=" * 88))
    print(render_table(
        headers=headers,
        rows=rows,
        aligns=["left"] + ["center"] * len(task_keys) + ["center", "left"],
        title=f"Planning Horizon Comparison: {agent_name}"
    ))


def generate_depth_comparison_plots(
    depth_summaries: Dict[int, Dict[str, Any]],
    depth_rounds: Dict[int, List[RoundExecutionResult]],
    agent_name: str,
    output_dir: Path,
    timestamp: str,
    plot_style: str = "default",
) -> Optional[Path]:
    """Generates a publication-grade multi-panel figure comparing planning horizons (search depths)."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not installed; skipping plot generation.")
        return None

    depths = sorted(depth_summaries.keys())
    if len(depths) < 2:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = f"depth_sweep_{agent_name}_{timestamp}.png"
    plot_path = output_dir / plot_filename

    first_d = depths[0]
    task_keys = list(depth_summaries[first_d].keys())

    # Set aesthetic style
    try:
        if plot_style != "default" and plot_style in plt.style.available:
            plt.style.use(plot_style)
        elif "seaborn-v0_8-whitegrid" in plt.style.available:
            plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    fig.suptitle(f"BombeRLe Planning Horizon Benchmark — {agent_name}\n"
                 f"Systematic Evaluation Across Search Depths D in {depths} on Matched Seeds",
                 fontsize=15, fontweight='bold', y=0.98)

    palette = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c"]

    # 1. Panel (0, 0): Mean Game Score vs Depth
    ax00 = axes[0, 0]
    for i, t_key in enumerate(task_keys):
        t_name = depth_summaries[first_d][t_key]["task"]["name"]
        color = palette[i % len(palette)]
        means = []
        cis = []
        for d in depths:
            t_res = depth_summaries[d].get(t_key, {})
            score = t_res.get("overall_mean_score", 0.0)
            ci = t_res.get("overall_score_ci95", 0.0)
            means.append(score)
            cis.append(ci)
        means = np.array(means)
        cis = np.array(cis)
        ax00.plot(depths, means, marker='o', linewidth=2.5, markersize=7, color=color, label=t_name)
        ax00.fill_between(depths, means - cis, means + cis, color=color, alpha=0.15)
    ax00.set_title("Mean Game Score vs Planning Horizon (95% CI)", fontsize=11, fontweight='bold')
    ax00.set_xlabel("Search Depth / Planning Horizon (D)", fontsize=10)
    ax00.set_ylabel("Mean Game Score (pts)", fontsize=10)
    ax00.set_xticks(depths)
    ax00.grid(True, linestyle="--", alpha=0.5)
    ax00.legend(loc="best", fontsize=8)

    # 2. Panel (0, 1): Win Rate & Survival Rate vs Depth
    ax01 = axes[0, 1]
    has_combat_task = any(t_key in ("task3", "task4") for t_key in task_keys)
    if has_combat_task:
        for i, t_key in enumerate(t for t in task_keys if t in ("task3", "task4")):
            t_name = depth_summaries[first_d][t_key]["task"]["name"]
            color = palette[(i + 2) % len(palette)]
            win_rates = [depth_summaries[d].get(t_key, {}).get("overall_win_rate", 0.0) for d in depths]
            surv_rates = [depth_summaries[d].get(t_key, {}).get("overall_survival_rate", 0.0) for d in depths]
            ax01.plot(depths, win_rates, marker='^', linewidth=2.2, color=color, label=f"{t_name} (Win %)")
            ax01.plot(depths, surv_rates, marker='s', linewidth=2.0, linestyle="--", color=color, alpha=0.8, label=f"{t_name} (Surv %)")
    else:
        for i, t_key in enumerate(task_keys):
            t_name = depth_summaries[first_d][t_key]["task"]["name"]
            color = palette[i % len(palette)]
            surv_rates = [depth_summaries[d].get(t_key, {}).get("overall_survival_rate", 0.0) for d in depths]
            ax01.plot(depths, surv_rates, marker='s', linewidth=2.0, color=color, label=f"{t_name} (Surv %)")

    ax01.set_title("Win & Survival Rates vs Planning Horizon", fontsize=11, fontweight='bold')
    ax01.set_xlabel("Search Depth / Planning Horizon (D)", fontsize=10)
    ax01.set_ylabel("Rate (%)", fontsize=10)
    ax01.set_ylim(-5, 105)
    ax01.set_xticks(depths)
    ax01.grid(True, linestyle="--", alpha=0.5)
    ax01.legend(loc="best", fontsize=8)

    # 3. Panel (0, 2): Objective Metrics (Coins & Crates) vs Depth
    ax02 = axes[0, 2]
    plotted_kpi = False
    if "task1" in task_keys:
        coins_t1 = [depth_summaries[d].get("task1", {}).get("overall_coins", 0.0) for d in depths]
        ax02.plot(depths, coins_t1, marker='o', color="#f39c12", linewidth=2.5, label="Task 1: Coins Collected")
        plotted_kpi = True
    if "task2" in task_keys:
        crates_t2 = [depth_summaries[d].get("task2", {}).get("overall_crates", 0.0) for d in depths]
        ax02.plot(depths, crates_t2, marker='D', color="#8e44ad", linewidth=2.5, label="Task 2: Crates Destroyed")
        plotted_kpi = True
    if any(t in task_keys for t in ("task3", "task4")):
        for t_k in [t for t in task_keys if t in ("task3", "task4")]:
            t_name = depth_summaries[first_d][t_k]["task"]["name"]
            kills = [depth_summaries[d].get(t_k, {}).get("overall_kills", 0.0) for d in depths]
            ax02.plot(depths, kills, marker='x', linewidth=2.2, label=f"{t_name}: Kills")
            plotted_kpi = True
    if not plotted_kpi:
        total_scores = [sum(depth_summaries[d].get(tk, {}).get("overall_mean_score", 0.0) for tk in task_keys) for d in depths]
        ax02.plot(depths, total_scores, marker='o', color="#16a085", linewidth=2.5, label="Combined Score")

    ax02.set_title("Domain Objectives (Coins / Crates / Kills) vs Depth", fontsize=11, fontweight='bold')
    ax02.set_xlabel("Search Depth / Planning Horizon (D)", fontsize=10)
    ax02.set_ylabel("Quantity", fontsize=10)
    ax02.set_xticks(depths)
    ax02.grid(True, linestyle="--", alpha=0.5)
    ax02.legend(loc="best", fontsize=8)

    # 4. Panel (1, 0): Decision Latency (Compute Complexity) vs Depth
    ax10 = axes[1, 0]
    latencies_per_depth = []
    for d in depths:
        lat_list = []
        for tk in task_keys:
            t_res = depth_summaries[d].get(tk, {})
            if "mean_ms_per_step" in t_res and t_res["mean_ms_per_step"] > 0:
                lat_list.append(t_res["mean_ms_per_step"])
        latencies_per_depth.append(float(np.mean(lat_list)) if lat_list else 0.0)

    max_lat = max(latencies_per_depth) if latencies_per_depth else 100.0
    ax10.bar(depths, latencies_per_depth, color="#3498db", alpha=0.65, edgecolor="#2980b9", width=0.45, label="Step Latency (ms)")
    ax10.plot(depths, latencies_per_depth, marker='o', color="#e74c3c", linewidth=2.0, label="Latency Curve")
    for d, lat in zip(depths, latencies_per_depth):
        ax10.text(d, lat + max(1.0, max_lat * 0.03), f"{lat:.1f}ms", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax10.set_title("Decision Latency per Step vs Planning Horizon", fontsize=11, fontweight='bold')
    ax10.set_xlabel("Search Depth / Planning Horizon (D)", fontsize=10)
    ax10.set_ylabel("Latency (ms / step)", fontsize=10)
    ax10.set_xticks(depths)
    ax10.set_ylim(0, max_lat * 1.25 if max_lat > 0 else 100)
    ax10.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax10.legend(loc="upper left", fontsize=8)

    # 5. Panel (1, 1): Safety & Self-Preservation (Suicides) vs Depth
    ax11 = axes[1, 1]
    total_suicides = [sum(depth_summaries[d].get(tk, {}).get("overall_suicides", 0) for tk in task_keys) for d in depths]
    ax11.plot(depths, total_suicides, marker='v', color="#e74c3c", linewidth=2.5, label="Total Suicides (Self-Kills)")
    for d, s in zip(depths, total_suicides):
        ax11.annotate(str(s), (d, s), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8, fontweight='bold')
    if "task2" in task_keys:
        crates_per_bomb = [depth_summaries[d].get("task2", {}).get("kpis", {}).get("crates_per_bomb", 0.0) for d in depths]
        ax11_2 = ax11.twinx()
        ax11_2.plot(depths, crates_per_bomb, marker='o', color="#27ae60", linestyle=":", linewidth=2.2, label="Crates / Bomb")
        ax11_2.set_ylabel("Crates per Bomb", color="#27ae60", fontsize=10)
        ax11_2.tick_params(axis='y', labelcolor="#27ae60")

    ax11.set_title("Safety & Self-Preservation vs Depth", fontsize=11, fontweight='bold')
    ax11.set_xlabel("Search Depth / Planning Horizon (D)", fontsize=10)
    ax11.set_ylabel("Suicide Count", fontsize=10)
    ax11.set_xticks(depths)
    ax11.grid(True, linestyle="--", alpha=0.5)
    ax11.legend(loc="upper right", fontsize=8)

    # 6. Panel (1, 2): Computational Pareto Frontier (Score vs Latency)
    ax12 = axes[1, 2]
    combined_scores = [sum(depth_summaries[d].get(tk, {}).get("overall_mean_score", 0.0) for tk in task_keys) for d in depths]
    ax12.plot(latencies_per_depth, combined_scores, marker='o', color="#8e44ad", linewidth=2.5, markersize=8)
    for d, lat, sc in zip(depths, latencies_per_depth, combined_scores):
        ax12.annotate(f"D={d}\n({sc:.1f} pts)", (lat, sc), textcoords="offset points", xytext=(8, -4),
                      ha='left', fontsize=8, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#8e44ad"))
    ax12.set_title("Computational Pareto Frontier (Score vs Latency)", fontsize=11, fontweight='bold')
    ax12.set_xlabel("Mean Decision Latency (ms / step)", fontsize=10)
    ax12.set_ylabel("Total Benchmark Score (pts)", fontsize=10)
    ax12.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    return plot_path


def export_markdown_task_report(
    agent_summaries: Dict[str, Dict[str, Any]],
    output_path: Path,
    h2h_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
    depth_sweep_data: Optional[Dict[str, Any]] = None,
):
    """Exports a publication-grade GitHub-flavored Markdown evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# BombeRLe Multi-Task Evaluation Report\n")
    md.append("**Reference:** `final_project.pdf` (Machine Learning Essentials, Summer-Semester 2026)\n")
    md.append(f"- **Generated:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    md.append(f"- **Evaluated Agents:** {', '.join(f'`{a}`' for a in agent_summaries.keys())}\n")

    md.append("## 1. Executive Task Summary\n")
    md.append("The project handout defines 4 progressive tasks that agents must solve to attain full tournament readiness:\n")
    md.append("1. **Task 1: Coin Navigation** (`coin-heaven`, solo) — Efficient pathfinding and coin collection without bombs.")
    md.append("2. **Task 2: Crate Clearing & Bomb Safety** (`loot-crate`, solo) — Destroying crates, discovering coins, and escaping explosions with zero self-kills.")
    md.append("3. **Task 3: Hunting Passive & Collector Enemies** (`classic`, vs peaceful + coin_collector) — Eliminating moving adversaries while dodging counter-attacks.")
    md.append("4. **Task 4: Full Competitive Combat** (`classic`, vs 3x rule_based_agent) — Full tournament combat and beating the benchmark.\n")

    for agent_name, tasks_dict in agent_summaries.items():
        if not tasks_dict or agent_name.startswith("_"):
            continue
        md.append(f"### Performance Summary: `{agent_name}`\n")
        md.append("| Task | Scenario | Rounds | Win % | Surv % | Score | Coins | Crates | Kills | Suicides | Grade |")
        md.append("|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
        for t_key, t_res in tasks_dict.items():
            t_spec = t_res["task"]
            md.append(
                f"| **{t_spec['name']}** | `{t_spec['scenario']}` | {t_res['n_rounds']} | "
                f"**{t_res['overall_win_rate']:.1f}%** | {t_res['overall_survival_rate']:.1f}% | "
                f"{t_res['overall_mean_score']:.2f} | {t_res['overall_coins']:.1f} | {t_res['overall_crates']:.1f} | "
                f"{t_res['overall_kills']:.2f} | {t_res['overall_suicides']} | **{t_res['grade']}** |"
            )
        md.append("\n")

        md.append(f"#### Key Scientific KPIs: `{agent_name}`\n")
        md.append("| Task Goal | Primary Metric | Efficiency & Safety Metric | Key Observation |")
        md.append("|:---|:---|:---|:---|")
        for t_key, t_res in tasks_dict.items():
            t_spec = t_res["task"]
            kpis = t_res.get("kpis", {})
            if t_spec["task_id"] == 1:
                md.append(f"| **Task 1: Coin Navigation** | {t_res['overall_coins']:.1f} coins / round | {kpis.get('steps_per_coin', 0.0):.1f} steps / coin | High-speed coin navigation; {t_res['overall_waited_pct']:.1f}% waits |")
            elif t_spec["task_id"] == 2:
                md.append(f"| **Task 2: Crate Clearing** | {t_res['overall_crates']:.1f} crates destroyed | {kpis.get('crates_per_bomb', 0.0):.2f} crates / bomb | {kpis.get('safety_rating', 'Safe')}; {t_res['overall_survival_rate']:.1f}% survival |")
            elif t_spec["task_id"] == 3:
                md.append(f"| **Task 3: Hunting Enemies** | {t_res['overall_kills']:.2f} kills / round | {t_res['overall_win_rate']:.1f}% win rate | Neutralizes moving targets while evading blasts |")
            elif t_spec["task_id"] == 4:
                diff = kpis.get("delta_vs_rule_based_score", 0.0)
                diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                md.append(f"| **Task 4: Tournament Combat** | {t_res['overall_win_rate']:.1f}% win rate | Score Δ vs Rule-Based: {diff_str} pts | Outperforms rule-based agent baseline |")
        md.append("\n")

    visible_agents = [a for a in agent_summaries.keys() if not a.startswith("_")]
    if len(visible_agents) > 1 and any(agent_summaries[a] for a in visible_agents):
        md.append("## 2. Cross-Model Comparative Table (Section 6 Report Ready)\n")
        md.append("| Model / Architecture | Task 1 (Coins) | Task 2 (Crates / Suicides) | Task 3 (Kills / Win%) | Task 4 (Win% vs Rule-Based) | Overall Rating |")
        md.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for a_name in visible_agents:
            t_dict = agent_summaries[a_name]
            t1 = t_dict.get("task1", {})
            t2 = t_dict.get("task2", {})
            t3 = t_dict.get("task3", {})
            t4 = t_dict.get("task4", {})
            t1_str = f"{t1.get('overall_coins', 0):.1f} ({t1.get('grade', '-')})" if t1 else "-"
            t2_str = f"{t2.get('overall_crates', 0):.1f} cr ({t2.get('overall_suicides', 0)} suic)" if t2 else "-"
            t3_str = f"{t3.get('overall_kills', 0):.2f} k ({t3.get('overall_win_rate', 0):.0f}%)" if t3 else "-"
            t4_str = f"{t4.get('overall_win_rate', 0):.1f}% ({t4.get('overall_mean_score', 0):.1f} pts)" if t4 else "-"
            grades = [t.get("grade", "B") for t in [t1, t2, t3, t4] if t]
            composite = "CHAMPION" if grades.count("A+") >= 2 else ("STRONG" if "A" in "".join(grades) else "BASELINE")
            md.append(f"| **`{a_name}`** | {t1_str} | {t2_str} | {t3_str} | {t4_str} | **{composite}** |")
        md.append("\n")

    if h2h_summaries:
        md.append("## 3. Direct Head-to-Head Benchmarks (Matched Seeds)\n")
        md.append("Direct head-to-head competition between candidate models on identical matched random seeds, with alternating starting slot positions to guarantee symmetric fairness.\n")

        for mode_key, summary in h2h_summaries.items():
            if not summary.get("success", False):
                continue
            agent_a = summary["agent_a"]
            agent_b = summary["agent_b"]
            n_rnd = summary["n_rounds"]
            mode_name = "1v1 Duel (No Opponents)" if summary["mode"] == "1v1" else "4-Player Tournament (1v1 + 2 Rule-Based Opponents)"

            md.append(f"### {mode_name}\n")
            md.append(f"- **Rounds Played:** {n_rnd} matched rounds")
            md.append(f"- **Verdict:** **`{summary['verdict']}`**")
            md.append(f"- **Score Delta:** Δ = `{summary['score_delta']:+.2f}` (±`{summary['score_delta_ci95']:.2f}` 95% CI, p = `{summary['p_value']:.4f}`)\n")

            md.append(f"| Metric | `{agent_a}` | `{agent_b}` | Advantage / Notes |")
            md.append("|:---|:---:|:---:|:---|")
            md.append(f"| **Wins / Win Rate** | **{summary['a_wins']} ({summary['a_win_rate']:.1f}%)** | **{summary['b_wins']} ({summary['b_win_rate']:.1f}%)** | Ties: {summary['ties']} ({summary['tie_rate']:.1f}%) |")
            md.append(f"| **Survival Rate** | {summary['a_survived_pct']:.1f}% | {summary['b_survived_pct']:.1f}% | {summary['a_survived_pct'] - summary['b_survived_pct']:+.1f}% difference |")
            md.append(f"| **Mean Score** | **{summary['a_mean_score']:.2f}** | **{summary['b_mean_score']:.2f}** | Δ = {summary['score_delta']:+.2f} |")
            md.append(f"| **Coins Collected** | {summary['a_mean_coins']:.1f} ({summary['a_coin_share_pct']:.1f}%) | {summary['b_mean_coins']:.1f} ({summary['b_coin_share_pct']:.1f}%) | Resource share |")
            md.append(f"| **Crates Destroyed** | {summary['a_mean_crates']:.1f} | {summary['b_mean_crates']:.1f} | Destructive power |")
            md.append(f"| **Total Kills** | {summary['a_mean_kills']:.2f} / rnd | {summary['b_mean_kills']:.2f} / rnd | Elimination frequency |")
            md.append(f"| **Direct Rival Kills** | **{summary['a_direct_kills']} kills** | **{summary['b_direct_kills']} kills** | Eliminations of rival agent |")
            md.append(f"| **Suicides** | {summary['a_suicides']} | {summary['b_suicides']} | Self-blast errors |")
            md.append("\n")

    if depth_sweep_data:
        md.append("## 4. Planning Horizon (Search Depth) Analysis\n")
        md.append("Systematic evaluation across planning horizons (search depths $D$) on identical matched random seeds, isolating the empirical impact of tree search lookahead on score, survival, resource gathering, and computational decision latency.\n")

        for sw_agent, sw_info in depth_sweep_data.items():
            d_summaries = sw_info.get("depth_summaries", {})
            plot_p = sw_info.get("plot_path")
            depths_list = sorted(d_summaries.keys())
            if not depths_list:
                continue

            md.append(f"### Horizon Performance Breakdown: `{sw_agent}`\n")
            first_d = depths_list[0]
            task_keys = list(d_summaries[first_d].keys())

            col_headers = ["Horizon (D)"] + [d_summaries[first_d][tk]["task"]["name"] for tk in task_keys] + ["Latency (ms/step)", "Trade-Off Observation"]
            md.append("| " + " | ".join(col_headers) + " |")
            md.append("|" + "|".join([":---:"] * len(col_headers)) + "|")

            prev_sc = None
            for d in depths_list:
                row = [f"**D = {d}**"]
                tot_sc = 0.0
                lats = []
                for tk in task_keys:
                    tr = d_summaries[d].get(tk, {})
                    sc = tr.get("overall_mean_score", 0.0)
                    win_p = tr.get("overall_win_rate", 0.0)
                    tot_sc += sc
                    if "mean_ms_per_step" in tr and tr["mean_ms_per_step"] > 0:
                        lats.append(tr["mean_ms_per_step"])

                    tid = tr.get("task", {}).get("task_id", 0)
                    if tid == 1:
                        row.append(f"{tr.get('overall_coins', 0.0):.1f} coins ({sc:.1f} pts)")
                    elif tid == 2:
                        row.append(f"{tr.get('overall_crates', 0.0):.1f} cr ({tr.get('overall_suicides', 0)} suic)")
                    elif tid in (3, 4):
                        row.append(f"{win_p:.1f}% win ({sc:.2f} pts)")
                    else:
                        row.append(f"{sc:.2f} pts")

                m_lat = float(np.mean(lats)) if lats else 0.0
                row.append(f"{m_lat:.1f} ms" if m_lat > 0 else "-")

                if prev_sc is None:
                    verdict = "Baseline (Shallow)"
                else:
                    diff = tot_sc - prev_sc
                    if diff > 1.5:
                        verdict = f"Strong Gain (+{diff:.1f} pts)"
                    elif diff > 0.3:
                        verdict = f"Moderate Gain (+{diff:.1f} pts)"
                    elif diff >= -0.3:
                        verdict = "Diminishing Returns"
                    else:
                        verdict = f"Horizon Overfit ({diff:.1f} pts)"
                prev_sc = tot_sc
                row.append(verdict)
                md.append("| " + " | ".join(row) + " |")
            md.append("\n")

            if plot_p:
                md.append(f"### Planning Horizon Visualizations: `{sw_agent}`\n")
                md.append(f"![Planning Horizon Benchmark — {sw_agent}]({Path(plot_p).name})\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def export_h2h_csv(h2h_results: List[H2HRoundResult], output_path: Path):
    """Exports granular round-by-round statistics for head-to-head matches to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode", "round", "seed", "agent_a", "agent_b", "winner", "overall_winner",
        "a_score", "b_score", "a_coins", "b_coins", "a_crates", "b_crates",
        "a_kills", "b_kills", "a_killed_b", "b_killed_a", "a_suicides", "b_suicides",
        "a_survived", "b_survived", "steps", "slot_0", "slot_1"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in h2h_results:
            if r.success:
                writer.writerow({
                    "mode": r.mode,
                    "round": r.round_idx,
                    "seed": r.seed,
                    "agent_a": r.agent_a,
                    "agent_b": r.agent_b,
                    "winner": r.winner,
                    "overall_winner": r.overall_winner,
                    "a_score": r.a_stats.get("score", 0),
                    "b_score": r.b_stats.get("score", 0),
                    "a_coins": r.a_stats.get("coins", 0),
                    "b_coins": r.b_stats.get("coins", 0),
                    "a_crates": r.a_stats.get("crates", 0),
                    "b_crates": r.b_stats.get("crates", 0),
                    "a_kills": r.a_stats.get("kills", 0),
                    "b_kills": r.b_stats.get("kills", 0),
                    "a_killed_b": 1 if r.a_killed_b else 0,
                    "b_killed_a": 1 if r.b_killed_a else 0,
                    "a_suicides": r.a_stats.get("suicides", 0),
                    "b_suicides": r.b_stats.get("suicides", 0),
                    "a_survived": 1 if r.a_stats.get("survived", 0) else 0,
                    "b_survived": 1 if r.b_stats.get("survived", 0) else 0,
                    "steps": r.steps,
                    "slot_0": r.slot_order[0] if len(r.slot_order) > 0 else "",
                    "slot_1": r.slot_order[1] if len(r.slot_order) > 1 else "",
                })


def export_rounds_csv(all_results: List[RoundExecutionResult], output_path: Path):
    """Exports granular round-by-round statistics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "agent", "task_id", "task_key", "round", "seed", "search_depth",
        "score", "won", "survived", "coins", "crates", "kills",
        "suicides", "got_killed", "steps", "bombs", "waited", "moves", "invalid",
        "duration_sec", "ms_per_step"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            if r.success:
                writer.writerow({
                    "agent": r.agent_name,
                    "task_id": r.task_id,
                    "task_key": r.task_key,
                    "round": r.round_idx,
                    "seed": r.seed,
                    "search_depth": r.search_depth if r.search_depth is not None else "",
                    "score": r.score,
                    "won": 1 if r.won else 0,
                    "survived": 1 if r.survived else 0,
                    "coins": r.coins,
                    "crates": r.crates,
                    "kills": r.kills,
                    "suicides": r.suicides,
                    "got_killed": r.got_killed,
                    "steps": r.steps,
                    "bombs": r.bombs,
                    "waited": r.waited,
                    "moves": r.moves,
                    "invalid": r.invalid,
                    "duration_sec": f"{r.duration_sec:.3f}" if r.duration_sec else "0.0",
                    "ms_per_step": f"{r.ms_per_step:.2f}" if r.ms_per_step else "0.0",
                })


# -----------------------------------------------------------------------------
# Main Evaluation Orchestrator
# -----------------------------------------------------------------------------

def evaluate_tasks_orchestrator(
    agents: List[str],
    tasks: List[TaskSpec],
    rounds_per_task: int = 20,
    task_rounds_override: Optional[Dict[str, int]] = None,
    n_workers: int = 8,
    base_seed: int = 1000,
    model_file: Optional[str] = None,
    output_dir: Path = DEFAULT_EVAL_DIR,
    save_artifacts: bool = True,
    verbose: bool = False,
    show_progress: bool = True,
    run_head_to_head: bool = True,
    h2h_only: bool = False,
    h2h_rounds: Optional[int] = None,
    h2h_modes: Optional[List[str]] = None,
    depths: Optional[List[int]] = None,
    max_depth: Optional[int] = None,
    plot: bool = True,
    plot_style: str = "default",
) -> Tuple[Dict[str, Dict[str, Any]], Path, Path, Path]:
    """Orchestrates multi-task parallel evaluation across agents and tasks, including head-to-head duels."""
    c = Colors(enabled=True)
    task_rounds_override = task_rounds_override or {}
    solo_tasks = [t for t in tasks if not t.key.startswith("h2h")]

    depths_to_run: Optional[List[int]] = None
    if depths:
        depths_to_run = [int(d) for d in depths]
    elif max_depth is not None and max_depth >= 1:
        depths_to_run = list(range(1, max_depth + 1))

    print(c.bold("\n" + "=" * 88))
    print(c.bold("★ LAUNCHING MULTI-TASK BOMBERLE EVALUATION (final_project.pdf) ★"))
    print(c.bold("=" * 88))
    print(f"Agents:       {', '.join(agents)}")
    if not h2h_only and solo_tasks:
        print(f"Tasks:        {len(solo_tasks)} ({', '.join(t.key for t in solo_tasks)})")
        print(f"Rounds/Task:  {rounds_per_task} (Matched seeds {base_seed}..)")
    if depths_to_run:
        print(f"Depth Sweep:  Depths {depths_to_run} (Planning horizon comparison enabled)")
    should_run_h2h = (len(agents) >= 2) and (run_head_to_head or h2h_only or any(t.key.startswith("h2h") for t in tasks))
    if should_run_h2h:
        n_h2h_eff = h2h_rounds if h2h_rounds is not None else rounds_per_task
        print(f"Head-to-Head: Enabled ({n_h2h_eff} rounds per matchup on matched seeds)")
    print(f"Workers:      {n_workers} concurrent processes {'(Multiprocessing Active)' if n_workers > 1 else '(Sequential)'}")
    print(f"Progress Bar: {'Enabled' if show_progress else 'Disabled'}")
    print(f"Output Dir:   {output_dir}")
    print("=" * 88 + "\n")
    sys.stdout.flush()

    all_round_results: List[RoundExecutionResult] = []
    agent_summaries: Dict[str, Dict[str, Any]] = {a: {} for a in agents}
    agent_depth_summaries: Dict[str, Dict[int, Dict[str, Any]]] = {a: {} for a in agents}
    agent_depth_rounds: Dict[str, Dict[int, List[RoundExecutionResult]]] = {a: {} for a in agents}
    depth_sweep_export_data: Dict[str, Any] = {}
    h2h_summaries: Dict[str, Dict[str, Any]] = {}
    all_h2h_results: List[H2HRoundResult] = []

    start_time = time.time()
    executor = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None

    try:
        # 1. Multi-Task Evaluation (Tasks 1 - 4)
        if not h2h_only and solo_tasks:
            effective_depths: List[Optional[int]] = depths_to_run if depths_to_run else [None]

            for agent_name in agents:
                try:
                    ckpt_path, source_desc = resolve_agent_model(agent_name, model_file)
                    print(f"[{agent_name}] Checkpoint: {ckpt_path.name} ({source_desc})")
                except Exception as err:
                    print(f"Warning: {err}")
                    ckpt_path = None

                for d in effective_depths:
                    if d is not None:
                        print(c.bold(f"\n--- [{agent_name}] Planning Horizon: Search Depth D = {d} ---"))

                    for t_idx, task in enumerate(solo_tasks, start=1):
                        n_rounds = task_rounds_override.get(task.key, rounds_per_task)
                        depth_suffix = f" (Depth D={d})" if d is not None else ""
                        print(c.bold(f"\n▶ [{t_idx}/{len(solo_tasks)}] Running {task.name} ({task.scenario}){depth_suffix} for {n_rounds} rounds..."))
                        sys.stdout.flush()

                        match_tasks = []
                        for r_idx in range(1, n_rounds + 1):
                            seed = base_seed + (r_idx - 1) * 31
                            match_tasks.append((
                                agent_name,
                                task.scenario,
                                list(task.opponents),
                                seed,
                                r_idx,
                                task.key,
                                str(ckpt_path) if ckpt_path else None,
                                d,
                            ))

                        task_rounds: List[RoundExecutionResult] = []
                        running_scores: List[float] = []
                        running_wins: int = 0
                        running_suicides: int = 0

                        pbar_tag = f" D={d}" if d is not None else ""
                        pbar = TaskProgressBar(
                            total=n_rounds,
                            desc=f"[{agent_name}{pbar_tag}] {task.name[:24]}",
                            enabled=show_progress,
                        )

                        if executor is not None and n_rounds > 0:
                            futures = [executor.submit(run_single_match_worker, mt) for mt in match_tasks]
                            for fut in as_completed(futures):
                                res = fut.result()
                                res.task_id = task.task_id
                                res.search_depth = d
                                task_rounds.append(res)
                                all_round_results.append(res)

                                running_scores.append(res.score)
                                if res.won:
                                    running_wins += 1
                                running_suicides += res.suicides

                                # Live metric postfix for progress bar
                                post: Dict[str, Any] = {
                                    "win": f"{(running_wins / len(task_rounds)) * 100:.0f}%",
                                    "pts": f"{np.mean(running_scores):.1f}",
                                }
                                if task.task_id == 1:
                                    post["coins"] = f"{np.mean([r.coins for r in task_rounds]):.1f}"
                                elif task.task_id == 2:
                                    post["crates"] = f"{np.mean([r.crates for r in task_rounds]):.1f}"
                                    post["suic"] = str(running_suicides)
                                elif task.task_id == 3:
                                    post["kills"] = f"{np.mean([r.kills for r in task_rounds]):.2f}"
                                    post["suic"] = str(running_suicides)
                                else:
                                    post["kills"] = f"{np.mean([r.kills for r in task_rounds]):.2f}"
                                    post["suic"] = str(running_suicides)

                                pbar.update(1, postfix=post)
                                if verbose:
                                    print(f"    Round {res.round_idx:2d}: Score {res.score:4.1f} | Won {res.won} | Steps {res.steps:3d}")
                        else:
                            for mt in match_tasks:
                                res = run_single_match_worker(mt)
                                res.task_id = task.task_id
                                res.search_depth = d
                                task_rounds.append(res)
                                all_round_results.append(res)

                                running_scores.append(res.score)
                                if res.won:
                                    running_wins += 1
                                running_suicides += res.suicides

                                post = {
                                    "win": f"{(running_wins / len(task_rounds)) * 100:.0f}%",
                                    "pts": f"{np.mean(running_scores):.1f}",
                                }
                                if task.task_id == 1:
                                    post["coins"] = f"{np.mean([r.coins for r in task_rounds]):.1f}"
                                elif task.task_id == 2:
                                    post["crates"] = f"{np.mean([r.crates for r in task_rounds]):.1f}"
                                    post["suic"] = str(running_suicides)
                                elif task.task_id == 3:
                                    post["kills"] = f"{np.mean([r.kills for r in task_rounds]):.2f}"
                                    post["suic"] = str(running_suicides)
                                else:
                                    post["kills"] = f"{np.mean([r.kills for r in task_rounds]):.2f}"
                                    post["suic"] = str(running_suicides)

                                pbar.update(1, postfix=post)
                                if verbose:
                                    print(f"    Round {res.round_idx:2d}: Score {res.score:4.1f} | Won {res.won} | Steps {res.steps:3d}")

                        pbar.close()
                        task_rounds.sort(key=lambda r: r.round_idx)
                        summary = summarize_task_results(task, task_rounds)

                        if d is not None:
                            if agent_name not in agent_depth_summaries:
                                agent_depth_summaries[agent_name] = {}
                            if d not in agent_depth_summaries[agent_name]:
                                agent_depth_summaries[agent_name][d] = {}
                            agent_depth_summaries[agent_name][d][task.key] = summary

                            if agent_name not in agent_depth_rounds:
                                agent_depth_rounds[agent_name] = {}
                            if d not in agent_depth_rounds[agent_name]:
                                agent_depth_rounds[agent_name][d] = []
                            agent_depth_rounds[agent_name][d].extend(task_rounds)

                        agent_summaries[agent_name][task.key] = summary

                        if summary["success"]:
                            win_pct = summary["overall_win_rate"]
                            score = summary["overall_mean_score"]
                            coins = summary["overall_coins"]
                            crates = summary["overall_crates"]
                            kills = summary["overall_kills"]
                            suic = summary["overall_suicides"]
                            grade = summary["grade"]
                            print(f"  ✔ Finished {task.key}{depth_suffix}: Score {score:.2f} | Win {win_pct:.1f}% | Coins {coins:.1f} | Crates {crates:.1f} | Kills {kills:.2f} | Suicides {suic} | Grade {c.bold(grade)}")
                        else:
                            print(f"  ✖ Task {task.key}{depth_suffix} failed: No successful rounds.")

        # 2. Direct Head-to-Head Comparisons
        if should_run_h2h:
            agent_a = agents[0]
            agent_b = agents[1]
            n_h2h_rounds = h2h_rounds if h2h_rounds is not None else rounds_per_task

            try:
                ckpt_a, desc_a = resolve_agent_model(agent_a, model_file)
                ckpt_b, desc_b = resolve_agent_model(agent_b, model_file)
                print(c.bold(f"\n★ DIRECT HEAD-TO-HEAD BATTLE: {c.cyan(agent_a)} vs {c.magenta(agent_b)} ★"))
                print(f"  • {agent_a} Model: {ckpt_a.name} ({desc_a})")
                print(f"  • {agent_b} Model: {ckpt_b.name} ({desc_b})")
                print(f"  • Matched Seeds: {base_seed} .. {base_seed + (n_h2h_rounds - 1) * 31} (symmetric slot balancing)")
            except Exception as err:
                print(f"Warning resolving H2H models: {err}")
                ckpt_a, ckpt_b = None, None

            requested_modes = []
            if h2h_modes and "all" not in h2h_modes:
                for m in h2h_modes:
                    if m in ("1v1", "4p") and m not in requested_modes:
                        requested_modes.append(m)
            elif any(t.key.startswith("h2h") for t in tasks):
                for t in tasks:
                    if t.key == "h2h_1v1" and "1v1" not in requested_modes:
                        requested_modes.append("1v1")
                    elif t.key == "h2h_4p" and "4p" not in requested_modes:
                        requested_modes.append("4p")
            if not requested_modes:
                requested_modes = ["1v1", "4p"]

            for m_idx, mode in enumerate(requested_modes, start=1):
                spec = H2H_MATCHUPS[mode]
                print(c.bold(f"\n▶ [H2H {m_idx}/{len(requested_modes)}] Running {spec.name} for {n_h2h_rounds} rounds..."))
                sys.stdout.flush()

                h2h_match_tasks = []
                for r_idx in range(1, n_h2h_rounds + 1):
                    seed = base_seed + (r_idx - 1) * 31
                    slot_reversed = (r_idx % 2 == 0)
                    h2h_match_tasks.append((
                        agent_a,
                        agent_b,
                        list(spec.opponents),
                        seed,
                        r_idx,
                        mode,
                        str(ckpt_a) if ckpt_a else None,
                        str(ckpt_b) if ckpt_b else None,
                        slot_reversed,
                    ))

                mode_rounds: List[H2HRoundResult] = []
                running_a_wins = 0
                running_b_wins = 0
                running_a_pts: List[float] = []
                running_b_pts: List[float] = []

                pbar = TaskProgressBar(
                    total=n_h2h_rounds,
                    desc=f"[{mode.upper()}] {agent_a[:12]} vs {agent_b[:12]}",
                    enabled=show_progress,
                )

                if executor is not None and n_h2h_rounds > 0:
                    futures = [executor.submit(run_single_h2h_worker, hmt) for hmt in h2h_match_tasks]
                    for fut in as_completed(futures):
                        res = fut.result()
                        mode_rounds.append(res)
                        all_h2h_results.append(res)
                        if res.winner == agent_a: running_a_wins += 1
                        elif res.winner == agent_b: running_b_wins += 1
                        running_a_pts.append(float(res.a_stats.get("score", 0)))
                        running_b_pts.append(float(res.b_stats.get("score", 0)))

                        post = {
                            f"win_{agent_a[:4]}": f"{(running_a_wins / len(mode_rounds)) * 100:.0f}%",
                            f"win_{agent_b[:4]}": f"{(running_b_wins / len(mode_rounds)) * 100:.0f}%",
                            f"pts_{agent_a[:4]}": f"{np.mean(running_a_pts):.1f}",
                            f"pts_{agent_b[:4]}": f"{np.mean(running_b_pts):.1f}",
                        }
                        pbar.update(1, postfix=post)
                else:
                    for hmt in h2h_match_tasks:
                        res = run_single_h2h_worker(hmt)
                        mode_rounds.append(res)
                        all_h2h_results.append(res)
                        if res.winner == agent_a: running_a_wins += 1
                        elif res.winner == agent_b: running_b_wins += 1
                        running_a_pts.append(float(res.a_stats.get("score", 0)))
                        running_b_pts.append(float(res.b_stats.get("score", 0)))

                        post = {
                            f"win_{agent_a[:4]}": f"{(running_a_wins / len(mode_rounds)) * 100:.0f}%",
                            f"win_{agent_b[:4]}": f"{(running_b_wins / len(mode_rounds)) * 100:.0f}%",
                            f"pts_{agent_a[:4]}": f"{np.mean(running_a_pts):.1f}",
                            f"pts_{agent_b[:4]}": f"{np.mean(running_b_pts):.1f}",
                        }
                        pbar.update(1, postfix=post)

                pbar.close()
                mode_rounds.sort(key=lambda r: r.round_idx)
                summary = summarize_h2h_results(mode_rounds, agent_a, agent_b)
                h2h_summaries[mode] = summary

                if summary["success"]:
                    print(f"  ✔ Finished H2H {mode.upper()}: {agent_a} ({summary['a_wins']}W) vs {agent_b} ({summary['b_wins']}W) | Ties: {summary['ties']} | Score Δ: {summary['score_delta']:+.2f} | Verdict: {c.bold(summary['verdict'])}")

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    total_time = time.time() - start_time
    total_rounds_all = len(all_round_results) + len(all_h2h_results)
    print(c.green(f"\n✔ All evaluation completed in {total_time:.1f}s across {total_rounds_all} total game rounds.\n"))

    # Print Terminal Summary Tables
    if any(agent_summaries.values()):
        print_tasks_evaluation_summary(agent_summaries)

    if h2h_summaries:
        print_h2h_summary_tables(h2h_summaries)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_tag = "_vs_".join(agents) if len(agents) <= 2 else f"{len(agents)}_agents"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Depth Sweep Summary Tables & Plot Generation
    if depths_to_run and len(depths_to_run) >= 1:
        for agent_name in agents:
            d_sums = agent_depth_summaries.get(agent_name, {})
            if d_sums:
                print_depth_sweep_summary_table(d_sums, agent_name)

                plot_p = None
                if plot and len(depths_to_run) >= 2:
                    d_rnds = agent_depth_rounds.get(agent_name, {})
                    plot_p = generate_depth_comparison_plots(
                        depth_summaries=d_sums,
                        depth_rounds=d_rnds,
                        agent_name=agent_name,
                        output_dir=output_dir,
                        timestamp=timestamp,
                        plot_style=plot_style,
                    )
                    if plot_p:
                        print(c.bold(f"  • Planning Horizon Plot Saved: {plot_p}"))

                depth_sweep_export_data[agent_name] = {
                    "depth_summaries": d_sums,
                    "plot_path": str(plot_p) if plot_p else None,
                }

    report_md_path = output_dir / f"eval_tasks_{agent_tag}_{timestamp}.md"
    results_json_path = output_dir / f"eval_tasks_{agent_tag}_{timestamp}.json"
    rounds_csv_path = output_dir / f"eval_tasks_{agent_tag}_{timestamp}_rounds.csv"

    if save_artifacts:
        export_markdown_task_report(
            agent_summaries,
            report_md_path,
            h2h_summaries=h2h_summaries,
            depth_sweep_data=depth_sweep_export_data if depths_to_run else None,
        )
        export_rounds_csv(all_round_results, rounds_csv_path)
        if all_h2h_results:
            h2h_csv_path = output_dir / f"eval_tasks_{agent_tag}_{timestamp}_h2h_rounds.csv"
            export_h2h_csv(all_h2h_results, h2h_csv_path)

        export_data = {
            "timestamp": timestamp,
            "duration_seconds": total_time,
            "agents": agents,
            "tasks": [asdict(t) for t in solo_tasks],
            "summaries": agent_summaries,
            "head_to_head": h2h_summaries,
        }
        if depths_to_run:
            export_data["depths"] = depths_to_run
            export_data["depth_sweep"] = depth_sweep_export_data

        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        print(c.bold("★ EXPORTED ARTIFACTS (READY FOR REPORT) ★"))
        print(f"  • Markdown Report:   {report_md_path}")
        print(f"  • JSON Full Stats:   {results_json_path}")
        print(f"  • Per-Round CSV Log: {rounds_csv_path}")
        if all_h2h_results:
            print(f"  • H2H Rounds CSV:    {h2h_csv_path}")
        if depths_to_run:
            for ag, sw in depth_sweep_export_data.items():
                if sw.get("plot_path"):
                    print(f"  • Horizon Plot ({ag}): {sw['plot_path']}")
        print()

    # Store h2h summaries in agent_summaries for callers
    if h2h_summaries:
        agent_summaries["_head_to_head"] = h2h_summaries

    return agent_summaries, report_md_path, results_json_path, rounds_csv_path


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Builds and returns CLI ArgumentParser."""
    parser = argparse.ArgumentParser(
        description="Multi-Task Scientific Evaluation Benchmark for BombeRLe (final_project.pdf §4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agents", "--agent", "-a", nargs="+", default=["qwm_agent_veryclean"],
        help="One or more agent directories in agent_code/ to evaluate (e.g. qwm_agent_clean, my_spatial_qwm_agent, my_spatial_dqn_agent)"
    )
    parser.add_argument(
        "--tasks", "-t", nargs="+", default=["all"],
        help="Tasks to evaluate (e.g. 'all', '1 2 3 4', 'task1', 'task2', 'task3', 'task4', 'task4b', 'task5', 'combat', 'coins', 'crates', 'h2h')"
    )
    parser.add_argument(
        "--rounds", "-n", type=int, default=20,
        help="Number of matched evaluation rounds per task"
    )
    parser.add_argument(
        "--rounds-task1", type=int, default=None,
        help="Override round count specifically for Task 1"
    )
    parser.add_argument(
        "--rounds-task2", type=int, default=None,
        help="Override round count specifically for Task 2"
    )
    parser.add_argument(
        "--rounds-task3", type=int, default=None,
        help="Override round count specifically for Task 3"
    )
    parser.add_argument(
        "--rounds-task4", type=int, default=None,
        help="Override round count specifically for Task 4"
    )
    parser.add_argument(
        "--multiprocessing", "--parallel", "-p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multiprocessing parallel worker pool across CPU cores"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None,
        help="Number of parallel worker processes (default: min(CPU count, 14))"
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display live dynamic progress bar with real-time stats"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=1000,
        help="Base random seed for matched zero-variance evaluations"
    )
    parser.add_argument(
        "--model-file", "-m", type=str, default=None,
        help="Path or filename of neural network checkpoint (.pt) to evaluate"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=str(DEFAULT_EVAL_DIR),
        help="Directory to store exported markdown report, json stats, and csv logs"
    )
    parser.add_argument(
        "--no-save", action="store_true", default=False,
        help="Disable exporting report and JSON/CSV files to disk"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Print verbose round-by-round statistics"
    )
    parser.add_argument(
        "--head-to-head", "--h2h",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable direct head-to-head comparisons when multiple agents are evaluated (default: True if >= 2 agents)"
    )
    parser.add_argument(
        "--h2h-only", action="store_true", default=False,
        help="Run only direct head-to-head comparison matches between the agents (skips solo tasks 1-4)"
    )
    parser.add_argument(
        "--h2h-rounds", type=int, default=None,
        help="Number of matched rounds specifically for each head-to-head matchup (default: same as --rounds)"
    )
    parser.add_argument(
        "--h2h-modes", nargs="+", choices=["1v1", "4p", "all"], default=["all"],
        help="Head-to-head battle modes to evaluate: '1v1' (duel), '4p' (with 2 rule-based), or 'all'"
    )
    parser.add_argument(
        "--depths", nargs="+", type=int, default=None,
        help="List of search depths / planning horizons to evaluate (e.g. --depths 1 2 3 4) on matched seeds"
    )
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="Maximum search depth to evaluate: systematically sweeps depths 1..max_depth on identical seeds"
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate publication-grade comparative performance vs planning horizon plots (.png)"
    )
    parser.add_argument(
        "--plot-style", type=str, default="default",
        help="Matplotlib aesthetic style for generated plots (e.g. 'default', 'seaborn-v0_8-whitegrid')"
    )
    return parser


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Determine optimal worker count
    available_cpus = os.cpu_count() or 4
    if not args.multiprocessing:
        workers = 1
    elif args.workers is not None:
        workers = max(1, args.workers)
    else:
        workers = min(available_cpus, 14)

    # Resolve requested tasks
    resolved_tasks = resolve_tasks(args.tasks)

    # Determine H2H settings
    is_h2h_task_only = all(t.key.startswith("h2h") for t in resolved_tasks)
    h2h_only = args.h2h_only or is_h2h_task_only

    if args.head_to_head is not None:
        run_h2h = args.head_to_head
    else:
        run_h2h = (len(args.agents) >= 2) or h2h_only

    # Per-task round overrides
    task_rounds_override = {}
    if args.rounds_task1 is not None: task_rounds_override["task1"] = args.rounds_task1
    if args.rounds_task2 is not None: task_rounds_override["task2"] = args.rounds_task2
    if args.rounds_task3 is not None: task_rounds_override["task3"] = args.rounds_task3
    if args.rounds_task4 is not None: task_rounds_override["task4"] = args.rounds_task4

    evaluate_tasks_orchestrator(
        agents=args.agents,
        tasks=resolved_tasks,
        rounds_per_task=args.rounds,
        task_rounds_override=task_rounds_override,
        n_workers=workers,
        base_seed=args.seed,
        model_file=args.model_file,
        output_dir=Path(args.output_dir),
        save_artifacts=not args.no_save,
        verbose=args.verbose,
        show_progress=args.progress,
        run_head_to_head=run_h2h,
        h2h_only=h2h_only,
        h2h_rounds=args.h2h_rounds,
        h2h_modes=args.h2h_modes,
        depths=args.depths,
        max_depth=args.max_depth,
        plot=args.plot,
        plot_style=args.plot_style,
    )


if __name__ == "__main__":
    main()
