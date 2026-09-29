#!/usr/bin/env python3
"""
tune_inference.py -- Hyperparameter Inference Tuning Engine for QWM Agent.

Finds optimal inference parameters (search_depth, beam_size, tree_discount,
alpha_vq, predicted_wait_penalty, predicted_loop_penalty) for any given checkpoint.

Core Capabilities:
  1. Optuna TPE Bayesian Optimization: Efficiently explores high-dimensional continuous
     and discrete inference space across expanded parameter ranges (e.g. horizons up to 8)
     without wasteful exhaustive grid search.
  2. Multiprocessing: Parallel match execution across all CPU cores.
  3. Matched Game Protocol: Zero-variance comparison by evaluating all candidate
     configurations on the identical set of random seeds (identical crate matrix,
     coin spawns, agent start positions, and opponent seeds).
  4. Multi-Scenario Benchmarking with Diverse Enemies:
     - 'classic_combat' vs 3x rule_based_agent (standard tournament combat)
     - 'dqn_rivalry' vs 1x my_spatial_dqn_agent, 2x rule_based_agent (combat vs learned DQN)
     - 'spatial_clash' vs 2x my_spatial_dqn_agent, 1x rule_based_agent (intense neural adversary match)
     - 'loot_crate' vs my_spatial_dqn_agent, rule_based_agent, coin_collector_agent (crate clearing & looting)
     - 'coin_heaven' vs 3x coin_collector_agent (high-speed pathfinding)
  5. Direct Combat Tournament: Top K candidate configurations are deployed into temporary
     agent wrappers and face each other in direct 4-player head-to-head deathmatches to
     determine the undisputed champion.
  6. Reporting & Auto-Save: Formatted tables, JSON telemetry, Markdown report, and
     optional auto-updating of config.json with winning parameters.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Root directory resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

AGENT_CODE_DIR = ROOT_DIR / "agent_code"
DEFAULT_AGENT_DIR = AGENT_CODE_DIR / "my_spatial_qwm_agent"


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class InferenceCandidate:
    name: str
    search_depth: int = 3
    beam_size: int = 12
    tree_discount: float = 0.12
    alpha_vq: float = 0.45
    predicted_wait_penalty: float = 1.0
    predicted_loop_penalty: float = 2.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioDef:
    key: str
    display_name: str
    scenario: str
    opponents: List[str]
    weight: float = 1.0


@dataclass
class MatchTask:
    candidate: InferenceCandidate
    scenario: ScenarioDef
    seed: int
    checkpoint_path: str
    root_dir: str = str(ROOT_DIR)
    timeout_sec: int = 240


@dataclass
class MatchResult:
    candidate_name: str
    scenario_key: str
    seed: int
    success: bool
    score: float = 0.0
    won: bool = False
    survived: bool = False
    coins: float = 0.0
    kills: float = 0.0
    suicides: float = 0.0
    got_killed: float = 0.0
    steps: int = 0
    crates: float = 0.0
    invalid: float = 0.0
    mean_think_time: float = 0.0
    error_msg: Optional[str] = None


# -----------------------------------------------------------------------------
# Formatting & Terminal Output Utilities
# -----------------------------------------------------------------------------

class Colors:
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


import re
ANSI_REGEX = re.compile(r'\x1b\[[0-9;]*m')

def visible_len(text: Any) -> int:
    return len(ANSI_REGEX.sub('', str(text)))


def render_table(headers: List[str], rows: List[List[Any]], aligns: Optional[List[str]] = None,
                 title: Optional[str] = None) -> str:
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

    tl, tm, tr = "┌", "┬", "┐"
    ml, mm, mr = "├", "┼", "┤"
    bl, bm, br = "└", "┴", "┘"
    h_line = "─"
    v_line = "│"

    lines = []
    if title:
        total_w = sum(col_widths) + len(col_widths) - 1
        lines.append(f" {title.upper()} ".center(total_w, "="))

    lines.append(tl + tm.join(h_line * w for w in col_widths) + tr)

    def pad_cell(text: str, width: int, align: str) -> str:
        vis_len = visible_len(text)
        avail = max(0, width - 2)
        pad = max(0, avail - vis_len)
        if align == 'right':
            return " " + " " * pad + text + " "
        elif align == 'center':
            l_pad = pad // 2
            r_pad = pad - l_pad
            return " " + " " * l_pad + text + " " * r_pad + " "
        else:
            return " " + text + " " * pad + " "

    header_cells = [pad_cell(h, col_widths[i], aligns[i]) for i, h in enumerate(str_headers)]
    lines.append(v_line + v_line.join(header_cells) + v_line)
    lines.append(ml + mm.join(h_line * w for w in col_widths) + mr)

    for row in str_rows:
        row_cells = [pad_cell(cell, col_widths[i], aligns[i]) for i, cell in enumerate(row)]
        lines.append(v_line + v_line.join(row_cells) + v_line)

    lines.append(bl + bm.join(h_line * w for w in col_widths) + br)
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Pre-defined Candidate Presets (With Expanded Ranges)
# -----------------------------------------------------------------------------

def get_candidate_presets(preset_name: str) -> List[InferenceCandidate]:
    """Generates candidate configurations based on the requested preset."""
    if preset_name == "quick":
        return [
            InferenceCandidate(
                name="Fast-D2-B6",
                search_depth=2, beam_size=6, tree_discount=0.15, alpha_vq=0.45,
                description="Fast shallow lookahead (ultra-low latency)"
            ),
            InferenceCandidate(
                name="Standard-D3-B12",
                search_depth=3, beam_size=12, tree_discount=0.12, alpha_vq=0.45,
                description="Balanced 3-step lookahead"
            ),
            InferenceCandidate(
                name="Deep-D4-B12",
                search_depth=4, beam_size=12, tree_discount=0.12, alpha_vq=0.40,
                description="Deep 4-step lookahead with value blending"
            ),
            InferenceCandidate(
                name="DeepHorizon-D6-B18",
                search_depth=6, beam_size=18, tree_discount=0.10, alpha_vq=0.40,
                description="Extended horizon (depth 6, beam 18)"
            ),
            InferenceCandidate(
                name="ValueHeavy-D3",
                search_depth=3, beam_size=12, tree_discount=0.15, alpha_vq=0.60,
                description="3-step lookahead prioritizing critic Q-value"
            ),
        ]

    elif preset_name == "standard":
        return [
            InferenceCandidate(
                name="Fast-D2-B6",
                search_depth=2, beam_size=6, tree_discount=0.15, alpha_vq=0.45,
                description="Shallow lookahead (depth 2, beam 6)"
            ),
            InferenceCandidate(
                name="Fast-D2-B12",
                search_depth=2, beam_size=12, tree_discount=0.15, alpha_vq=0.45,
                description="Wide shallow lookahead (depth 2, beam 12)"
            ),
            InferenceCandidate(
                name="Balanced-D3-B8",
                search_depth=3, beam_size=8, tree_discount=0.15, alpha_vq=0.45,
                description="Lean 3-step lookahead (depth 3, beam 8)"
            ),
            InferenceCandidate(
                name="Balanced-D3-B12",
                search_depth=3, beam_size=12, tree_discount=0.12, alpha_vq=0.45,
                description="Standard balanced 3-step (depth 3, beam 12)"
            ),
            InferenceCandidate(
                name="Balanced-D3-B18",
                search_depth=3, beam_size=18, tree_discount=0.12, alpha_vq=0.45,
                description="Wide 3-step lookahead (depth 3, beam 18)"
            ),
            InferenceCandidate(
                name="Deep-D4-B12",
                search_depth=4, beam_size=12, tree_discount=0.12, alpha_vq=0.40,
                description="Deep 4-step lookahead (depth 4, beam 12)"
            ),
            InferenceCandidate(
                name="Deep-D4-B18",
                search_depth=4, beam_size=18, tree_discount=0.10, alpha_vq=0.40,
                description="Deep wide lookahead (depth 4, beam 18)"
            ),
            InferenceCandidate(
                name="Extended-D5-B18",
                search_depth=5, beam_size=18, tree_discount=0.10, alpha_vq=0.40,
                description="Extended 5-step horizon"
            ),
            InferenceCandidate(
                name="DeepHorizon-D6-B24",
                search_depth=6, beam_size=24, tree_discount=0.08, alpha_vq=0.35,
                description="Deep 6-step horizon with wide beam"
            ),
            InferenceCandidate(
                name="MaxHorizon-D8-B24",
                search_depth=8, beam_size=24, tree_discount=0.08, alpha_vq=0.35,
                description="Maximum 8-step horizon (Dong et al. random horizon limit)"
            ),
            InferenceCandidate(
                name="RewardHeavy-D3",
                search_depth=3, beam_size=12, tree_discount=0.15, alpha_vq=0.30,
                description="Reward-heavy prospective blending (alpha=0.30)"
            ),
            InferenceCandidate(
                name="ValueHeavy-D3",
                search_depth=3, beam_size=12, tree_discount=0.15, alpha_vq=0.65,
                description="Critic-heavy prospective blending (alpha=0.65)"
            ),
            InferenceCandidate(
                name="HighDiscount-D4",
                search_depth=4, beam_size=12, tree_discount=0.25, alpha_vq=0.45,
                description="High lookahead discount (lambda=0.25)"
            ),
        ]

    elif preset_name == "thorough":
        candidates = []
        depths = [2, 3, 4, 6, 8]
        beams = [6, 12, 18, 24]
        for d in depths:
            for b in beams:
                for a in [0.30, 0.45, 0.60]:
                    name = f"D{d}-B{b}-A{int(a*100)}"
                    disc = 0.15 if d <= 2 else (0.12 if d <= 4 else 0.08)
                    candidates.append(InferenceCandidate(
                        name=name,
                        search_depth=d,
                        beam_size=b,
                        tree_discount=disc,
                        alpha_vq=a,
                        description=f"Grid config depth={d}, beam={b}, alpha={a}"
                    ))
        return candidates

    else:
        raise ValueError(f"Unknown preset: '{preset_name}'. Choose from: quick, standard, thorough.")


# -----------------------------------------------------------------------------
# Scenarios Suite Definition (Includes my_spatial_dqn_agent as enemy)
# -----------------------------------------------------------------------------

DEFAULT_SCENARIOS = [
    ScenarioDef(
        key="classic_combat",
        display_name="Classic vs 3x Rule-Based",
        scenario="classic",
        opponents=["rule_based_agent", "rule_based_agent", "rule_based_agent"],
        weight=1.5
    ),
    ScenarioDef(
        key="dqn_rivalry",
        display_name="DQN Rivalry (1x Spatial DQN + 2x Rule-Based)",
        scenario="classic",
        opponents=["my_spatial_dqn_agent", "rule_based_agent", "rule_based_agent"],
        weight=1.5
    ),
    ScenarioDef(
        key="spatial_clash",
        display_name="Spatial Clash (2x Spatial DQN + 1x Rule-Based)",
        scenario="classic",
        opponents=["my_spatial_dqn_agent", "my_spatial_dqn_agent", "rule_based_agent"],
        weight=1.2
    ),
    ScenarioDef(
        key="loot_crate",
        display_name="Loot Crate Mixed (Spatial DQN + Rule-Based + Coin Collector)",
        scenario="loot-crate",
        opponents=["my_spatial_dqn_agent", "rule_based_agent", "coin_collector_agent"],
        weight=1.0
    ),
    ScenarioDef(
        key="coin_heaven",
        display_name="Coin Heaven Sprint (3x Coin Collector)",
        scenario="coin-heaven",
        opponents=["coin_collector_agent", "coin_collector_agent", "coin_collector_agent"],
        weight=0.8
    ),
]


# -----------------------------------------------------------------------------
# Multiprocessing Worker Function
# -----------------------------------------------------------------------------

def run_matched_game_worker(task: MatchTask) -> MatchResult:
    """
    Executes a single matched game in an isolated headless subprocess.
    """
    cand = task.candidate
    scen = task.scenario

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        stats_path = Path(tmp_file.name)

    env = os.environ.copy()
    # Configure candidate inference hyperparameters via environment overrides
    env["MY_QWM_MODEL_FILE"] = str(task.checkpoint_path)
    env["MY_QWM_SEARCH_DEPTH"] = str(cand.search_depth)
    env["MY_QWM_BEAM_SIZE"] = str(cand.beam_size)
    env["MY_QWM_TREE_DISCOUNT"] = str(cand.tree_discount)
    env["MY_QWM_ALPHA_VQ"] = str(cand.alpha_vq)
    env["MY_QWM_WAIT_PENALTY"] = str(cand.predicted_wait_penalty)
    env["MY_QWM_LOOP_PENALTY"] = str(cand.predicted_loop_penalty)
    env["MY_QWM_EXPLORE_UNSAFE"] = "0.0"
    env["MY_QWM_EPSILON"] = "0.0"

    # Restrict single-thread CPU execution per worker to prevent CPU thrashing
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"

    cmd = [
        sys.executable,
        str(Path(task.root_dir) / "main.py"),
        "play",
        "--agents", "my_spatial_qwm_agent", *scen.opponents,
        "--scenario", scen.scenario,
        "--seed", str(task.seed),
        "--n-rounds", "1",
        "--no-gui",
        "--silence-errors",
        "--save-stats", str(stats_path)
    ]

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=task.root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=task.timeout_sec
        )

        if not stats_path.is_file() or stats_path.stat().st_size == 0:
            err_text = proc.stderr[-300:] if proc.stderr else "No stats file generated."
            return MatchResult(
                candidate_name=cand.name,
                scenario_key=scen.key,
                seed=task.seed,
                success=False,
                error_msg=f"Exit code {proc.returncode}: {err_text}"
            )

        with open(stats_path, "r") as f:
            raw = json.load(f)

        by_round = raw.get("by_round", {})
        if not by_round:
            return MatchResult(
                candidate_name=cand.name,
                scenario_key=scen.key,
                seed=task.seed,
                success=False,
                error_msg="Empty stats JSON"
            )

        first_round_id = next(iter(by_round.keys()))
        round_data = by_round[first_round_id]
        agent_data = round_data.get("by_agent", {}).get("my_spatial_qwm_agent", {})

        agent_steps = agent_data.get("steps", 0)
        time_total = agent_data.get("time", 0.0)
        mean_think_time = (time_total / agent_steps) if agent_steps > 0 else 0.0

        return MatchResult(
            candidate_name=cand.name,
            scenario_key=scen.key,
            seed=task.seed,
            success=True,
            score=float(agent_data.get("score", 0)),
            won=bool(agent_data.get("won", 0)),
            survived=bool(agent_data.get("survived", 0)),
            coins=float(agent_data.get("coins", 0)),
            kills=float(agent_data.get("kills", 0)),
            suicides=float(agent_data.get("suicides", 0)),
            got_killed=float(agent_data.get("got_killed", 0)),
            steps=int(agent_steps),
            crates=float(agent_data.get("crates", 0)),
            invalid=float(agent_data.get("invalid", 0)),
            mean_think_time=mean_think_time,
        )

    except subprocess.TimeoutExpired:
        return MatchResult(
            candidate_name=cand.name,
            scenario_key=scen.key,
            seed=task.seed,
            success=False,
            error_msg=f"Match timed out after {task.timeout_sec}s"
        )
    except Exception as exc:
        return MatchResult(
            candidate_name=cand.name,
            scenario_key=scen.key,
            seed=task.seed,
            success=False,
            error_msg=str(exc)
        )
    finally:
        stats_path.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Matched Evaluation Metrics Calculation
# -----------------------------------------------------------------------------

def compute_candidate_scenario_summary(
    cand: InferenceCandidate,
    scenarios: List[ScenarioDef],
    results: List[MatchResult],
    baseline_cand_name: Optional[str] = None,
    baseline_summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Computes aggregated KPIs and composite score for a single candidate."""
    scen_summaries: Dict[str, Any] = {}
    for scen in scenarios:
        cand_scen_results = [
            r for r in results
            if r.candidate_name == cand.name and r.scenario_key == scen.key and r.success
        ]
        n_m = len(cand_scen_results)
        if n_m == 0:
            continue

        scores = [r.score for r in cand_scen_results]
        wins = [1.0 if r.won else 0.0 for r in cand_scen_results]
        surv = [1.0 if r.survived else 0.0 for r in cand_scen_results]
        coins = [r.coins for r in cand_scen_results]
        kills = [r.kills for r in cand_scen_results]
        crates = [r.crates for r in cand_scen_results]
        suicides = [r.suicides for r in cand_scen_results]
        think_ms = [r.mean_think_time * 1000.0 for r in cand_scen_results]

        win_rate = (sum(wins) / n_m) * 100.0
        surv_rate = (sum(surv) / n_m) * 100.0
        mean_score = sum(scores) / n_m
        mean_coins = sum(coins) / n_m
        mean_kills = sum(kills) / n_m
        mean_crates = sum(crates) / n_m
        mean_think = sum(think_ms) / n_m

        # Balanced scenario utility function
        scen_utility = (
            win_rate * 40.0 +
            mean_score * 10.0 +
            surv_rate * 20.0 +
            mean_coins * 2.0 +
            mean_kills * 5.0 -
            (sum(suicides) / n_m) * 5.0
        )

        scen_summaries[scen.key] = {
            "n_matches": n_m,
            "win_rate": win_rate,
            "survival_rate": surv_rate,
            "mean_score": mean_score,
            "mean_coins": mean_coins,
            "mean_kills": mean_kills,
            "mean_crates": mean_crates,
            "mean_think_ms": mean_think,
            "scenario_utility": scen_utility,
            "per_seed": {r.seed: r.score for r in cand_scen_results}
        }

    total_weights = sum(s.weight for s in scenarios)
    w_score = 0.0
    w_win = 0.0
    w_surv = 0.0
    w_mean_score = 0.0
    w_coins = 0.0
    w_kills = 0.0
    w_think = 0.0
    total_m = 0

    for scen in scenarios:
        s_dict = scen_summaries.get(scen.key)
        if s_dict:
            w = scen.weight
            w_score += s_dict["scenario_utility"] * w
            w_win += s_dict["win_rate"] * w
            w_surv += s_dict["survival_rate"] * w
            w_mean_score += s_dict["mean_score"] * w
            w_coins += s_dict["mean_coins"] * w
            w_kills += s_dict["mean_kills"] * w
            w_think += s_dict["mean_think_ms"] * w
            total_m += s_dict["n_matches"]

    res = {
        "candidate": cand.to_dict(),
        "scenarios": scen_summaries,
        "overall_score": (w_score / total_weights) if total_weights > 0 else 0.0,
        "overall_win_rate": (w_win / total_weights) if total_weights > 0 else 0.0,
        "overall_survival_rate": (w_surv / total_weights) if total_weights > 0 else 0.0,
        "overall_mean_score": (w_mean_score / total_weights) if total_weights > 0 else 0.0,
        "overall_coins": (w_coins / total_weights) if total_weights > 0 else 0.0,
        "overall_kills": (w_kills / total_weights) if total_weights > 0 else 0.0,
        "overall_think_time_ms": (w_think / total_weights) if total_weights > 0 else 0.0,
        "n_matches": total_m,
    }

    if baseline_cand_name and baseline_summary:
        paired_wins = 0
        paired_ties = 0
        paired_losses = 0
        paired_delta_sum = 0.0
        paired_count = 0
        for scen in scenarios:
            cand_seeds = scen_summaries.get(scen.key, {}).get("per_seed", {})
            base_seeds = baseline_summary.get("scenarios", {}).get(scen.key, {}).get("per_seed", {})
            for s_id, cand_s in cand_seeds.items():
                if s_id in base_seeds:
                    diff = cand_s - base_seeds[s_id]
                    paired_delta_sum += diff
                    paired_count += 1
                    if diff > 0.01: paired_wins += 1
                    elif diff < -0.01: paired_losses += 1
                    else: paired_ties += 1
        res["paired_vs_baseline"] = {
            "baseline": baseline_cand_name,
            "matched_rounds": paired_count,
            "paired_wins": paired_wins,
            "paired_ties": paired_ties,
            "paired_losses": paired_losses,
            "mean_delta_score": (paired_delta_sum / paired_count) if paired_count > 0 else 0.0
        }
    else:
        res["paired_vs_baseline"] = {
            "baseline": cand.name,
            "matched_rounds": total_m,
            "paired_wins": 0, "paired_ties": total_m, "paired_losses": 0, "mean_delta_score": 0.0
        }

    return res


# -----------------------------------------------------------------------------
# Matched Evaluation Aggregator (Grid / Preset Mode)
# -----------------------------------------------------------------------------

def evaluate_candidates_matched(
    candidates: List[InferenceCandidate],
    scenarios: List[ScenarioDef],
    seeds: List[int],
    checkpoint_path: str,
    n_workers: int = 8,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes all candidate evaluations on identical seeds across scenarios in parallel.
    """
    c = Colors(enabled=True)
    tasks: List[MatchTask] = []
    for cand in candidates:
        for scen in scenarios:
            for seed in seeds:
                tasks.append(MatchTask(
                    candidate=cand,
                    scenario=scen,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                ))

    total_tasks = len(tasks)
    if verbose:
        print(c.bold("\n" + "="*80))
        print(c.bold(f"★ PHASE 1: PARALLEL MATCHED-GAME EVALUATION ({total_tasks} MATCHES) ★"))
        print(c.bold("="*80))
        print(f"Candidates:    {len(candidates)} configurations")
        print(f"Scenarios:     {len(scenarios)} ({', '.join(s.key for s in scenarios)})")
        print(f"Matched Seeds: {len(seeds)} seeds per scenario ({seeds[0]}..{seeds[-1]})")
        print(f"Workers:       {n_workers} concurrent processes")
        print(f"Checkpoint:    {Path(checkpoint_path).name}\n")
        sys.stdout.flush()

    start_time = time.time()
    results: List[MatchResult] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {executor.submit(run_matched_game_worker, t): t for t in tasks}

        if TQDM_AVAILABLE and verbose:
            pbar = tqdm(
                total=total_tasks,
                desc="Evaluation Matches",
                unit="match",
                dynamic_ncols=True,
                leave=True
            )
            for future in as_completed(future_to_task):
                res = future.result()
                results.append(res)
                pbar.update(1)
                pbar.set_postfix({
                    "cand": res.candidate_name[:12],
                    "scen": res.scenario_key[:10],
                    "score": f"{res.score:.1f}",
                    "won": res.won
                })
            pbar.close()
        else:
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                res = future.result()
                results.append(res)
                if verbose and (completed % max(1, total_tasks // 20) == 0 or completed == total_tasks):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_tasks - completed) / rate if rate > 0 else 0
                    status_color = c.green if res.success else c.red
                    print(f"  [{completed:4d}/{total_tasks:4d}] "
                          f"{status_color(res.candidate_name[:14]):14s} | "
                          f"{res.scenario_key[:14]:14s} | seed {res.seed:5d} | "
                          f"score {res.score:4.1f} | won {res.won} | "
                          f"{rate:.1f} matches/s | ETA {eta:.0f}s")
                    sys.stdout.flush()

    total_time = time.time() - start_time
    if verbose:
        print(c.bold(f"\nCompleted {len(results)} matches in {total_time:.1f}s ({len(results)/total_time:.2f} matches/s)."))

    # Compute summaries
    baseline_cand = candidates[0]
    baseline_summary = compute_candidate_scenario_summary(baseline_cand, scenarios, results)

    summary_by_cand: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        summary_by_cand[cand.name] = compute_candidate_scenario_summary(
            cand, scenarios, results, baseline_cand.name, baseline_summary
        )

    ranked_candidates = sorted(
        summary_by_cand.values(),
        key=lambda x: x["overall_score"],
        reverse=True
    )

    return {
        "ranked_candidates": ranked_candidates,
        "summary_by_cand": summary_by_cand,
        "scenarios": [asdict(s) for s in scenarios],
        "seeds": seeds,
        "checkpoint_path": checkpoint_path,
        "total_matches": total_tasks,
        "elapsed_seconds": total_time,
        "strategy": "preset",
    }


# -----------------------------------------------------------------------------
# Optuna TPE Bayesian Optimization Engine (Expanded Parameter Space)
# -----------------------------------------------------------------------------

def tune_with_optuna(
    checkpoint_path: str,
    scenarios: List[ScenarioDef],
    seeds: List[int],
    n_trials: int = 30,
    n_jobs: int = 1,
    timeout_sec: Optional[int] = None,
    n_workers: int = 8,
    prune: bool = True,
    seed: int = 42,
    storage: Optional[str] = None,
    study_name: str = "qwm_inference_tuning",
    reset_study: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Runs Bayesian optimization via Optuna TPE across expanded hyperparameter ranges
    on matched game seeds, persistently saved to an Optuna SQLite storage file.
    Supports concurrent trials (n_jobs) sharing the match worker pool.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Please install it via 'pip install optuna'.")

    import threading
    c = Colors(enabled=True)
    lock = threading.Lock()

    # Format SQLite storage URL
    if storage:
        if "://" in storage:
            storage_url = storage
        else:
            db_path = Path(storage).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            storage_url = f"sqlite:///{db_path}"
    else:
        default_db = DEFAULT_AGENT_DIR / "optuna_study.db"
        default_db.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{default_db}"

    if reset_study:
        try:
            optuna.delete_study(study_name=study_name, storage=storage_url)
            if verbose:
                print(c.yellow(f"[optuna] Reset study '{study_name}' in storage: {storage_url}"))
        except Exception:
            pass

    if verbose:
        print(c.bold("\n" + "="*80))
        print(c.bold(f"★ PHASE 1: OPTUNA BAYESIAN OPTIMIZATION ({n_trials} TRIALS) ★"))
        print(c.bold("="*80))
        print(f"Study Name:    {study_name}")
        print(f"Persistent DB: {storage_url}")
        print(f"Scenarios:     {len(scenarios)} ({', '.join(s.key for s in scenarios)})")
        print(f"Matched Seeds: {len(seeds)} seeds per scenario ({seeds[0]}..{seeds[-1]})")
        print(f"Workers:       {n_workers} match processes | {n_jobs} concurrent trial(s)")
        print(f"Checkpoint:    {Path(checkpoint_path).name}")
        print("Search Ranges:")
        print("  • Search Depth:           [1, 8]  (Extended Horizons)")
        print("  • Beam Size:              [6, 36] (step 6)")
        print("  • Tree Discount (λ):      [0.05, 0.40]")
        print("  • Alpha VQ (α):           [0.10, 0.90]")
        print("  • Predicted Wait Penalty: [0.0, 5.0]")
        print("  • Predicted Loop Penalty: [0.0, 8.0]\n")
        sys.stdout.flush()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed, constant_liar=(n_jobs > 1))
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1) if prune else None
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    prior_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if verbose and prior_completed > 0:
        print(c.cyan(f"Resumed persistent study '{study_name}' with {prior_completed} previously completed trials."))

    all_trial_results: Dict[str, Dict[str, Any]] = {}
    all_raw_match_results: List[MatchResult] = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        def objective(trial: optuna.Trial) -> float:
            search_depth = trial.suggest_int("search_depth", 3, 12)
            beam_size = trial.suggest_int("beam_size", 6, 36, step=6)
            tree_discount = trial.suggest_float("tree_discount", 0.00, 0.40, step=0.01)
            alpha_vq = trial.suggest_float("alpha_vq", 0.10, 0.90, step=0.02)
            predicted_wait_penalty = trial.suggest_float("predicted_wait_penalty", 0.0, 5.0, step=0.25)
            predicted_loop_penalty = trial.suggest_float("predicted_loop_penalty", 0.0, 8.0, step=0.5)

            cand_name = f"Optuna-{trial.number:03d}"
            cand = InferenceCandidate(
                name=cand_name,
                search_depth=search_depth,
                beam_size=beam_size,
                tree_discount=tree_discount,
                alpha_vq=alpha_vq,
                predicted_wait_penalty=predicted_wait_penalty,
                predicted_loop_penalty=predicted_loop_penalty,
                description=f"D={search_depth}, B={beam_size}, λ={tree_discount:.2f}, α={alpha_vq:.2f}"
            )

            # Build tasks for this trial
            trial_tasks: List[MatchTask] = []
            for scen in scenarios:
                for s_id in seeds:
                    trial_tasks.append(MatchTask(
                        candidate=cand,
                        scenario=scen,
                        seed=s_id,
                        checkpoint_path=checkpoint_path,
                    ))

            futures = [executor.submit(run_matched_game_worker, t) for t in trial_tasks]
            trial_match_results = [f.result() for f in futures]

            summary = compute_candidate_scenario_summary(cand, scenarios, trial_match_results)
            score = summary["overall_score"]
            with lock:
                all_raw_match_results.extend(trial_match_results)
                all_trial_results[cand_name] = summary

            # Store compact scalar metrics in trial user_attrs (no large JSON strings)
            trial.set_user_attr("overall_score", score)
            trial.set_user_attr("overall_win_rate", summary["overall_win_rate"])
            trial.set_user_attr("overall_mean_score", summary["overall_mean_score"])
            trial.set_user_attr("overall_survival_rate", summary["overall_survival_rate"])
            trial.set_user_attr("overall_coins", summary["overall_coins"])
            trial.set_user_attr("overall_kills", summary["overall_kills"])
            trial.set_user_attr("overall_think_time_ms", summary["overall_think_time_ms"])

            # Report to Optuna
            trial.report(score, step=0)
            if prune and trial.should_prune():
                if verbose:
                    with lock:
                        print(f"  Trial #{trial.number:2d} ({cand_name}) PRUNED with score {score:.1f}")
                raise optuna.TrialPruned()

            if verbose:
                elapsed = time.time() - start_time
                win_pct = summary["overall_win_rate"]
                mean_sc = summary["overall_mean_score"]
                with lock:
                    print(f"  Trial #{trial.number:2d} | {cand_name} (D={search_depth}, B={beam_size}, λ={tree_discount:.2f}, α={alpha_vq:.2f}) "
                          f"-> Score {score:6.1f} | Win {win_pct:4.1f}% | AvgPts {mean_sc:4.1f} | {elapsed:.0f}s elapsed")
                    sys.stdout.flush()

            return score

        study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, n_jobs=n_jobs)
    total_time = time.time() - start_time

    # Recover all completed trials from persistent storage (including previous runs)
    for past_trial in study.trials:
        if past_trial.state == optuna.trial.TrialState.COMPLETE:
            cand_name = f"Optuna-{past_trial.number:03d}"
            if cand_name not in all_trial_results and past_trial.value is not None:
                params = past_trial.params
                cand = InferenceCandidate(
                    name=cand_name,
                    search_depth=params.get("search_depth", 2),
                    beam_size=params.get("beam_size", 12),
                    tree_discount=params.get("tree_discount", 0.15),
                    alpha_vq=params.get("alpha_vq", 0.50),
                    predicted_wait_penalty=params.get("predicted_wait_penalty", 1.0),
                    predicted_loop_penalty=params.get("predicted_loop_penalty", 2.0),
                    description=f"D={params.get('search_depth', 2)}, B={params.get('beam_size', 12)}, λ={params.get('tree_discount', 0.15):.2f}, α={params.get('alpha_vq', 0.50):.2f}"
                )
                u_attrs = past_trial.user_attrs
                all_trial_results[cand_name] = {
                    "candidate": cand.to_dict(),
                    "scenarios": {},
                    "overall_score": u_attrs.get("overall_score", past_trial.value),
                    "overall_win_rate": u_attrs.get("overall_win_rate", 0.0),
                    "overall_survival_rate": u_attrs.get("overall_survival_rate", 0.0),
                    "overall_mean_score": u_attrs.get("overall_mean_score", 0.0),
                    "overall_coins": u_attrs.get("overall_coins", 0.0),
                    "overall_kills": u_attrs.get("overall_kills", 0.0),
                    "overall_think_time_ms": u_attrs.get("overall_think_time_ms", 0.0),
                    "n_matches": 0,
                    "paired_vs_baseline": {
                        "baseline": cand_name,
                        "matched_rounds": 0,
                        "paired_wins": 0,
                        "paired_ties": 0,
                        "paired_losses": 0,
                        "mean_delta_score": 0.0
                    }
                }

    if verbose:
        print(c.bold(f"\nOptuna study complete: {len(study.trials)} total trials in storage ({total_time:.1f}s elapsed)."))
        if study.best_trial:
            print(c.green(c.bold(f"Best Trial #{study.best_trial.number}: Score {study.best_trial.value:.1f}")))
            print(f"Params: {study.best_trial.params}\n")

    # Collate ranked candidates
    ranked_candidates = sorted(
        all_trial_results.values(),
        key=lambda x: x["overall_score"],
        reverse=True
    )

    # Recompute baseline comparison relative to the #1 candidate
    if ranked_candidates:
        best_cand_name = ranked_candidates[0]["candidate"]["name"]
        best_cand_summary = ranked_candidates[0]
        for r in ranked_candidates:
            c_obj = InferenceCandidate(**r["candidate"])
            upd = compute_candidate_scenario_summary(
                c_obj, scenarios, all_raw_match_results, best_cand_name, best_cand_summary
            )
            r["paired_vs_baseline"] = upd["paired_vs_baseline"]

    return {
        "ranked_candidates": ranked_candidates,
        "summary_by_cand": all_trial_results,
        "scenarios": [asdict(s) for s in scenarios],
        "seeds": seeds,
        "checkpoint_path": checkpoint_path,
        "total_matches": len(all_raw_match_results),
        "elapsed_seconds": total_time,
        "strategy": "optuna",
        "storage": storage_url,
        "study_name": study_name,
        "best_params": study.best_trial.params if study.best_trial else {},
    }


# -----------------------------------------------------------------------------
# Direct Combat Head-to-Head Tournament
# -----------------------------------------------------------------------------

def _create_candidate_agent_wrapper(
    wrapper_name: str,
    cand: InferenceCandidate,
    checkpoint_path: str
) -> Path:
    """Creates a temporary agent directory in agent_code/ for direct combat."""
    target_dir = AGENT_CODE_DIR / wrapper_name
    target_dir.mkdir(parents=True, exist_ok=True)

    config_content = {
        "model_file": str(checkpoint_path),
        "tree_search": True,
        "search_depth": cand.search_depth,
        "beam_size": cand.beam_size,
        "tree_discount": cand.tree_discount,
        "alpha_vq": cand.alpha_vq,
        "predicted_wait_penalty": cand.predicted_wait_penalty,
        "predicted_loop_penalty": cand.predicted_loop_penalty,
        "candidate_name": cand.name,
    }
    with open(target_dir / "config.json", "w") as f:
        json.dump(config_content, f, indent=2)

    cb_code = f'''# Auto-generated wrapper for tournament candidate: {cand.name}
import json
import os
from pathlib import Path
import sys

# Ensure my_spatial_qwm_agent is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "my_spatial_qwm_agent"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib
qwm_cb = importlib.import_module("agent_code.my_spatial_qwm_agent.callbacks")

def setup(self):
    cfg_file = Path(__file__).parent / 'config.json'
    if cfg_file.is_file():
        with open(cfg_file) as f:
            cfg = json.load(f)
    else:
        cfg = {{}}
    self.cfg = cfg
    qwm_cb.setup(self)
    # Direct candidate hyperparameter injection
    for k in ['search_depth', 'beam_size', 'tree_discount', 'alpha_vq', 'predicted_wait_penalty', 'predicted_loop_penalty']:
        if k in cfg:
            setattr(self, k, cfg[k])

def act(self, game_state: dict):
    return qwm_cb.act(self, game_state)
'''
    with open(target_dir / "callbacks.py", "w") as f:
        f.write(cb_code)

    return target_dir


def _h2h_round_worker(args_tuple: Tuple[List[str], int, str, int, str]) -> Dict[str, Any]:
    """Runs a single 4-player head-to-head tournament round."""
    agent_names, seed, scenario, timeout_sec, root_dir = args_tuple
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        stats_path = Path(tmp_file.name)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"

    cmd = [
        sys.executable,
        str(Path(root_dir) / "main.py"),
        "play",
        "--agents", *agent_names,
        "--scenario", scenario,
        "--seed", str(seed),
        "--n-rounds", "1",
        "--no-gui",
        "--silence-errors",
        "--save-stats", str(stats_path)
    ]

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec
        )

        if not stats_path.is_file() or stats_path.stat().st_size == 0:
            err = proc.stderr[-300:] if proc.stderr else "No stats file generated"
            return {"success": False, "seed": seed, "error": f"Exit {proc.returncode}: {err}"}

        with open(stats_path, "r") as f:
            raw = json.load(f)

        by_round = raw.get("by_round", {})
        if not by_round:
            return {"success": False, "seed": seed, "error": "Empty stats"}

        first_id = next(iter(by_round.keys()))
        round_agents = by_round[first_id].get("by_agent", {})

        agent_scores = {}
        for a_name, a_data in round_agents.items():
            agent_scores[a_name] = {
                "score": a_data.get("score", 0),
                "won": bool(a_data.get("won", 0)),
                "survived": bool(a_data.get("survived", 0)),
                "kills": a_data.get("kills", 0),
                "coins": a_data.get("coins", 0),
                "suicides": a_data.get("suicides", 0),
            }

        return {
            "success": True,
            "seed": seed,
            "agents": agent_scores
        }
    except Exception as exc:
        return {"success": False, "seed": seed, "error": str(exc)}
    finally:
        stats_path.unlink(missing_ok=True)


def run_head_to_head_tournament(
    top_candidates: List[InferenceCandidate],
    checkpoint_path: str,
    n_rounds: int = 20,
    seeds: Optional[List[int]] = None,
    scenario: str = "classic",
    n_workers: int = 8,
    include_dqn_rival: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Deploys the top candidate configurations into temporary agent wrappers
    and pits them directly against each other in 4-player head-to-head combat.
    """
    c = Colors(enabled=True)
    if seeds is None:
        seeds = [5000 + i for i in range(n_rounds)]

    n_cand = len(top_candidates)
    if n_cand < 2:
        raise ValueError("Need at least 2 candidates for head-to-head tournament.")

    # Limit candidates to at most 4 direct combatants (Bomberman board limit)
    participants = top_candidates[:4]
    wrapper_names = [f"_tune_cand_{i}" for i in range(len(participants))]
    cand_by_wrapper = {w: p for w, p in zip(wrapper_names, participants)}

    created_dirs: List[Path] = []
    try:
        # Create temporary candidate wrappers in agent_code/
        for w_name, cand in zip(wrapper_names, participants):
            wdir = _create_candidate_agent_wrapper(w_name, cand, checkpoint_path)
            created_dirs.append(wdir)

        # Fill remaining slots with my_spatial_dqn_agent and rule_based_agent if < 4 candidates
        active_roster = list(wrapper_names)
        if len(active_roster) < 4 and include_dqn_rival:
            active_roster.append("my_spatial_dqn_agent")
        while len(active_roster) < 4:
            active_roster.append("rule_based_agent")

        if verbose:
            print(c.bold("\n" + "="*80))
            print(c.bold(f"★ PHASE 2: DIRECT COMBAT TOURNAMENT ({n_rounds} ROUNDS) ★"))
            print(c.bold("="*80))
            print(f"Combatants: {', '.join(f'{p.name} ({w})' for w, p in cand_by_wrapper.items())}")
            if "my_spatial_dqn_agent" in active_roster:
                print(f"Rival Agent: my_spatial_dqn_agent (active adversary)")
            print(f"Scenario:   {scenario}")
            print(f"Rounds:     {n_rounds} direct deathmatches across seeds {seeds[0]}..{seeds[-1]}")
            print(f"Workers:    {n_workers} concurrent processes\n")
            sys.stdout.flush()

        # Generate tournament tasks with rotated starting position ordering
        h2h_tasks = []
        for r_idx, seed in enumerate(seeds[:n_rounds]):
            # Rotate roster so each candidate starts from each corner equally
            rot = r_idx % len(active_roster)
            rotated_roster = active_roster[rot:] + active_roster[:rot]
            h2h_tasks.append((rotated_roster, seed, scenario, 240, str(ROOT_DIR)))

        start_time = time.time()
        tournament_records: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_h2h_round_worker, task) for task in h2h_tasks]
            if TQDM_AVAILABLE and verbose:
                pbar = tqdm(
                    total=n_rounds,
                    desc="H2H Tournament",
                    unit="round",
                    dynamic_ncols=True,
                    leave=True
                )
                for f in as_completed(futures):
                    res = f.result()
                    if res.get("success"):
                        tournament_records.append(res)
                    pbar.update(1)
                    pbar.set_postfix({"valid": len(tournament_records)})
                pbar.close()
            else:
                completed = 0
                for f in as_completed(futures):
                    completed += 1
                    res = f.result()
                    if res.get("success"):
                        tournament_records.append(res)
                    if verbose:
                        print(f"  [Round {completed:2d}/{n_rounds:2d}] complete.")
                        sys.stdout.flush()

        elapsed = time.time() - start_time
        if verbose:
            print(c.bold(f"\nTournament completed in {elapsed:.1f}s.\n"))

        # Aggregate Head-to-Head Statistics
        h2h_stats: Dict[str, Dict[str, Any]] = {}
        for w_name in wrapper_names:
            c_name = cand_by_wrapper[w_name].name
            h2h_stats[c_name] = {
                "candidate": cand_by_wrapper[w_name].to_dict(),
                "wrapper": w_name,
                "wins": 0,
                "survived_count": 0,
                "total_score": 0.0,
                "total_kills": 0.0,
                "total_coins": 0.0,
                "total_suicides": 0.0,
                "rounds_played": 0,
            }

        # Also track my_spatial_dqn_agent if participating
        if "my_spatial_dqn_agent" in active_roster:
            h2h_stats["my_spatial_dqn_agent"] = {
                "candidate": {"name": "my_spatial_dqn_agent", "search_depth": 0, "beam_size": 0, "tree_discount": 0, "alpha_vq": 0},
                "wrapper": "my_spatial_dqn_agent",
                "wins": 0,
                "survived_count": 0,
                "total_score": 0.0,
                "total_kills": 0.0,
                "total_coins": 0.0,
                "total_suicides": 0.0,
                "rounds_played": 0,
            }

        valid_rounds = len(tournament_records)
        for r in tournament_records:
            agents_in_round = r["agents"]
            max_score = max(ad["score"] for ad in agents_in_round.values())
            for a_key in h2h_stats.keys():
                w_key = h2h_stats[a_key]["wrapper"]
                if w_key in agents_in_round:
                    ad = agents_in_round[w_key]
                    stat = h2h_stats[a_key]
                    stat["rounds_played"] += 1
                    stat["total_score"] += ad["score"]
                    stat["total_kills"] += ad["kills"]
                    stat["total_coins"] += ad["coins"]
                    stat["total_suicides"] += ad["suicides"]
                    if ad["survived"]:
                        stat["survived_count"] += 1
                    if ad["won"] or (ad["score"] == max_score and max_score > 0):
                        stat["wins"] += 1

        for c_name, stat in h2h_stats.items():
            n_r = max(1, stat["rounds_played"])
            stat["win_rate"] = (stat["wins"] / n_r) * 100.0
            stat["survival_rate"] = (stat["survived_count"] / n_r) * 100.0
            stat["mean_score"] = stat["total_score"] / n_r
            stat["mean_kills"] = stat["total_kills"] / n_r
            stat["mean_coins"] = stat["total_coins"] / n_r

        ranked_h2h = sorted(
            h2h_stats.values(),
            key=lambda x: (x["wins"], x["mean_score"], x["survival_rate"]),
            reverse=True
        )

        return {
            "ranked_h2h": ranked_h2h,
            "h2h_stats": h2h_stats,
            "valid_rounds": valid_rounds,
            "total_rounds": n_rounds,
            "scenario": scenario,
            "elapsed_seconds": elapsed,
        }

    finally:
        # Guarantee cleanup of temporary agent wrapper folders
        for d in created_dirs:
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


# -----------------------------------------------------------------------------
# Reporting & File Exports
# -----------------------------------------------------------------------------

def print_tuning_summary(
    eval_res: Dict[str, Any],
    h2h_res: Optional[Dict[str, Any]] = None,
    save_best_path: Optional[Path] = None
):
    """Renders comprehensive ANSI terminal summary tables."""
    c = Colors(enabled=True)
    ranked = eval_res["ranked_candidates"]

    # Table 1: Overall Cross-Scenario Leaderboard
    print(c.bold("\n" + "="*85))
    print(c.bold("★ CROSS-SCENARIO COMPOSITE LEADERBOARD ★"))
    print(c.bold("="*85))
    if eval_res.get("strategy") == "optuna" and eval_res.get("storage"):
        print(f"Persistent DB: {eval_res['storage']} (Study: '{eval_res.get('study_name', 'default')}')\n")

    headers = [
        "Rank", "Candidate", "Depth", "Beam", "Disc(λ)", "Alpha(α)",
        "Comp Score", "Win %", "Surv %", "Score", "Coins", "Kills", "Think ms"
    ]
    rows = []
    for rank_idx, r in enumerate(ranked[:15], 1):  # Display top 15 candidates
        cand = r["candidate"]
        cand_str = c.bold(cand["name"]) if rank_idx == 1 else cand["name"]
        rows.append([
            f"#{rank_idx}",
            cand_str,
            str(cand["search_depth"]),
            str(cand["beam_size"]),
            f"{cand['tree_discount']:.2f}",
            f"{cand['alpha_vq']:.2f}",
            f"{r['overall_score']:.1f}",
            f"{r['overall_win_rate']:.1f}%",
            f"{r['overall_survival_rate']:.1f}%",
            f"{r['overall_mean_score']:.2f}",
            f"{r['overall_coins']:.2f}",
            f"{r['overall_kills']:.2f}",
            f"{r['overall_think_time_ms']:.1f}",
        ])

    print(render_table(
        headers=headers,
        rows=rows,
        aligns=['center', 'left', 'right', 'right', 'right', 'right',
                'right', 'right', 'right', 'right', 'right', 'right', 'right']
    ))

    # Table 2: Paired Zero-Variance Deltas vs Baseline
    print(c.bold("\n★ ZERO-VARIANCE MATCHED SEED COMPARISON (VS BASELINE) ★"))
    p_headers = ["Candidate", "Baseline", "Matched Rnds", "Wins", "Ties", "Losses", "Win/Loss Diff", "Mean Δ Score"]
    p_rows = []
    for r in ranked[:10]:
        pvb = r["paired_vs_baseline"]
        diff = pvb["paired_wins"] - pvb["paired_losses"]
        diff_str = c.green(f"+{diff}") if diff > 0 else (c.red(str(diff)) if diff < 0 else "0")
        d_score = pvb["mean_delta_score"]
        d_score_str = c.green(f"+{d_score:.2f}") if d_score > 0 else f"{d_score:.2f}"
        p_rows.append([
            r["candidate"]["name"],
            pvb["baseline"],
            str(pvb["matched_rounds"]),
            str(pvb["paired_wins"]),
            str(pvb["paired_ties"]),
            str(pvb["paired_losses"]),
            diff_str,
            d_score_str,
        ])
    print(render_table(headers=p_headers, rows=p_rows, aligns=['left', 'left', 'right', 'right', 'right', 'right', 'right', 'right']))

    # Table 3: Head-to-Head Direct Combat Tournament (if available)
    if h2h_res and h2h_res.get("ranked_h2h"):
        print(c.bold("\n" + "="*85))
        print(c.bold(f"★ HEAD-TO-HEAD DIRECT COMBAT TOURNAMENT ({h2h_res['valid_rounds']} ROUNDS) ★"))
        print(c.bold("="*85))
        h_headers = ["Place", "Candidate", "Direct Wins", "Direct Win %", "Surv %", "Score / Rnd", "Kills", "Coins"]
        h_rows = []
        for p_idx, p in enumerate(h2h_res["ranked_h2h"], 1):
            p_name = c.bold(c.green(p["candidate"]["name"])) if p_idx == 1 else p["candidate"]["name"]
            medal = "🥇 " if p_idx == 1 else ("🥈 " if p_idx == 2 else ("🥉 " if p_idx == 3 else "   "))
            h_rows.append([
                f"{medal}#{p_idx}",
                p_name,
                str(p["wins"]),
                f"{p['win_rate']:.1f}%",
                f"{p['survival_rate']:.1f}%",
                f"{p['mean_score']:.2f}",
                f"{p['mean_kills']:.2f}",
                f"{p['mean_coins']:.2f}",
            ])
        print(render_table(headers=h_headers, rows=h_rows, aligns=['center', 'left', 'right', 'right', 'right', 'right', 'right', 'right']))

    # Print Recommendation (filter out external baseline opponents like my_spatial_dqn_agent)
    valid_cands = [
        p["candidate"] for p in (h2h_res["ranked_h2h"] if (h2h_res and h2h_res.get("ranked_h2h")) else ranked)
        if p["candidate"]["name"] != "my_spatial_dqn_agent"
    ]
    champion = valid_cands[0] if valid_cands else ranked[0]["candidate"]

    print(c.bold("\n" + "="*85))
    print(c.bold(f"★ RECOMMENDED INFERENCE HYPERPARAMETERS ★"))
    print(c.bold("="*85))
    print(f"  • Candidate:      {c.green(c.bold(champion['name']))}")
    print(f"  • Search Depth:   {champion['search_depth']} (Lookahead Horizon)")
    print(f"  • Beam Size:      {champion['beam_size']}")
    print(f"  • Tree Discount:  {champion['tree_discount']:.3f}")
    print(f"  • Alpha VQ:       {champion['alpha_vq']:.3f}")
    print(f"  • Wait Penalty:   {champion['predicted_wait_penalty']:.2f}")
    print(f"  • Loop Penalty:   {champion['predicted_loop_penalty']:.2f}\n")

    if save_best_path:
        print(c.green(f"✔ Successfully saved best configuration to: {save_best_path}\n"))


def export_markdown_tuning_report(
    eval_res: Dict[str, Any],
    h2h_res: Optional[Dict[str, Any]],
    output_file: Path
):
    """Generates a complete GitHub-flavored Markdown tuning report."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ranked = eval_res["ranked_candidates"]
    valid_cands = [
        p["candidate"] for p in (h2h_res["ranked_h2h"] if (h2h_res and h2h_res.get("ranked_h2h")) else ranked)
        if p["candidate"]["name"] != "my_spatial_dqn_agent"
    ]
    champion = valid_cands[0] if valid_cands else ranked[0]["candidate"]

    md = []
    md.append("# Inference Hyperparameter Tuning Report\n")
    md.append(f"- **Evaluated Checkpoint:** `{eval_res['checkpoint_path']}`")
    md.append(f"- **Strategy:** `{eval_res.get('strategy', 'unknown')}`")
    if eval_res.get("strategy") == "optuna" and eval_res.get("storage"):
        md.append(f"- **Optuna Persistent DB:** `{eval_res['storage']}`")
        md.append(f"- **Optuna Study Name:** `{eval_res.get('study_name', 'default')}`")
    md.append(f"- **Timestamp:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    md.append(f"- **Total Matched Games:** {eval_res['total_matches']}")
    md.append(f"- **Evaluation Duration:** {eval_res['elapsed_seconds']:.1f}s\n")

    md.append("## 1. Undisputed Best Configuration\n")
    md.append(f"**Recommended Configuration:** `{champion['name']}`\n")
    md.append("| Hyperparameter | Recommended Value | Description |")
    md.append("|:---|---:|:---|")
    md.append(f"| `search_depth` | **{champion['search_depth']}** | Lookahead horizon / tree depth |")
    md.append(f"| `beam_size` | **{champion['beam_size']}** | Beam search width |")
    md.append(f"| `tree_discount` | **{champion['tree_discount']:.3f}** | Lookahead discount factor $\\lambda$ |")
    md.append(f"| `alpha_vq` | **{champion['alpha_vq']:.3f}** | Blending weight $\\alpha$ ($V_Q$ vs $V_r$) |")
    md.append(f"| `predicted_wait_penalty` | **{champion['predicted_wait_penalty']:.2f}** | Penalty for predicted wait actions |")
    md.append(f"| `predicted_loop_penalty` | **{champion['predicted_loop_penalty']:.2f}** | Penalty for revisiting recent coordinates |\n")

    md.append("## 2. Cross-Scenario Composite Leaderboard\n")
    md.append("| Rank | Candidate | Depth | Beam | Disc ($\\lambda$) | Alpha ($\\alpha$) | Comp Score | Win % | Surv % | Score | Coins | Kills | Think (ms) |")
    md.append("|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx, r in enumerate(ranked[:20], 1):
        c_cand = r["candidate"]
        md.append(
            f"| #{idx} | **{c_cand['name']}** | {c_cand['search_depth']} | {c_cand['beam_size']} | "
            f"{c_cand['tree_discount']:.2f} | {c_cand['alpha_vq']:.2f} | **{r['overall_score']:.1f}** | "
            f"{r['overall_win_rate']:.1f}% | {r['overall_survival_rate']:.1f}% | {r['overall_mean_score']:.2f} | "
            f"{r['overall_coins']:.2f} | {r['overall_kills']:.2f} | {r['overall_think_time_ms']:.1f} |"
        )
    md.append("")

    if h2h_res and h2h_res.get("ranked_h2h"):
        md.append(f"## 3. Direct Combat Head-to-Head Tournament ({h2h_res['valid_rounds']} Rounds)\n")
        md.append("| Place | Combatant | Direct Wins | Win % | Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |")
        md.append("|:---|:---|---:|---:|---:|---:|---:|---:|")
        for idx, p in enumerate(h2h_res["ranked_h2h"], 1):
            c_cand = p["candidate"]
            medal = "🥇 " if idx == 1 else ("🥈 " if idx == 2 else ("🥉 " if idx == 3 else ""))
            md.append(
                f"| {medal}#{idx} | **{c_cand['name']}** | {p['wins']} | **{p['win_rate']:.1f}%** | "
                f"{p['survival_rate']:.1f}% | {p['mean_score']:.2f} | {p['mean_kills']:.2f} | {p['mean_coins']:.2f} |"
            )
        md.append("")

    with open(output_file, "w") as f:
        f.write("\n".join(md))


# -----------------------------------------------------------------------------
# Checkpoint Resolution
# -----------------------------------------------------------------------------

def resolve_checkpoint(requested: Optional[str] = None) -> Path:
    """Finds the model checkpoint file to evaluate."""
    if requested:
        p = Path(requested)
        if p.is_file():
            return p.resolve()
        p2 = ROOT_DIR / requested
        if p2.is_file():
            return p2.resolve()
        p3 = DEFAULT_AGENT_DIR / requested
        if p3.is_file():
            return p3.resolve()
        raise FileNotFoundError(f"Requested checkpoint not found: {requested}")

    # Check agent root for default saved model
    for fname in ["my-saved-model-spatial-qwm.pt", "my-saved-model-spatial-dqn.pt"]:
        p = DEFAULT_AGENT_DIR / fname
        if p.is_file():
            return p.resolve()

    # Check runs directory
    runs_dir = DEFAULT_AGENT_DIR / "runs"
    if runs_dir.is_dir():
        best_pts = sorted(runs_dir.glob("**/best_model_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if best_pts:
            return best_pts[0].resolve()

    raise FileNotFoundError("Could not find any model checkpoint (.pt) in agent directory or runs.")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel Matched-Game & Tournament Hyperparameter Tuning for QWM Agent."
    )
    parser.add_argument(
        "--checkpoint", "--model-file", "-m", dest="checkpoint", type=str, default=None,
        help="Path to trained agent checkpoint (.pt). Defaults to my-saved-model-spatial-qwm.pt"
    )
    parser.add_argument(
        "--strategy", type=str, default="optuna" if OPTUNA_AVAILABLE else "preset",
        choices=["optuna", "preset"],
        help="Search strategy: 'optuna' (Bayesian TPE optimization) or 'preset' (fixed grid)."
    )
    parser.add_argument(
        "--n-trials", type=int, default=25,
        help="Number of Optuna Bayesian optimization trials (default: 25)."
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Timeout in seconds for the Optuna study (default: None)."
    )
    parser.add_argument(
        "--prune", action="store_true", default=True,
        help="Enable Optuna median pruning for unpromising configurations."
    )
    parser.add_argument(
        "--no-prune", dest="prune", action="store_false",
        help="Disable Optuna pruning."
    )
    parser.add_argument(
        "--optuna-seed", type=int, default=42,
        help="Random seed for Optuna TPE sampler (default: 42)."
    )
    parser.add_argument(
        "--storage", type=str, default=str(DEFAULT_AGENT_DIR / "optuna_study.db"),
        help="Path or URL to Optuna persistent storage SQLite file (default: 'agent_code/my_spatial_qwm_agent/optuna_study.db')."
    )
    parser.add_argument(
        "--study-name", type=str, default="qwm_inference_tuning",
        help="Optuna study name (default: 'qwm_inference_tuning')."
    )
    parser.add_argument(
        "--reset-study", action="store_true", default=False,
        help="Reset/delete existing Optuna study in storage before running."
    )
    parser.add_argument(
        "--n-jobs", "--trial-workers", dest="n_jobs", type=int, default=1,
        help="Number of concurrent Optuna trials to run in parallel (default: 1). "
             "All concurrent trials share the worker match pool with zero CPU oversubscription."
    )
    parser.add_argument(
        "--preset", type=str, default="standard", choices=["quick", "standard", "thorough"],
        help="Preset grid when --strategy preset is used ('quick', 'standard', 'thorough')."
    )
    parser.add_argument(
        "--candidates-json", type=str, default=None,
        help="Path to custom JSON file specifying candidate configurations."
    )
    parser.add_argument(
        "--scenarios", type=str, default="classic_combat,dqn_rivalry,loot_crate,coin_heaven",
        help="Comma-separated scenario list to include (classic_combat, dqn_rivalry, spatial_clash, loot_crate, coin_heaven)."
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of matched random seeds to evaluate per scenario (default: 5)."
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=max(1, min(os.cpu_count() or 4, 8)),
        help=f"Number of parallel worker processes (default: {max(1, min(os.cpu_count() or 4, 8))})."
    )
    parser.add_argument(
        "--h2h-rounds", type=int, default=12,
        help="Number of direct combat tournament rounds among top candidates (default: 12, set 0 to disable)."
    )
    parser.add_argument(
        "--top-k", type=int, default=4,
        help="Number of top candidates to advance to direct combat tournament (max 4, default: 4)."
    )
    parser.add_argument(
        "--save-best", action="store_true", default=False,
        help="Automatically write the winning hyperparameters into config.json."
    )
    parser.add_argument(
        "--output-dir", type=str, default="eval_results",
        help="Directory to store markdown report and JSON results."
    )
    return parser


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"[tune_inference] Using checkpoint: {checkpoint_path}")

    # Filter scenarios
    requested_scen_keys = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    scenarios: List[ScenarioDef] = []
    for s_def in DEFAULT_SCENARIOS:
        if any(req in s_def.scenario or req in s_def.key for req in requested_scen_keys):
            scenarios.append(s_def)
    if not scenarios:
        scenarios = DEFAULT_SCENARIOS

    seeds = [1000 + i for i in range(args.n_seeds)]

    # Phase 1: Candidate Exploration (Optuna Bayesian vs Preset Grid)
    if args.strategy == "optuna" and OPTUNA_AVAILABLE:
        eval_res = tune_with_optuna(
            checkpoint_path=str(checkpoint_path),
            scenarios=scenarios,
            seeds=seeds,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout_sec=args.timeout,
            n_workers=args.workers,
            prune=args.prune,
            seed=args.optuna_seed,
            storage=args.storage,
            study_name=args.study_name,
            reset_study=args.reset_study,
            verbose=True
        )
    else:
        if args.candidates_json:
            with open(args.candidates_json) as f:
                raw_cands = json.load(f)
            candidates = [InferenceCandidate(**c) for c in raw_cands]
        else:
            candidates = get_candidate_presets(args.preset)

        eval_res = evaluate_candidates_matched(
            candidates=candidates,
            scenarios=scenarios,
            seeds=seeds,
            checkpoint_path=str(checkpoint_path),
            n_workers=args.workers,
            verbose=True
        )

    # Phase 2: Direct Combat Head-to-Head Tournament
    h2h_res = None
    if args.h2h_rounds > 0 and len(eval_res["ranked_candidates"]) >= 2:
        top_k = min(args.top_k, 4, len(eval_res["ranked_candidates"]))
        top_cands = [
            InferenceCandidate(**r["candidate"])
            for r in eval_res["ranked_candidates"][:top_k]
        ]
        h2h_seeds = [6000 + i for i in range(args.h2h_rounds)]
        h2h_res = run_head_to_head_tournament(
            top_candidates=top_cands,
            checkpoint_path=str(checkpoint_path),
            n_rounds=args.h2h_rounds,
            seeds=h2h_seeds,
            scenario="classic",
            n_workers=args.workers,
            include_dqn_rival=True,
            verbose=True
        )

    # Determine winning candidate (ignoring my_spatial_dqn_agent adversary if present)
    valid_cands = [
        p["candidate"] for p in (h2h_res["ranked_h2h"] if (h2h_res and h2h_res.get("ranked_h2h")) else eval_res["ranked_candidates"])
        if p["candidate"]["name"] != "my_spatial_dqn_agent"
    ]
    winning_cand_dict = valid_cands[0] if valid_cands else eval_res["ranked_candidates"][0]["candidate"]

    # Save to config.json if requested
    saved_cfg_path = None
    if args.save_best:
        cfg_path = DEFAULT_AGENT_DIR / "config.json"
        if cfg_path.is_file():
            with open(cfg_path, "r") as f:
                cur_cfg = json.load(f)
        else:
            cur_cfg = {}
        for k in ["search_depth", "beam_size", "tree_discount", "alpha_vq", "predicted_wait_penalty", "predicted_loop_penalty"]:
            if k in winning_cand_dict:
                cur_cfg[k] = winning_cand_dict[k]
        with open(cfg_path, "w") as f:
            json.dump(cur_cfg, f, indent=2)
        saved_cfg_path = cfg_path

    # Terminal summary tables
    print_tuning_summary(eval_res, h2h_res, save_best_path=saved_cfg_path)

    # Export artifacts
    out_dir = ROOT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_md_path = out_dir / f"tune_inference_report_{timestamp}.md"
    results_json_path = out_dir / f"tune_inference_results_{timestamp}.json"

    export_markdown_tuning_report(eval_res, h2h_res, report_md_path)

    full_export_data = {
        "eval_res": eval_res,
        "h2h_res": h2h_res,
        "winning_candidate": winning_cand_dict,
        "timestamp": timestamp,
    }
    with open(results_json_path, "w") as f:
        json.dump(full_export_data, f, indent=2)

    c = Colors(enabled=True)
    print(c.bold("★ EXPORTED TUNING ARTIFACTS ★"))
    if eval_res.get("strategy") == "optuna" and eval_res.get("storage"):
        print(f"  • Optuna SQLite DB: {eval_res['storage']}")
        print(f"  • Optuna Study Name: {eval_res.get('study_name', 'default')}")
    print(f"  • Markdown Report:   {report_md_path}")
    print(f"  • JSON Results:      {results_json_path}\n")


if __name__ == "__main__":
    main()
