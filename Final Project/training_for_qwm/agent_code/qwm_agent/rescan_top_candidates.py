#!/usr/bin/env python3
"""
rescan_top_candidates.py - Deep Rescan & Multi-Scenario Direct Tournament for Pure QWM Agent.

Explicitly dedicated to `qwm_agent` (pure neural world model tree search).

Key Features:
  1. Loads the top-k candidates discovered during Optuna Bayesian search from SQLite storage
     (defaults to agent_code/qwm_agent/optuna_study.db, study 'qwm_agent_inference_tuning').
  2. Rescans top candidates across diverse scenarios with significantly more matched seeds (episodes).
  3. Separates Point Wins (highest positive score) and Survival Wins (sole survivor).
  4. Provides explicit per-scenario / per-phase performance breakdowns for each game scenario.
  5. Pits the top candidates (if k <= 4) directly against each other in multi-scenario 4-player
     deathmatches with rotated corner positions.
  6. Exports rich ANSI console tables, GitHub-flavored Markdown reports, and full JSON artifacts.
  7. Supports --save-best to write optimal parameters directly into agent_code/qwm_agent/config.json.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
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
from typing import List, Dict, Any, Optional, Tuple

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


FILE_PATH = Path(__file__).resolve()
AGENT_DIR = FILE_PATH.parent
AGENT_CODE_DIR = AGENT_DIR.parent
ROOT_DIR = AGENT_CODE_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


# -----------------------------------------------------------------------------
# Terminal Formatting & Box-Drawing Tables
# -----------------------------------------------------------------------------

ANSI_REGEX = re.compile(r'\x1b\[[0-9;]*m')


def visible_len(text: Any) -> int:
    """Returns character length excluding ANSI escape codes."""
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


def render_table(
    headers: List[str],
    rows: List[List[Any]],
    aligns: Optional[List[str]] = None,
    title: Optional[str] = None
) -> str:
    """Renders an elegant Unicode box-drawing table."""
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
    h_line, v_line = "─", "│"

    lines = []
    if title:
        c = Colors(enabled=True)
        lines.append(c.bold(f"\n{title}"))

    # Top border
    lines.append(tl + tm.join(h_line * w for w in col_widths) + tr)

    # Header
    hdr_cells = []
    for h, w in zip(str_headers, col_widths):
        pad = w - visible_len(h)
        hdr_cells.append(" " + h + " " * (pad - 1))
    lines.append(v_line + v_line.join(hdr_cells) + v_line)

    # Middle border
    lines.append(ml + mm.join(h_line * w for w in col_widths) + mr)

    # Rows
    for row in str_rows:
        row_cells = []
        for cell, w, align in zip(row, col_widths, aligns):
            pad = w - visible_len(cell)
            if align == 'right':
                row_cells.append(" " * (pad - 1) + cell + " ")
            elif align == 'center':
                left_pad = pad // 2
                right_pad = pad - left_pad
                row_cells.append(" " * left_pad + cell + " " * right_pad)
            else:
                row_cells.append(" " + cell + " " * (pad - 1))
        lines.append(v_line + v_line.join(row_cells) + v_line)

    # Bottom border
    lines.append(bl + bm.join(h_line * w for w in col_widths) + br)
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class CandidateConfig:
    name: str
    search_depth: int
    beam_size: int
    tree_discount: float
    alpha_vq: float
    source_trial: Optional[int] = None
    optuna_score: Optional[float] = None
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
    candidate: CandidateConfig
    scenario: ScenarioDef
    seed: int
    checkpoint_path: str
    root_dir: str
    timeout_sec: int = 180


@dataclass
class MatchResult:
    candidate_name: str
    scenario_key: str
    seed: int
    score: float = 0.0
    point_win: bool = False
    survival_win: bool = False
    total_win: bool = False
    survived: bool = False
    coins: int = 0
    kills: int = 0
    crates: int = 0
    suicides: int = 0
    got_killed: int = 0
    steps: int = 0
    think_time_ms: float = 0.0
    success: bool = True
    error_msg: str = ""


# Default benchmark scenarios covering diverse game phases
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
# Top-K Candidate Extraction from Optuna Study
# -----------------------------------------------------------------------------

def extract_top_k_from_optuna(
    storage: str,
    study_name: str = "qwm_agent_inference_tuning",
    top_k: int = 4
) -> List[CandidateConfig]:
    """
    Loads Optuna study from persistent SQLite storage and extracts the
    top-k unique hyperparameter candidates for qwm_agent.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required to extract top-k candidates from storage. Run 'pip install optuna'.")

    storage_url = storage if "://" in storage else f"sqlite:///{Path(storage).resolve()}"

    # Auto-detect study name if omitted or not found directly
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        if summaries:
            matched = [s for s in summaries if study_name in s.study_name or "qwm" in s.study_name]
            study_name = (matched[0] if matched else summaries[0]).study_name
    except Exception:
        pass

    print(f"[Optuna] Loading study '{study_name}' from: {storage_url}")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as exc:
        raise FileNotFoundError(f"Could not load Optuna study '{study_name}' from {storage_url}: {exc}")

    # Prioritize completed trials, fall back to running trials if interrupted
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not complete_trials:
        complete_trials = [t for t in study.trials if t.params]
        complete_trials.sort(key=lambda t: (t.value is not None, t.value or 0.0), reverse=True)
    else:
        complete_trials.sort(key=lambda t: t.value or 0.0, reverse=True)

    if not complete_trials:
        raise ValueError(f"No trials with valid parameters found in study '{study_name}'.")

    # Deduplicate candidates by unique parameter tuple
    seen_configs = set()
    top_candidates: List[CandidateConfig] = []

    for trial in complete_trials:
        p = trial.params
        d = int(p.get("search_depth", 4))
        b = int(p.get("beam_size", 12))
        disc = round(float(p.get("tree_discount", 0.12)), 3)
        alpha = round(float(p.get("alpha_vq", 0.45)), 3)

        config_key = (d, b, disc, alpha)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)

        rank_idx = len(top_candidates) + 1
        name = f"Top{rank_idx}-D{d}-B{b}"
        desc = f"Depth={d}, Beam={b}, λ={disc:.2f}, α={alpha:.2f}"

        top_candidates.append(CandidateConfig(
            name=name,
            search_depth=d,
            beam_size=b,
            tree_discount=disc,
            alpha_vq=alpha,
            source_trial=trial.number,
            optuna_score=trial.value,
            description=desc
        ))

        if len(top_candidates) >= top_k:
            break

    return top_candidates


# -----------------------------------------------------------------------------
# Checkpoint Resolution
# -----------------------------------------------------------------------------

def resolve_checkpoint(requested: Optional[str] = None) -> Path:
    """Finds the qwm_agent model checkpoint (.pt)."""
    if requested:
        p = Path(requested)
        if p.is_file(): return p.resolve()
        p2 = ROOT_DIR / requested
        if p2.is_file(): return p2.resolve()
        p3 = AGENT_DIR / requested
        if p3.is_file(): return p3.resolve()
        raise FileNotFoundError(f"Requested checkpoint not found: {requested}")

    for fname in ["my-saved-model-spatial-qwm.pt", "my-saved-model-spatial-qwm5.pt", "my-saved-model-spatial-qwm4.pt"]:
        p = AGENT_DIR / fname
        if p.is_file(): return p.resolve()

    runs_dir = AGENT_DIR / "runs"
    if runs_dir.is_dir():
        pts = sorted(runs_dir.glob("**/best_model_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if pts: return pts[0].resolve()

    raise FileNotFoundError(f"Could not find model checkpoint (.pt) in {AGENT_DIR}")


# -----------------------------------------------------------------------------
# Phase 1: Deep Matched-Game Evaluation Subprocess Worker
# -----------------------------------------------------------------------------

def run_matched_game_worker(task: MatchTask) -> MatchResult:
    """
    Executes a single matched game for qwm_agent in an isolated subprocess.
    Extracts Point Wins vs Survival Wins separately from round statistics.
    """
    cand = task.candidate
    scen = task.scenario

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        stats_path = Path(tmp_file.name)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"

    # Hyperparameter overrides via environment variables for pure QWM
    env["MY_QWM_MODEL_FILE"] = str(task.checkpoint_path)
    env["MY_QWM_TREE_SEARCH"] = "1"
    env["MY_QWM_SEARCH_DEPTH"] = str(cand.search_depth)
    env["MY_QWM_BEAM_SIZE"] = str(cand.beam_size)
    env["MY_QWM_TREE_DISCOUNT"] = str(cand.tree_discount)
    env["MY_QWM_ALPHA_VQ"] = str(cand.alpha_vq)

    cmd = [
        sys.executable,
        str(Path(task.root_dir) / "main.py"),
        "play",
        "--agents", "qwm_agent", *scen.opponents,
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
                error_msg=f"Exit {proc.returncode}: {err_text}"
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
        by_agent = round_data.get("by_agent", {})

        # Locate qwm_agent in round data
        target_name = "qwm_agent"
        agent_data = None
        if target_name in by_agent:
            agent_data = by_agent[target_name]
        else:
            for k, v in by_agent.items():
                if "qwm_agent" in k:
                    agent_data = v
                    target_name = k
                    break

        if agent_data is None:
            return MatchResult(
                candidate_name=cand.name,
                scenario_key=scen.key,
                seed=task.seed,
                success=False,
                error_msg="qwm_agent not found in round stats"
            )

        # Extract per-agent core statistics
        score = float(agent_data.get("score", 0))
        coins = int(agent_data.get("coins", 0))
        kills = int(agent_data.get("kills", 0))
        crates = int(agent_data.get("crates", 0))
        suicides = int(agent_data.get("suicides", 0))
        got_k = int(agent_data.get("got_killed", 0))
        steps = int(agent_data.get("steps", 0))
        survived = bool(agent_data.get("survived", 0)) or (suicides == 0 and got_k == 0 and not agent_data.get("dead", False))

        # --- SEPARATION OF POINT WINS VS SURVIVAL WINS ---
        all_scores = {k: v.get("score", 0) for k, v in by_agent.items()}
        max_score = max(all_scores.values()) if all_scores else 0
        point_win = (score == max_score and score > 0 and list(all_scores.values()).count(max_score) == 1)

        # Surviving agents count (alive at round end)
        alive_agents = [k for k, v in by_agent.items() if not v.get("dead", True) or v.get("survived", 0) > 0]
        survival_win = (len(alive_agents) == 1 and alive_agents[0] == target_name)

        # Total win: either highest positive score or sole survivor (matches engine's won stat)
        engine_won = bool(agent_data.get("won", 0))
        total_win = point_win or survival_win or engine_won

        # Think time calculation
        top_by_agent = raw.get("by_agent", {})
        agent_top = top_by_agent.get(target_name, {})
        total_time = float(agent_top.get("time", 0.0))
        think_time_ms = (total_time / steps * 1000.0) if steps > 0 else 0.0

        return MatchResult(
            candidate_name=cand.name,
            scenario_key=scen.key,
            seed=task.seed,
            score=score,
            point_win=point_win,
            survival_win=survival_win,
            total_win=total_win,
            survived=survived,
            coins=coins,
            kills=kills,
            crates=crates,
            suicides=suicides,
            got_killed=got_k,
            steps=steps,
            think_time_ms=think_time_ms,
            success=True
        )

    except subprocess.TimeoutExpired:
        return MatchResult(
            candidate_name=cand.name,
            scenario_key=scen.key,
            seed=task.seed,
            success=False,
            error_msg=f"Match timed out after {task.timeout_sec}s"
        )
    except Exception as e:
        return MatchResult(
            candidate_name=cand.name,
            scenario_key=scen.key,
            seed=task.seed,
            success=False,
            error_msg=str(e)
        )
    finally:
        if stats_path.is_file():
            stats_path.unlink()


# -----------------------------------------------------------------------------
# Phase 2: Multi-Scenario Head-to-Head Tournament Subprocess Worker & Runner
# -----------------------------------------------------------------------------

def _create_candidate_wrapper(
    wrapper_name: str,
    cand: CandidateConfig,
    checkpoint_path: str
) -> Path:
    """Generates an isolated candidate agent directory in agent_code/ inheriting from qwm_agent."""
    target_dir = AGENT_CODE_DIR / wrapper_name
    target_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "model_file": str(checkpoint_path),
        "tree_search": True,
        "search_depth": cand.search_depth,
        "beam_size": cand.beam_size,
        "tree_discount": cand.tree_discount,
        "alpha_vq": cand.alpha_vq,
        "candidate_name": cand.name,
    }

    with open(target_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    cb_code = f'''# Auto-generated tournament wrapper for pure QWM: {cand.name}
import json
import os
from pathlib import Path
import sys
import importlib

SRC_DIR = Path(__file__).resolve().parent.parent / "qwm_agent"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

orig_cb = importlib.import_module("agent_code.qwm_agent.callbacks")

def setup(self):
    cfg_file = Path(__file__).parent / 'config.json'
    if cfg_file.is_file():
        with open(cfg_file) as f:
            cfg = json.load(f)
    else:
        cfg = {{}}
    self.cfg = cfg
    orig_cb.setup(self)
    for k in ['search_depth', 'beam_size', 'tree_discount', 'alpha_vq']:
        if k in cfg:
            setattr(self, k, cfg[k])

def act(self, game_state: dict):
    return orig_cb.act(self, game_state)
'''
    with open(target_dir / "callbacks.py", "w") as f:
        f.write(cb_code)

    return target_dir


def _h2h_match_worker(args_tuple: Tuple[List[str], int, str, int, str]) -> Dict[str, Any]:
    """Runs a single 4-player head-to-head match and records point/survival wins."""
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
            return {"success": False, "seed": seed, "scenario": scenario, "error": f"Exit {proc.returncode}: {err}"}

        with open(stats_path, "r") as f:
            raw = json.load(f)

        by_round = raw.get("by_round", {})
        if not by_round:
            return {"success": False, "seed": seed, "scenario": scenario, "error": "Empty stats"}

        first_id = next(iter(by_round.keys()))
        round_agents = by_round[first_id].get("by_agent", {})

        # Compute Point Wins and Survival Wins
        all_scores = {k: v.get("score", 0) for k, v in round_agents.items()}
        max_score = max(all_scores.values()) if all_scores else 0
        alive_agents = [k for k, v in round_agents.items() if not v.get("dead", True) or v.get("survived", 0) > 0]

        agent_scores = {}
        for a_name, a_data in round_agents.items():
            s = float(a_data.get("score", 0))
            is_alive = (not a_data.get("dead", True)) or bool(a_data.get("survived", 0))
            pt_win = (s == max_score and s > 0 and list(all_scores.values()).count(max_score) == 1)
            surv_win = (len(alive_agents) == 1 and alive_agents[0] == a_name)
            tot_win = pt_win or surv_win or bool(a_data.get("won", 0))

            agent_scores[a_name] = {
                "score": s,
                "point_win": pt_win,
                "survival_win": surv_win,
                "total_win": tot_win,
                "survived": is_alive,
                "kills": int(a_data.get("kills", 0)),
                "coins": int(a_data.get("coins", 0)),
                "suicides": int(a_data.get("suicides", 0)),
            }

        return {
            "success": True,
            "seed": seed,
            "scenario": scenario,
            "agents": agent_scores
        }
    except Exception as exc:
        return {"success": False, "seed": seed, "scenario": scenario, "error": str(exc)}
    finally:
        stats_path.unlink(missing_ok=True)


def run_multi_scenario_h2h_tournament(
    top_candidates: List[CandidateConfig],
    checkpoint_path: str,
    h2h_scenarios: List[str],
    n_rounds_per_scen: int = 20,
    n_workers: int = 8,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Runs multi-scenario head-to-head tournament for the top qwm_agent candidates (k <= 4).
    Pits them directly against each other across each requested scenario.
    """
    c = Colors(enabled=True)
    participants = top_candidates[:4]
    wrapper_names = [f"_rescan_qwm_h2h_{i}" for i in range(len(participants))]
    cand_by_wrapper = {w: p for w, p in zip(wrapper_names, participants)}

    created_dirs: List[Path] = []
    try:
        for w_name, cand in zip(wrapper_names, participants):
            wdir = _create_candidate_wrapper(w_name, cand, checkpoint_path)
            created_dirs.append(wdir)

        active_roster = list(wrapper_names)
        if len(active_roster) < 4:
            active_roster.append("my_spatial_dqn_agent")
        while len(active_roster) < 4:
            active_roster.append("rule_based_agent")

        if verbose:
            print(c.bold("\n" + "="*85))
            print(c.bold(f"★ PHASE 2: DIRECT MULTI-SCENARIO TOURNAMENT ({n_rounds_per_scen * len(h2h_scenarios)} TOTAL MATCHES) ★"))
            print(c.bold("="*85))
            print(f"Combatants:  {', '.join(f'{p.name} ({w})' for w, p in cand_by_wrapper.items())}")
            if "my_spatial_dqn_agent" in active_roster:
                print(f"Rival Agent: my_spatial_dqn_agent (learned CNN adversary)")
            print(f"Scenarios:   {', '.join(h2h_scenarios)} ({n_rounds_per_scen} rounds each)")
            print(f"Workers:     {n_workers} concurrent processes\n")
            sys.stdout.flush()

        all_tasks = []
        for scen_idx, scen_name in enumerate(h2h_scenarios):
            base_seed = 6000 + scen_idx * 1000
            for r_idx in range(n_rounds_per_scen):
                seed = base_seed + r_idx
                rot = r_idx % len(active_roster)
                rotated_roster = active_roster[rot:] + active_roster[:rot]
                all_tasks.append((rotated_roster, seed, scen_name, 200, str(ROOT_DIR)))

        start_time = time.time()
        round_results: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_h2h_match_worker, task) for task in all_tasks]

            if TQDM_AVAILABLE and verbose:
                pbar = tqdm(
                    total=len(all_tasks),
                    desc="Phase 2 Deathmatches",
                    unit="match",
                    dynamic_ncols=True,
                    leave=True
                )
                for f in as_completed(futures):
                    res = f.result()
                    if res.get("success"):
                        round_results.append(res)
                    pbar.update(1)
                    pbar.set_postfix({
                        "scen": res.get("scenario", "")[:10],
                        "valid": len(round_results)
                    })
                pbar.close()
            else:
                completed = 0
                for f in as_completed(futures):
                    completed += 1
                    res = f.result()
                    if res.get("success"):
                        round_results.append(res)
                    if verbose and (completed % max(1, len(all_tasks) // 10) == 0 or completed == len(all_tasks)):
                        print(f"  [Tournament: {completed:3d}/{len(all_tasks):3d} rounds completed]")
                        sys.stdout.flush()

        elapsed = time.time() - start_time
        if verbose:
            print(c.bold(f"\nTournament completed in {elapsed:.1f}s.\n"))

        def make_stat_dict(cand_dict: Dict[str, Any], wrapper_id: str) -> Dict[str, Any]:
            return {
                "candidate": cand_dict,
                "wrapper": wrapper_id,
                "rounds_played": 0,
                "point_wins": 0,
                "survival_wins": 0,
                "total_wins": 0,
                "survivals": 0,
                "total_score": 0.0,
                "total_kills": 0,
                "total_coins": 0,
                "total_suicides": 0,
            }

        overall_stats: Dict[str, Dict[str, Any]] = {}
        per_scenario_stats: Dict[str, Dict[str, Dict[str, Any]]] = {scen: {} for scen in h2h_scenarios}

        for w_name in wrapper_names:
            c_name = cand_by_wrapper[w_name].name
            overall_stats[c_name] = make_stat_dict(cand_by_wrapper[w_name].to_dict(), w_name)
            for scen in h2h_scenarios:
                per_scenario_stats[scen][c_name] = make_stat_dict(cand_by_wrapper[w_name].to_dict(), w_name)

        if "my_spatial_dqn_agent" in active_roster:
            dqn_cand = {"name": "my_spatial_dqn_agent", "search_depth": 0, "beam_size": 0, "tree_discount": 0, "alpha_vq": 0}
            overall_stats["my_spatial_dqn_agent"] = make_stat_dict(dqn_cand, "my_spatial_dqn_agent")
            for scen in h2h_scenarios:
                per_scenario_stats[scen]["my_spatial_dqn_agent"] = make_stat_dict(dqn_cand, "my_spatial_dqn_agent")

        for r in round_results:
            scen_name = r["scenario"]
            agents_in_round = r["agents"]
            for a_key in overall_stats.keys():
                w_key = overall_stats[a_key]["wrapper"]
                if w_key in agents_in_round:
                    ad = agents_in_round[w_key]
                    for stat_target in [overall_stats[a_key], per_scenario_stats[scen_name][a_key]]:
                        stat_target["rounds_played"] += 1
                        stat_target["total_score"] += ad["score"]
                        stat_target["total_kills"] += ad["kills"]
                        stat_target["total_coins"] += ad["coins"]
                        stat_target["total_suicides"] += ad["suicides"]
                        if ad["point_win"]: stat_target["point_wins"] += 1
                        if ad["survival_win"]: stat_target["survival_wins"] += 1
                        if ad["total_win"]: stat_target["total_wins"] += 1
                        if ad["survived"]: stat_target["survivals"] += 1

        def finalize_stats(stats_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
            ranked = []
            for name, st in stats_dict.items():
                rp = max(1, st["rounds_played"])
                st["point_win_rate"] = (st["point_wins"] / rp) * 100.0
                st["survival_win_rate"] = (st["survival_wins"] / rp) * 100.0
                st["total_win_rate"] = (st["total_wins"] / rp) * 100.0
                st["survival_rate"] = (st["survivals"] / rp) * 100.0
                st["mean_score"] = st["total_score"] / rp
                st["mean_kills"] = st["total_kills"] / rp
                st["mean_coins"] = st["total_coins"] / rp
                ranked.append(st)
            ranked.sort(key=lambda x: (x["total_wins"], x["mean_score"], x["survival_rate"]), reverse=True)
            return ranked

        ranked_overall = finalize_stats(overall_stats)
        ranked_by_scenario = {scen: finalize_stats(per_scenario_stats[scen]) for scen in h2h_scenarios}

        return {
            "ranked_overall": ranked_overall,
            "ranked_by_scenario": ranked_by_scenario,
            "valid_rounds": len(round_results),
            "total_rounds": len(all_tasks),
            "h2h_scenarios": h2h_scenarios,
            "elapsed_seconds": elapsed,
        }

    finally:
        for d in created_dirs:
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


# -----------------------------------------------------------------------------
# Metric Aggregation & Scenario Summaries
# -----------------------------------------------------------------------------

def aggregate_phase1_results(
    candidates: List[CandidateConfig],
    scenarios: List[ScenarioDef],
    results: List[MatchResult]
) -> Dict[str, Any]:
    """
    Computes per-scenario and cross-scenario composite summaries for each candidate,
    explicitly calculating point wins and survival wins.
    """
    summary_by_cand: Dict[str, Dict[str, Any]] = {}

    for cand in candidates:
        c_results = [r for r in results if r.candidate_name == cand.name and r.success]
        scen_map = {}
        for scen in scenarios:
            s_res = [r for r in c_results if r.scenario_key == scen.key]
            n_m = len(s_res)
            if n_m == 0:
                scen_map[scen.key] = {
                    "scenario_key": scen.key,
                    "scenario_name": scen.display_name,
                    "n_matches": 0,
                    "point_wins": 0,
                    "point_win_rate": 0.0,
                    "survival_wins": 0,
                    "survival_win_rate": 0.0,
                    "total_wins": 0,
                    "total_win_rate": 0.0,
                    "survivals": 0,
                    "survival_rate": 0.0,
                    "mean_score": 0.0,
                    "mean_coins": 0.0,
                    "mean_kills": 0.0,
                    "mean_crates": 0.0,
                    "mean_suicides": 0.0,
                    "mean_got_killed": 0.0,
                    "mean_think_ms": 0.0,
                    "scenario_score": 0.0
                }
                continue

            pt_wins = sum(1 for r in s_res if r.point_win)
            surv_wins = sum(1 for r in s_res if r.survival_win)
            tot_wins = sum(1 for r in s_res if r.total_win)
            survs = sum(1 for r in s_res if r.survived)

            m_score = sum(r.score for r in s_res) / n_m
            m_coins = sum(r.coins for r in s_res) / n_m
            m_kills = sum(r.kills for r in s_res) / n_m
            m_crates = sum(r.crates for r in s_res) / n_m
            m_suicides = sum(r.suicides for r in s_res) / n_m
            m_got_k = sum(r.got_killed for r in s_res) / n_m
            m_think = sum(r.think_time_ms for r in s_res) / n_m

            pt_win_rate = (pt_wins / n_m) * 100.0
            surv_win_rate = (surv_wins / n_m) * 100.0
            tot_win_rate = (tot_wins / n_m) * 100.0
            surv_rate = (survs / n_m) * 100.0

            scen_score = (
                tot_win_rate * 40.0 +
                surv_rate * 25.0 +
                m_score * 120.0 +
                m_coins * 30.0 +
                m_kills * 150.0 +
                m_crates * 10.0 -
                m_suicides * 80.0
            ) * scen.weight

            scen_map[scen.key] = {
                "scenario_key": scen.key,
                "scenario_name": scen.display_name,
                "n_matches": n_m,
                "point_wins": pt_wins,
                "point_win_rate": pt_win_rate,
                "survival_wins": surv_wins,
                "survival_win_rate": surv_win_rate,
                "total_wins": tot_wins,
                "total_win_rate": tot_win_rate,
                "survivals": survs,
                "survival_rate": surv_rate,
                "mean_score": m_score,
                "mean_coins": m_coins,
                "mean_kills": m_kills,
                "mean_crates": m_crates,
                "mean_suicides": m_suicides,
                "mean_got_killed": m_got_k,
                "mean_think_ms": m_think,
                "scenario_score": scen_score
            }

        total_valid = len(c_results)
        if total_valid > 0:
            overall_pt_wins = sum(1 for r in c_results if r.point_win)
            overall_surv_wins = sum(1 for r in c_results if r.survival_win)
            overall_tot_wins = sum(1 for r in c_results if r.total_win)
            overall_survs = sum(1 for r in c_results if r.survived)

            overall_pt_win_rate = (overall_pt_wins / total_valid) * 100.0
            overall_surv_win_rate = (overall_surv_wins / total_valid) * 100.0
            overall_tot_win_rate = (overall_tot_wins / total_valid) * 100.0
            overall_surv_rate = (overall_survs / total_valid) * 100.0

            overall_mean_score = sum(r.score for r in c_results) / total_valid
            overall_coins = sum(r.coins for r in c_results) / total_valid
            overall_kills = sum(r.kills for r in c_results) / total_valid
            overall_crates = sum(r.crates for r in c_results) / total_valid
            overall_think = sum(r.think_time_ms for r in c_results) / total_valid

            total_weight = sum(s.weight for s in scenarios)
            weighted_score = sum(scen_map[s.key]["scenario_score"] for s in scenarios) / total_weight if total_weight > 0 else 0.0
        else:
            overall_pt_wins = overall_surv_wins = overall_tot_wins = overall_survs = 0
            overall_pt_win_rate = overall_surv_win_rate = overall_tot_win_rate = overall_surv_rate = 0.0
            overall_mean_score = overall_coins = overall_kills = overall_crates = overall_think = 0.0
            weighted_score = 0.0

        summary_by_cand[cand.name] = {
            "candidate": cand.to_dict(),
            "n_matches": total_valid,
            "overall_score": weighted_score,
            "overall_point_wins": overall_pt_wins,
            "overall_point_win_rate": overall_pt_win_rate,
            "overall_survival_wins": overall_surv_wins,
            "overall_survival_win_rate": overall_surv_win_rate,
            "overall_total_wins": overall_tot_wins,
            "overall_total_win_rate": overall_tot_win_rate,
            "overall_survival_rate": overall_surv_rate,
            "overall_mean_score": overall_mean_score,
            "overall_coins": overall_coins,
            "overall_kills": overall_kills,
            "overall_crates": overall_crates,
            "overall_think_time_ms": overall_think,
            "scenarios": scen_map
        }

    ranked_overall = sorted(summary_by_cand.values(), key=lambda x: x["overall_score"], reverse=True)

    return {
        "summary_by_cand": summary_by_cand,
        "ranked_overall": ranked_overall,
        "total_results": len(results)
    }


# -----------------------------------------------------------------------------
# Terminal Printing & Reporting
# -----------------------------------------------------------------------------

def print_rescan_summary(
    candidates: List[CandidateConfig],
    scenarios: List[ScenarioDef],
    phase1_summary: Dict[str, Any],
    h2h_summary: Optional[Dict[str, Any]] = None,
    save_best_path: Optional[Path] = None
):
    """Renders all terminal tables with explicit scenario breakdowns and win splits."""
    c = Colors(enabled=True)

    print(c.bold("\n" + "="*95))
    print(c.bold("★ PHASE 1: EXPLICIT PER-SCENARIO PERFORMANCE BREAKDOWNS ★"))
    print(c.bold("="*95))

    # Explicit breakdown table for EACH scenario
    for scen in scenarios:
        headers = [
            "Rank", "Candidate", "Total Win %", "Point Win %", "Surv Win %",
            "Surv %", "Score", "Coins", "Kills", "Crates", "Think ms"
        ]
        rows = []
        scen_ranks = []
        for cand in candidates:
            s_data = phase1_summary["summary_by_cand"][cand.name]["scenarios"].get(scen.key, {})
            scen_ranks.append((cand, s_data))

        scen_ranks.sort(key=lambda x: (x[1].get("total_win_rate", 0), x[1].get("mean_score", 0), x[1].get("survival_rate", 0)), reverse=True)

        for idx, (cand, s_data) in enumerate(scen_ranks, 1):
            cand_str = c.bold(cand.name) if idx == 1 else cand.name
            rows.append([
                f"#{idx}",
                cand_str,
                f"{s_data.get('total_win_rate', 0.0):.1f}% ({s_data.get('total_wins', 0)})",
                f"{s_data.get('point_win_rate', 0.0):.1f}% ({s_data.get('point_wins', 0)})",
                f"{s_data.get('survival_win_rate', 0.0):.1f}% ({s_data.get('survival_wins', 0)})",
                f"{s_data.get('survival_rate', 0.0):.1f}%",
                f"{s_data.get('mean_score', 0.0):.2f}",
                f"{s_data.get('mean_coins', 0.0):.2f}",
                f"{s_data.get('mean_kills', 0.0):.2f}",
                f"{s_data.get('mean_crates', 0.0):.1f}",
                f"{s_data.get('mean_think_ms', 0.0):.1f}",
            ])

        print(render_table(
            headers=headers,
            rows=rows,
            title=f"Scenario: {scen.display_name} ({scen.scenario}, {s_data.get('n_matches', 0)} rounds)",
            aligns=['center', 'left', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
        ))

    # Overall Composite Leaderboard across all scenarios
    print(c.bold("\n" + "="*95))
    print(c.bold("★ CROSS-SCENARIO COMPOSITE LEADERBOARD (MATCHED SEEDS) ★"))
    print(c.bold("="*95))
    o_headers = [
        "Rank", "Candidate", "Depth", "Beam", "Disc(λ)", "Alpha(α)",
        "Comp Score", "Total Win %", "Point Win %", "Surv Win %", "Surv %", "Avg Score", "Coins", "Kills", "Think ms"
    ]
    o_rows = []
    for idx, r in enumerate(phase1_summary["ranked_overall"], 1):
        cand = r["candidate"]
        c_str = c.bold(c.green(cand["name"])) if idx == 1 else cand["name"]
        o_rows.append([
            f"#{idx}",
            c_str,
            str(cand["search_depth"]),
            str(cand["beam_size"]),
            f"{cand['tree_discount']:.2f}",
            f"{cand['alpha_vq']:.2f}",
            f"{r['overall_score']:.1f}",
            f"{r['overall_total_win_rate']:.1f}%",
            f"{r['overall_point_win_rate']:.1f}%",
            f"{r['overall_survival_win_rate']:.1f}%",
            f"{r['overall_survival_rate']:.1f}%",
            f"{r['overall_mean_score']:.2f}",
            f"{r['overall_coins']:.2f}",
            f"{r['overall_kills']:.2f}",
            f"{r['overall_think_time_ms']:.1f}",
        ])

    print(render_table(
        headers=o_headers,
        rows=o_rows,
        aligns=['center', 'left', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
    ))

    # Phase 2: Direct Head-to-Head Tournament Results (if executed)
    if h2h_summary:
        print(c.bold("\n" + "="*95))
        print(c.bold("★ PHASE 2: MULTI-SCENARIO DIRECT COMBAT TOURNAMENT ★"))
        print(c.bold("="*95))

        for scen, ranked_scen in h2h_summary["ranked_by_scenario"].items():
            h_headers = ["Place", "Combatant", "Total Wins", "Point Wins", "Surv Wins", "Surv %", "Score / Rnd", "Kills", "Coins"]
            h_rows = []
            for p_idx, p in enumerate(ranked_scen, 1):
                p_name = c.bold(p["candidate"]["name"]) if p_idx == 1 else p["candidate"]["name"]
                medal = "🥇 " if p_idx == 1 else ("🥈 " if p_idx == 2 else ("🥉 " if p_idx == 3 else "   "))
                h_rows.append([
                    f"{medal}#{p_idx}",
                    p_name,
                    f"{p['total_wins']} ({p['total_win_rate']:.1f}%)",
                    f"{p['point_wins']} ({p['point_win_rate']:.1f}%)",
                    f"{p['survival_wins']} ({p['survival_win_rate']:.1f}%)",
                    f"{p['survival_rate']:.1f}%",
                    f"{p['mean_score']:.2f}",
                    f"{p['mean_kills']:.2f}",
                    f"{p['mean_coins']:.2f}",
                ])
            print(render_table(
                headers=h_headers,
                rows=h_rows,
                title=f"Direct Deathmatches: Scenario '{scen}'",
                aligns=['center', 'left', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
            ))

        t_headers = ["Place", "Candidate", "Total Wins", "Point Wins", "Surv Wins", "Overall Surv %", "Score / Rnd", "Kills", "Coins"]
        t_rows = []
        for p_idx, p in enumerate(h2h_summary["ranked_overall"], 1):
            p_name = c.bold(c.green(p["candidate"]["name"])) if p_idx == 1 else p["candidate"]["name"]
            medal = "🥇 " if p_idx == 1 else ("🥈 " if p_idx == 2 else ("🥉 " if p_idx == 3 else "   "))
            t_rows.append([
                f"{medal}#{p_idx}",
                p_name,
                f"{p['total_wins']} ({p['total_win_rate']:.1f}%)",
                f"{p['point_wins']} ({p['point_win_rate']:.1f}%)",
                f"{p['survival_wins']} ({p['survival_win_rate']:.1f}%)",
                f"{p['survival_rate']:.1f}%",
                f"{p['mean_score']:.2f}",
                f"{p['mean_kills']:.2f}",
                f"{p['mean_coins']:.2f}",
            ])
        print(render_table(
            headers=t_headers,
            rows=t_rows,
            title=f"Overall Tournament Leaderboard ({h2h_summary['valid_rounds']} Total Deathmatches)",
            aligns=['center', 'left', 'right', 'right', 'right', 'right', 'right', 'right', 'right']
        ))

    # Champion Recommendation
    if h2h_summary and h2h_summary.get("ranked_overall"):
        non_dqn = [p for p in h2h_summary["ranked_overall"] if p["candidate"]["name"] != "my_spatial_dqn_agent"]
        champ = (non_dqn[0] if non_dqn else h2h_summary["ranked_overall"][0])["candidate"]
        rec_source = "Phase 2 Direct Combat Tournament Champion"
    else:
        champ = phase1_summary["ranked_overall"][0]["candidate"]
        rec_source = "Phase 1 Cross-Scenario Benchmark Leader"

    print(c.bold("\n" + "="*95))
    print(c.bold("★ FINAL BENCHMARK CHAMPION RECOMMENDATION ★"))
    print(c.bold("="*95))
    print(f"  • Source:         {rec_source}")
    print(f"  • Candidate:      {c.green(c.bold(champ['name']))}")
    print(f"  • Search Depth:   {champ['search_depth']}")
    print(f"  • Beam Size:      {champ['beam_size']}")
    print(f"  • Tree Discount:  {champ['tree_discount']:.3f}")
    print(f"  • Alpha VQ:       {champ['alpha_vq']:.3f}")

    if save_best_path:
        print(c.green(f"\n✔ Successfully updated configuration at: {save_best_path}\n"))
    print()


def export_markdown_report(
    candidates: List[CandidateConfig],
    scenarios: List[ScenarioDef],
    phase1_summary: Dict[str, Any],
    h2h_summary: Optional[Dict[str, Any]],
    output_path: Path,
    checkpoint_path: str
):
    """Generates an executive GitHub-Flavored Markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if h2h_summary and h2h_summary.get("ranked_overall"):
        non_dqn = [p for p in h2h_summary["ranked_overall"] if p["candidate"]["name"] != "my_spatial_dqn_agent"]
        champ = (non_dqn[0] if non_dqn else h2h_summary["ranked_overall"][0])["candidate"]
        rec_source = "Phase 2 Direct Combat Tournament"
    else:
        champ = phase1_summary["ranked_overall"][0]["candidate"]
        rec_source = "Phase 1 Cross-Scenario Matched Benchmark"

    md = []
    md.append("# Top-K Candidate Rescan & Tournament Report (`qwm_agent`)\n")
    md.append(f"- **Evaluated Checkpoint:** `{checkpoint_path}`")
    md.append(f"- **Agent Architecture:** `qwm_agent` (Pure Neural World Model)")
    md.append(f"- **Timestamp:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    md.append(f"- **Candidates Benchmarked:** {len(candidates)}")
    md.append(f"- **Total Phase 1 Matches:** {sum(phase1_summary['summary_by_cand'][c.name]['n_matches'] for c in candidates)}\n")

    md.append("## 1. Executive Champion Recommendation\n")
    md.append(f"**Undisputed Winner:** `{champ['name']}` ({rec_source})\n")
    md.append("| Hyperparameter | Optimal Value | Parameter Purpose |")
    md.append("|:---|---:|:---|")
    md.append(f"| `search_depth` | **{champ['search_depth']}** | Lookahead horizon steps |")
    md.append(f"| `beam_size` | **{champ['beam_size']}** | Beam search pruning width |")
    md.append(f"| `tree_discount` ($\\lambda$) | **{champ['tree_discount']:.3f}** | Lookahead prospective reward discount |")
    md.append(f"| `alpha_vq` ($\\alpha$) | **{champ['alpha_vq']:.3f}** | Balance between Q-critic and prospective return |")
    md.append("\n---\n")

    md.append("## 2. Phase 1: Explicit Per-Scenario Performance Breakdowns\n")
    md.append("> [!NOTE]\n> Wins are explicitly separated into **Point Wins** (highest score) and **Survival Wins** (sole survivor).\n")

    for scen in scenarios:
        md.append(f"### Scenario: {scen.display_name} (`{scen.scenario}`)\n")
        md.append("| Rank | Candidate | Total Win % (Wins) | Point Win % (Wins) | Surv Win % (Wins) | Surv % | Score | Coins | Kills | Crates | Think (ms) |")
        md.append("|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        scen_ranks = []
        for cand in candidates:
            s_data = phase1_summary["summary_by_cand"][cand.name]["scenarios"].get(scen.key, {})
            scen_ranks.append((cand, s_data))
        scen_ranks.sort(key=lambda x: (x[1].get("total_win_rate", 0), x[1].get("mean_score", 0), x[1].get("survival_rate", 0)), reverse=True)

        for idx, (cand, s_data) in enumerate(scen_ranks, 1):
            md.append(
                f"| #{idx} | **{cand.name}** | {s_data.get('total_win_rate', 0.0):.1f}% ({s_data.get('total_wins', 0)}) | "
                f"{s_data.get('point_win_rate', 0.0):.1f}% ({s_data.get('point_wins', 0)}) | "
                f"{s_data.get('survival_win_rate', 0.0):.1f}% ({s_data.get('survival_wins', 0)}) | "
                f"{s_data.get('survival_rate', 0.0):.1f}% | {s_data.get('mean_score', 0.0):.2f} | "
                f"{s_data.get('mean_coins', 0.0):.2f} | {s_data.get('mean_kills', 0.0):.2f} | "
                f"{s_data.get('mean_crates', 0.0):.1f} | {s_data.get('mean_think_ms', 0.0):.1f} |"
            )
        md.append("")

    md.append("### Cross-Scenario Weighted Leaderboard\n")
    md.append("| Rank | Candidate | Depth | Beam | Disc ($\\lambda$) | Alpha ($\\alpha$) | Comp Score | Total Win % | Point Win % | Surv Win % | Surv % | Avg Score | Coins | Kills |")
    md.append("|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx, r in enumerate(phase1_summary["ranked_overall"], 1):
        c_c = r["candidate"]
        md.append(
            f"| #{idx} | **{c_c['name']}** | {c_c['search_depth']} | {c_c['beam_size']} | "
            f"{c_c['tree_discount']:.2f} | {c_c['alpha_vq']:.2f} | **{r['overall_score']:.1f}** | "
            f"{r['overall_total_win_rate']:.1f}% | {r['overall_point_win_rate']:.1f}% | "
            f"{r['overall_survival_win_rate']:.1f}% | {r['overall_survival_rate']:.1f}% | "
            f"{r['overall_mean_score']:.2f} | {r['overall_coins']:.2f} | {r['overall_kills']:.2f} |"
        )
    md.append("\n---\n")

    if h2h_summary:
        md.append("## 3. Phase 2: Direct Multi-Scenario Head-to-Head Tournament\n")
        for scen, ranked_scen in h2h_summary["ranked_by_scenario"].items():
            md.append(f"### Direct Combat: Scenario `{scen}`\n")
            md.append("| Place | Combatant | Total Wins | Point Wins | Surv Wins | Surv % | Score / Rnd | Kills | Coins |")
            md.append("|:---|:---|---:|---:|---:|---:|---:|---:|---:|")
            for p_idx, p in enumerate(ranked_scen, 1):
                medal = "🥇 " if p_idx == 1 else ("🥈 " if p_idx == 2 else ("🥉 " if p_idx == 3 else ""))
                md.append(
                    f"| {medal}#{p_idx} | **{p['candidate']['name']}** | {p['total_wins']} ({p['total_win_rate']:.1f}%) | "
                    f"{p['point_wins']} ({p['point_win_rate']:.1f}%) | {p['survival_wins']} ({p['survival_win_rate']:.1f}%) | "
                    f"{p['survival_rate']:.1f}% | {p['mean_score']:.2f} | {p['mean_kills']:.2f} | {p['mean_coins']:.2f} |"
                )
            md.append("")

        md.append(f"### Combined Tournament Leaderboard ({h2h_summary['valid_rounds']} Rounds)\n")
        md.append("| Place | Candidate | Total Wins | Point Wins | Surv Wins | Overall Surv % | Score / Rnd | Kills / Rnd | Coins / Rnd |")
        md.append("|:---|:---|---:|---:|---:|---:|---:|---:|---:|")
        for p_idx, p in enumerate(h2h_summary["ranked_overall"], 1):
            medal = "🥇 " if p_idx == 1 else ("🥈 " if p_idx == 2 else ("🥉 " if p_idx == 3 else ""))
            md.append(
                f"| {medal}#{p_idx} | **{p['candidate']['name']}** | {p['total_wins']} ({p['total_win_rate']:.1f}%) | "
                f"{p['point_wins']} ({p['point_win_rate']:.1f}%) | {p['survival_wins']} ({p['survival_win_rate']:.1f}%) | "
                f"{p['survival_rate']:.1f}% | {p['mean_score']:.2f} | {p['mean_kills']:.2f} | {p['mean_coins']:.2f} |"
            )
        md.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(md))


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deep Rescan & Multi-Scenario Direct Tournament for Pure QWM Agent (agent_code/qwm_agent)."
    )
    parser.add_argument(
        "--agent", type=str, default="qwm_agent",
        help="Target agent (explicitly qwm_agent)."
    )
    parser.add_argument(
        "--top-k", type=int, default=4,
        help="Number of top candidates to extract from Optuna storage (default: 4)."
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help=f"Path or SQLite URL to Optuna persistent storage (default: {AGENT_DIR / 'optuna_study.db'})."
    )
    parser.add_argument(
        "--study-name", type=str, default="qwm_agent_inference_tuning",
        help="Optuna study name (default: qwm_agent_inference_tuning)."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=f"Path to model checkpoint (.pt) (default: {AGENT_DIR / 'my-saved-model-spatial-qwm.pt'})."
    )
    parser.add_argument(
        "--candidates-json", type=str, default=None,
        help="Path to optional JSON file specifying custom candidate configurations."
    )
    parser.add_argument(
        "--n-seeds", type=int, default=100,
        help="Number of matched random seeds per scenario for deep evaluation (default: 20)."
    )
    parser.add_argument(
        "--scenarios", type=str, default="classic_combat,dqn_rivalry,spatial_clash,loot_crate,coin_heaven",
        help="Comma-separated scenario list for Phase 1 (classic_combat, dqn_rivalry, spatial_clash, loot_crate, coin_heaven)."
    )
    parser.add_argument(
        "--h2h-scenarios", type=str, default="coin-heaven,classic,loot-crate",
        help="Comma-separated scenario list for Phase 2 direct combat tournament (default: 'classic,loot-crate')."
    )
    parser.add_argument(
        "--h2h-rounds", type=int, default=100,
        help="Number of direct deathmatches per scenario in Phase 2 tournament (default: 20, set 0 to disable)."
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=14,
        help=f"Number of parallel worker processes (default: {max(1, min(os.cpu_count() or 4, 8))})."
    )
    parser.add_argument(
        "--save-best", action="store_true", default=False,
        help="Automatically write the winning hyperparameters into agent_code/qwm_agent/config.json."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save generated Markdown report and JSON results. Defaults to eval_results."
    )
    return parser


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    c = Colors(enabled=True)

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(c.bold("\n[rescan_top_candidates] Target Agent: qwm_agent (Pure Neural QWM)"))
    print(f"[rescan_top_candidates] Checkpoint:   {checkpoint_path}")

    # 1. Obtain Candidates (From JSON or Optuna Study)
    candidates: List[CandidateConfig] = []
    if args.candidates_json:
        c_json_p = Path(args.candidates_json)
        if not c_json_p.is_file():
            raise FileNotFoundError(f"Candidates JSON file not found: {c_json_p}")
        with open(c_json_p) as f:
            raw_cands = json.load(f)
        for idx, item in enumerate(raw_cands):
            candidates.append(CandidateConfig(
                name=item.get("name", f"Cand-{idx:02d}"),
                search_depth=int(item.get("search_depth", 4)),
                beam_size=int(item.get("beam_size", 12)),
                tree_discount=float(item.get("tree_discount", 0.12)),
                alpha_vq=float(item.get("alpha_vq", 0.45)),
                description=item.get("description", "")
            ))
    else:
        storage_target = args.storage or str(AGENT_DIR / "optuna_study.db")
        candidates = extract_top_k_from_optuna(
            storage=storage_target,
            study_name=args.study_name,
            top_k=args.top_k
        )

    print(c.bold("\n" + "="*85))
    print(c.bold(f"★ SELECTED TOP-{len(candidates)} CANDIDATES FOR DEEP EVALUATION ★"))
    print(c.bold("="*85))
    init_headers = ["Rank", "Candidate", "Depth", "Beam", "Disc (λ)", "Alpha (α)", "Source Trial", "Optuna Score"]
    init_rows = []
    for idx, cand in enumerate(candidates, 1):
        init_rows.append([
            f"#{idx}",
            c.bold(cand.name),
            str(cand.search_depth),
            str(cand.beam_size),
            f"{cand.tree_discount:.3f}",
            f"{cand.alpha_vq:.3f}",
            str(cand.source_trial) if cand.source_trial is not None else "-",
            f"{cand.optuna_score:.1f}" if cand.optuna_score is not None else "-"
        ])
    print(render_table(headers=init_headers, rows=init_rows, aligns=['center', 'left', 'right', 'right', 'right', 'right', 'right', 'right']))

    # 2. Filter Scenarios & Seeds
    requested_scens = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    scenarios = [s for s in DEFAULT_SCENARIOS if any(r in s.key or r in s.scenario for r in requested_scens)]
    if not scenarios:
        scenarios = DEFAULT_SCENARIOS

    seeds = [1000 + i for i in range(args.n_seeds)]

    # 3. Phase 1: Deep Matched-Game Evaluation
    tasks: List[MatchTask] = []
    for cand in candidates:
        for scen in scenarios:
            for seed in seeds:
                tasks.append(MatchTask(
                    candidate=cand,
                    scenario=scen,
                    seed=seed,
                    checkpoint_path=str(checkpoint_path),
                    root_dir=str(ROOT_DIR)
                ))

    total_matches = len(tasks)
    print(c.bold("\n" + "="*85))
    print(c.bold(f"★ PHASE 1: PARALLEL MATCHED-GAME EVALUATION ({total_matches} TOTAL MATCHES) ★"))
    print(c.bold("="*85))
    print(f"Agent:         qwm_agent (Pure Neural)")
    print(f"Candidates:    {len(candidates)}")
    print(f"Scenarios:     {len(scenarios)} ({', '.join(s.key for s in scenarios)})")
    print(f"Matched Seeds: {len(seeds)} seeds per scenario ({seeds[0]}..{seeds[-1]})")
    print(f"Workers:       {args.workers} concurrent match processes\n")
    sys.stdout.flush()

    start_time = time.time()
    results: List[MatchResult] = []

    running_pt_wins = 0
    running_surv_wins = 0
    running_scores = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_matched_game_worker, t): t for t in tasks}

        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=total_matches,
                desc="Phase 1 Rescan",
                unit="match",
                dynamic_ncols=True,
                leave=True
            )
            for f in as_completed(futures):
                res = f.result()
                results.append(res)
                if res.success:
                    if res.point_win: running_pt_wins += 1
                    if res.survival_win: running_surv_wins += 1
                    running_scores.append(res.score)
                pbar.update(1)
                n_c = len(results)
                pbar.set_postfix({
                    "cand": res.candidate_name[:12],
                    "scen": res.scenario_key[:10],
                    "pt_win": f"{(running_pt_wins/n_c)*100:.0f}%",
                    "surv_win": f"{(running_surv_wins/n_c)*100:.0f}%",
                    "avg_pts": f"{(sum(running_scores)/len(running_scores)):.1f}" if running_scores else "0.0"
                })
            pbar.close()
        else:
            completed = 0
            for f in as_completed(futures):
                completed += 1
                res = f.result()
                results.append(res)
                if completed % max(1, total_matches // 10) == 0 or completed == total_matches:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_matches - completed) / rate if rate > 0 else 0
                    print(f"  [{completed:4d}/{total_matches:4d}] matches completed ({completed/total_matches*100:5.1f}%) | ETA: {eta:.0f}s")
                    sys.stdout.flush()

    phase1_summary = aggregate_phase1_results(candidates, scenarios, results)

    # 4. Phase 2: Direct Multi-Scenario Head-to-Head Tournament (if k <= 4 and h2h_rounds > 0)
    h2h_summary = None
    if len(candidates) <= 4 and args.h2h_rounds > 0:
        h2h_scens = [s.strip() for s in args.h2h_scenarios.split(",") if s.strip()]
        h2h_summary = run_multi_scenario_h2h_tournament(
            top_candidates=candidates,
            checkpoint_path=str(checkpoint_path),
            h2h_scenarios=h2h_scens,
            n_rounds_per_scen=args.h2h_rounds,
            n_workers=args.workers,
            verbose=True
        )

    # 5. Save Best Hyperparameters to agent_code/qwm_agent/config.json if requested
    save_path = None
    if args.save_best:
        champ_dict = None
        if h2h_summary and h2h_summary.get("ranked_overall"):
            non_dqn = [p for p in h2h_summary["ranked_overall"] if p["candidate"]["name"] != "my_spatial_dqn_agent"]
            champ_dict = (non_dqn[0] if non_dqn else h2h_summary["ranked_overall"][0])["candidate"]
        else:
            champ_dict = phase1_summary["ranked_overall"][0]["candidate"]

        config_path = AGENT_DIR / "config.json"
        cfg_data = {}
        if config_path.is_file():
            try:
                with open(config_path) as f:
                    cfg_data = json.load(f)
            except Exception:
                cfg_data = {}

        cfg_data["search_depth"] = champ_dict["search_depth"]
        cfg_data["beam_size"] = champ_dict["beam_size"]
        cfg_data["tree_discount"] = champ_dict["tree_discount"]
        cfg_data["alpha_vq"] = champ_dict["alpha_vq"]

        with open(config_path, "w") as f:
            json.dump(cfg_data, f, indent=2)
        save_path = config_path

    # 6. Print Comprehensive Summary Tables
    print_rescan_summary(
        candidates=candidates,
        scenarios=scenarios,
        phase1_summary=phase1_summary,
        h2h_summary=h2h_summary,
        save_best_path=save_path
    )

    # 7. Export Markdown & JSON Reports
    out_dir = Path(args.output_dir) if args.output_dir else (AGENT_DIR / "eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    md_file = out_dir / f"top_k_rescan_qwm_agent_{ts}.md"
    json_file = out_dir / f"top_k_rescan_qwm_agent_{ts}.json"

    export_markdown_report(
        candidates=candidates,
        scenarios=scenarios,
        phase1_summary=phase1_summary,
        h2h_summary=h2h_summary,
        output_path=md_file,
        checkpoint_path=str(checkpoint_path)
    )

    json_data = {
        "agent": "qwm_agent",
        "checkpoint": str(checkpoint_path),
        "timestamp": ts,
        "candidates": [c.to_dict() for c in candidates],
        "phase1": phase1_summary,
        "phase2_h2h": h2h_summary
    }
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    print(c.bold("★ EXPORTED ARTIFACTS ★"))
    print(f"  • Markdown Report: {md_file.resolve()}")
    print(f"  • JSON Metrics:    {json_file.resolve()}\n")


if __name__ == "__main__":
    main()
