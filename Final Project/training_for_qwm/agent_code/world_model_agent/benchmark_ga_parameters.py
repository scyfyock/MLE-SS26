"""Systematic Benchmark and Evaluation Suite for Genetic Algorithm World Model Planning.

Evaluates GA parameters across three axes:
1. Planning Horizon H (1 to 4)
2. Population Size P (4 to 16)
3. Generation Count G (1 to 6)
Across Bomberman scenarios ('coin-heaven', 'loot-crate', 'classic').

Features:
- Multiprocessing trial accumulation across CPU cores for fast parallel evaluation
- Multi-trial statistical aggregation (mean ± std, survival rate, latency, cache hits)
- Publication-quality analytical plots with uncertainty/std bands
- Export of both aggregated summary and raw trial-level metrics
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Backward compatibility alias
import agent_code.world_model_agent as _wma
sys.modules.setdefault("agent_code.world_model_agent_cf", _wma)
import agent_code.world_model_agent.world_model as _wm_mod
sys.modules.setdefault("agent_code.world_model_agent_cf.world_model", _wm_mod)
import agent_code.world_model_agent.q_network as _qn_mod
sys.modules.setdefault("agent_code.world_model_agent_cf.q_network", _qn_mod)

import numpy.core as _np_core
sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)
sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
sys.modules.setdefault("numpy._core.umath", _np_core.umath)

from numpy.random.bit_generator import BitGenerator as _BitGenerator
from numpy.random import RandomState as _RandomState
import numpy.random._pickle as _nprp

if not getattr(_nprp, "_wma_patched", False):
    _orig_bg_ctor = _nprp.__bit_generator_ctor
    _orig_rs_ctor = _nprp.__randomstate_ctor

    def _patched_bg_ctor(bit_generator="MT19937"):
        if isinstance(bit_generator, type):
            return bit_generator()
        if isinstance(bit_generator, _BitGenerator):
            return bit_generator
        if bit_generator in _nprp.BitGenerators:
            return _nprp.BitGenerators[bit_generator]()
        raise ValueError(f"{bit_generator} is not a known BitGenerator module.")

    def _patched_rs_ctor(bit_generator_name="MT19937", bit_generator_ctor=_patched_bg_ctor):
        if isinstance(bit_generator_name, _BitGenerator):
            return _RandomState(bit_generator_name)
        return _orig_rs_ctor(bit_generator_name, bit_generator_ctor)

    _nprp.__bit_generator_ctor = _patched_bg_ctor
    _nprp.__randomstate_ctor = _patched_rs_ctor
    _orig_load_build = pickle._Unpickler.load_build

    def _hooked_load_build(self):
        state = self.stack[-1]
        inst = self.stack[-2]
        if isinstance(inst, _BitGenerator) and isinstance(state, tuple):
            self.stack[-1] = state[0]
        return _orig_load_build(self)

    pickle._Unpickler.dispatch[pickle.BUILD[0]] = _hooked_load_build
    _nprp._wma_patched = True

from environment import BombeRLeWorld
import events as e
import settings as s
from agent_code.world_model_agent.genetic_planner import GeneticPlannerConfig
from agent_code.world_model_agent.xpu_accelerator import get_target_device, get_device_info


import subprocess
import tempfile

OUTPUT_DIR = REPO_ROOT / "results" / "ga_evaluation"


def run_single_trial(
    scenario: str,
    use_ga: bool,
    horizon: int = 3,
    pop_size: int = 8,
    generations: int = 3,
    seed: int = 42,
    max_steps: int = 80,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one matched-seed game trial in an isolated subprocess to prevent thread explosion."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_game:
        game_stats_path = Path(tmp_game.name)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_ga:
        ga_stats_path = Path(tmp_ga.name)

    env = os.environ.copy()
    # Strictly limit CPU thread counts per worker to prevent runaway CPU saturation and freezing
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    if device:
        env["WMA_DEVICE"] = str(device)
    env["USE_GA_PLANNER"] = "1" if use_ga else "0"
    env["GA_HORIZON"] = str(horizon)
    env["GA_POP_SIZE"] = str(pop_size)
    env["GA_GENERATIONS"] = str(generations)
    env["GA_STATS_FILE"] = str(ga_stats_path)

    # Scenarios agent setup
    if scenario == "classic":
        agents = [
            "world_model_agent",
            "rule_based_agent",
            "rule_based_agent",
            "rule_based_agent",
        ]
    else:
        agents = ["world_model_agent"]

    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "play",
        "--agents", *agents,
        "--scenario", scenario,
        "--seed", str(seed),
        "--n-rounds", "1",
        "--no-gui",
        "--silence-errors",
        "--save-stats", str(game_stats_path),
    ]

    t_start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
        duration_sec = time.perf_counter() - t_start

        # Read game stats JSON
        score = 0
        coins = 0
        crates = 0
        kills = 0
        survived = True
        step_count = max_steps

        if game_stats_path.is_file() and game_stats_path.stat().st_size > 0:
            try:
                with open(game_stats_path, "r") as gf:
                    gdata = json.load(gf)
                by_round = gdata.get("by_round", {})
                if by_round:
                    r_val = list(by_round.values())[0]
                    wma_info = r_val.get("by_agent", {}).get("world_model_agent", {})
                    score = int(wma_info.get("score", 0))
                    coins = int(wma_info.get("coins", 0))
                    crates = int(wma_info.get("crates", 0))
                    kills = int(wma_info.get("kills", 0))
                    survived = bool(wma_info.get("survived", False))
                    step_count = int(wma_info.get("steps", r_val.get("steps", max_steps)))
            except Exception:
                pass

        # Read GA stats JSON
        latencies = [0.0]
        p95_lat = 0.0
        best_fitnesses = [0.0]
        cache_rates = [0.0]
        avg_gen_fitness = [0.0] * (generations + 1)

        if ga_stats_path.is_file() and ga_stats_path.stat().st_size > 0:
            try:
                with open(ga_stats_path, "r") as gaf:
                    ga_info = json.load(gaf)
                latencies = [ga_info.get("mean_latency_ms", 0.0)]
                p95_lat = float(ga_info.get("p95_latency_ms", 0.0))
                best_fitnesses = [ga_info.get("mean_best_fitness", 0.0)]
                cache_rates = [ga_info.get("mean_cache_hit_rate", 0.0)]
                if ga_info.get("gen_fitness_curve"):
                    avg_gen_fitness = ga_info["gen_fitness_curve"]
            except Exception:
                pass

        return {
            "scenario": scenario,
            "use_ga": use_ga,
            "horizon": horizon if use_ga else 0,
            "pop_size": pop_size if use_ga else 0,
            "generations": generations if use_ga else 0,
            "score": score,
            "coins": coins,
            "crates": crates,
            "kills": kills,
            "survived": survived,
            "steps_survived": step_count,
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": p95_lat if p95_lat > 0 else float(np.mean(latencies)),
            "mean_best_fitness": float(np.mean(best_fitnesses)),
            "mean_cache_hit_rate": float(np.mean(cache_rates)),
            "gen_fitness_curve": avg_gen_fitness,
            "seed": seed,
            "duration_sec": duration_sec,
        }
    finally:
        # Cleanup temporary files
        try:
            if game_stats_path.is_file():
                game_stats_path.unlink()
        except Exception:
            pass
        try:
            if ga_stats_path.is_file():
                ga_stats_path.unlink()
        except Exception:
            pass


def _run_trial_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entrypoint for executing one trial in isolated subprocess."""
    device = task.get("device")
    res = run_single_trial(
        scenario=task["scenario"],
        use_ga=task["use_ga"],
        horizon=task.get("horizon", 3),
        pop_size=task.get("pop_size", 8),
        generations=task.get("generations", 3),
        seed=task["seed"],
        max_steps=task.get("max_steps", 80),
        device=device,
    )
    res["sweep_type"] = task["sweep_type"]
    res["trial_idx"] = task["trial_idx"]
    return res


def _accumulate_trials(raw_results: List[Dict[str, Any]], configs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate multi-trial repetitions per parameter configuration into statistical metrics."""
    grouped = defaultdict(list)
    for r in raw_results:
        key = (r["scenario"], r["sweep_type"], r["horizon"], r["pop_size"], r["generations"])
        grouped[key].append(r)

    accumulated = []
    for cfg in configs:
        key = (cfg["scenario"], cfg["sweep_type"], cfg["horizon"], cfg["pop_size"], cfg["generations"])
        trials = grouped[key]
        if not trials:
            continue

        scores = [float(t["score"]) for t in trials]
        crates = [float(t["crates"]) for t in trials]
        coins = [float(t["coins"]) for t in trials]
        kills = [float(t["kills"]) for t in trials]
        survived = [1.0 if t["survived"] else 0.0 for t in trials]
        steps = [float(t["steps_survived"]) for t in trials]
        latencies = [float(t["mean_latency_ms"]) for t in trials]
        p95s = [float(t["p95_latency_ms"]) for t in trials]
        fitnesses = [float(t["mean_best_fitness"]) for t in trials]
        cache_rates = [float(t["mean_cache_hit_rate"]) for t in trials]

        # Calculate element-wise mean fitness curve across trials
        max_curve_len = max(len(t.get("gen_fitness_curve", [])) for t in trials)
        avg_gen_fitness = []
        for g_idx in range(max_curve_len):
            vals = [t["gen_fitness_curve"][g_idx] for t in trials if len(t.get("gen_fitness_curve", [])) > g_idx]
            avg_gen_fitness.append(float(np.mean(vals)) if vals else 0.0)

        accumulated.append({
            "scenario": cfg["scenario"],
            "sweep_type": cfg["sweep_type"],
            "use_ga": cfg["use_ga"],
            "horizon": cfg["horizon"],
            "pop_size": cfg["pop_size"],
            "generations": cfg["generations"],
            "score": float(np.mean(scores)),
            "score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "crates": float(np.mean(crates)),
            "crates_std": float(np.std(crates, ddof=1)) if len(crates) > 1 else 0.0,
            "coins": float(np.mean(coins)),
            "coins_std": float(np.std(coins, ddof=1)) if len(coins) > 1 else 0.0,
            "kills": float(np.mean(kills)),
            "kills_std": float(np.std(kills, ddof=1)) if len(kills) > 1 else 0.0,
            "survived": bool(np.mean(survived) >= 0.5),
            "survival_rate": float(np.mean(survived) * 100.0),
            "steps_survived": float(np.mean(steps)),
            "steps_survived_std": float(np.std(steps, ddof=1)) if len(steps) > 1 else 0.0,
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.mean(p95s)),
            "mean_best_fitness": float(np.mean(fitnesses)),
            "mean_cache_hit_rate": float(np.mean(cache_rates)),
            "gen_fitness_curve": avg_gen_fitness,
            "n_trials": len(trials),
            "raw_trials": trials,
        })

    return pd.DataFrame(accumulated)


def run_all_benchmarks(
    n_trials: int = 3,
    workers: Optional[int] = None,
    max_steps: int = 80,
    base_seed: int = 42,
    scenarios: Optional[List[str]] = None,
    quick: bool = False,
    device: str = "auto",
    skip_baseline: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run parameter sweep with parallel multiprocessing trial accumulation.

    Returns:
        (accumulated_df, raw_trials_df)
    """
    if scenarios is None:
        scenarios = ["coin-heaven", "loot-crate", "classic"]

    cpu_cores = os.cpu_count() or 4
    if workers is None:
        workers = min(cpu_cores, 8)

    target_device = get_target_device(device)
    device_info = get_device_info(target_device)

    if quick:
        n_trials = n_trials if n_trials > 1 else 1
        max_steps = min(max_steps, 40)
        horizon_list = [1, 2, 3]
        pop_list = [4, 8]
        gen_list = [1, 3]
    else:
        horizon_list = [1, 2, 3, 4, 6]
        pop_list = [4, 8, 16, 24]
        gen_list = [1, 2, 4, 6]

    # Generate benchmark configuration definitions
    configs = []

    # 1. Baseline: Pure Double-DQN (No GA) (skipped by default per user request)
    if not skip_baseline:
        for scen in scenarios:
            configs.append({
                "scenario": scen,
                "use_ga": False,
                "horizon": 0,
                "pop_size": 0,
                "generations": 0,
                "sweep_type": "baseline",
            })

    # 2. Planning Horizon Sweep (fixed P=8, G=3)
    for scen in scenarios:
        for h in horizon_list:
            configs.append({
                "scenario": scen,
                "use_ga": True,
                "horizon": h,
                "pop_size": 8,
                "generations": 3,
                "sweep_type": "horizon",
            })

    # 3. Population Size Sweep (fixed H=3, G=3)
    for scen in scenarios:
        for p in pop_list:
            configs.append({
                "scenario": scen,
                "use_ga": True,
                "horizon": 3,
                "pop_size": p,
                "generations": 3,
                "sweep_type": "pop_size",
            })

    # 4. Generation Count Sweep (fixed H=3, P=8)
    for scen in scenarios:
        for g in gen_list:
            configs.append({
                "scenario": scen,
                "use_ga": True,
                "horizon": 3,
                "pop_size": 8,
                "generations": g,
                "sweep_type": "generations",
            })

    # Expand into individual trial tasks
    tasks = []
    for cfg in configs:
        for trial_idx in range(n_trials):
            seed = base_seed + trial_idx * 100
            task = dict(cfg)
            task["seed"] = seed
            task["trial_idx"] = trial_idx
            task["max_steps"] = max_steps
            task["device"] = device
            tasks.append(task)

    total_tasks = len(tasks)
    print("=" * 80)
    print("STARTING GENETIC ALGORITHM WORLD MODEL BENCHMARK (MULTIPROCESSING + ACCELERATION)")
    print("=" * 80)
    print(f"Configurations      : {len(configs)} unique parameter sets")
    print(f"Trials / Config     : {n_trials} independent seeds per setting")
    print(f"Total Trials        : {total_tasks} parallel games")
    print(f"Parallel Workers    : {workers} worker processes (Detected CPUs: {cpu_cores})")
    print(f"Hardware Device     : {device_info}")
    print(f"Max Steps / Game    : {max_steps} steps")
    print(f"Scenarios           : {scenarios}")
    print("=" * 80)

    start_benchmark_time = time.perf_counter()
    raw_results: List[Dict[str, Any]] = []

    if workers > 1 and total_tasks > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(_run_trial_worker, task): task for task in tasks}
            completed_count = 0
            for future in as_completed(future_to_task):
                completed_count += 1
                task = future_to_task[future]
                try:
                    res = future.result()
                    raw_results.append(res)
                    ga_str = (
                        f"H={res['horizon']} P={res['pop_size']} G={res['generations']}"
                        if res["use_ga"]
                        else "Baseline"
                    )
                    print(
                        f"[{completed_count:3d}/{total_tasks:3d}] "
                        f"{res['scenario']:<11} | {res['sweep_type']:<11} | {ga_str:<14} "
                        f"(trial {res['trial_idx']+1}/{n_trials}) -> "
                        f"Score: {res['score']:2d}, Crates: {res['crates']:2d}, "
                        f"Latency: {res['mean_latency_ms']:5.1f}ms"
                    )
                except Exception as ex:
                    print(f"[{completed_count:3d}/{total_tasks:3d}] ERROR in task {task}: {ex}")
    else:
        # Sequential fallback
        for idx, task in enumerate(tasks):
            res = _run_trial_worker(task)
            raw_results.append(res)
            ga_str = (
                f"H={res['horizon']} P={res['pop_size']} G={res['generations']}"
                if res["use_ga"]
                else "Baseline"
            )
            print(
                f"[{idx+1:3d}/{total_tasks:3d}] "
                f"{res['scenario']:<11} | {res['sweep_type']:<11} | {ga_str:<14} "
                f"(trial {res['trial_idx']+1}/{n_trials}) -> "
                f"Score: {res['score']:2d}, Crates: {res['crates']:2d}, "
                f"Latency: {res['mean_latency_ms']:5.1f}ms"
            )

    elapsed_total_s = time.perf_counter() - start_benchmark_time
    print("-" * 80)
    print(f"All {len(raw_results)} trials finished in {elapsed_total_s:.2f}s "
          f"({elapsed_total_s / max(1, len(raw_results)):.2f}s per trial equivalent).")

    # Accumulate trials into summary statistics
    df = _accumulate_trials(raw_results, configs)
    raw_df = pd.DataFrame(raw_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUTPUT_DIR / "ga_benchmark_results.pkl")
    df.to_csv(OUTPUT_DIR / "ga_benchmark_summary.csv", index=False)
    raw_df.to_pickle(OUTPUT_DIR / "ga_benchmark_raw_trials.pkl")
    raw_df.to_csv(OUTPUT_DIR / "ga_benchmark_raw_trials.csv", index=False)

    print(f"Saved aggregated benchmark results to {OUTPUT_DIR / 'ga_benchmark_results.pkl'}")
    print(f"Saved raw trial records to {OUTPUT_DIR / 'ga_benchmark_raw_trials.pkl'}")
    return df, raw_df


def generate_plots(df: pd.DataFrame, raw_df: Optional[pd.DataFrame] = None):
    """Generate all analytical plots comparing parameters and scenarios with trial uncertainty."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    scenarios = [s for s in ["coin-heaven", "loot-crate", "classic"] if s in df["scenario"].values]
    colors = {"coin-heaven": "#2ca02c", "loot-crate": "#ff7f0e", "classic": "#1f77b4"}

    # -------------------------------------------------------------------------
    # Plot 1: Horizon Sensitivity (Score & Latency vs Horizon)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    h_df = df[df["sweep_type"] == "horizon"]

    # Subplot A: Score vs Horizon
    for scen in scenarios:
        sub = h_df[h_df["scenario"] == scen].sort_values("horizon")
        if sub.empty:
            continue
        axes[0].plot(
            sub["horizon"], sub["score"],
            marker="o", linewidth=2.5, label=scen, color=colors[scen]
        )
        if "score_std" in sub.columns and (sub["score_std"] > 0).any():
            axes[0].fill_between(
                sub["horizon"],
                sub["score"] - sub["score_std"],
                sub["score"] + sub["score_std"],
                alpha=0.18, color=colors[scen]
            )

        # Baseline horizontal line & band
        base_sub = df[(df["sweep_type"] == "baseline") & (df["scenario"] == scen)]
        if not base_sub.empty:
            base_score = base_sub["score"].values[0]
            axes[0].axhline(base_score, linestyle="--", alpha=0.5, color=colors[scen])
            if "score_std" in base_sub.columns and base_sub["score_std"].values[0] > 0:
                base_std = base_sub["score_std"].values[0]
                axes[0].axhspan(base_score - base_std, base_score + base_std, alpha=0.07, color=colors[scen])

    axes[0].set_title("Game Score vs. Planning Horizon (Mean ± Std)", fontweight="bold")
    axes[0].set_xlabel("Planning Horizon (Steps)")
    axes[0].set_ylabel("Final Score (Coins / Points)")
    if not h_df.empty:
        axes[0].set_xticks(sorted(h_df["horizon"].unique()))
    axes[0].legend(title="Scenario (dashed=baseline)")

    # Subplot B: Latency vs Horizon
    for scen in scenarios:
        sub = h_df[h_df["scenario"] == scen].sort_values("horizon")
        if sub.empty:
            continue
        axes[1].plot(
            sub["horizon"], sub["mean_latency_ms"],
            marker="s", linewidth=2.5, label=scen, color=colors[scen]
        )

    axes[1].set_title("Decision Latency vs. Planning Horizon", fontweight="bold")
    axes[1].set_xlabel("Planning Horizon (Steps)")
    axes[1].set_ylabel("Mean Planning Latency (ms / step)")
    if not h_df.empty:
        axes[1].set_xticks(sorted(h_df["horizon"].unique()))
    axes[1].axhline(400, color="red", linestyle=":", label="Bomberman Timeout Limit (~400ms)")
    axes[1].legend()

    plt.savefig(OUTPUT_DIR / "ga_horizon_sensitivity.png", dpi=200)
    plt.close()
    print("Saved -> ga_horizon_sensitivity.png")

    # -------------------------------------------------------------------------
    # Plot 2: Population Size and Generation Count Trade-Offs
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # A: Score vs Pop Size
    pop_df = df[df["sweep_type"] == "pop_size"]
    for scen in scenarios:
        sub = pop_df[pop_df["scenario"] == scen].sort_values("pop_size")
        if sub.empty:
            continue
        axes[0, 0].plot(sub["pop_size"], sub["score"], marker="o", linewidth=2.2, label=scen, color=colors[scen])
        if "score_std" in sub.columns and (sub["score_std"] > 0).any():
            axes[0, 0].fill_between(
                sub["pop_size"],
                sub["score"] - sub["score_std"],
                sub["score"] + sub["score_std"],
                alpha=0.15, color=colors[scen]
            )
    axes[0, 0].set_title("Score vs. Population Size (H=3, G=3)", fontweight="bold")
    axes[0, 0].set_xlabel("Population Size P")
    axes[0, 0].set_ylabel("Score (Mean ± Std)")
    axes[0, 0].legend()

    # B: Latency vs Pop Size
    for scen in scenarios:
        sub = pop_df[pop_df["scenario"] == scen].sort_values("pop_size")
        if sub.empty:
            continue
        axes[0, 1].plot(sub["pop_size"], sub["mean_latency_ms"], marker="s", linewidth=2.2, label=scen, color=colors[scen])
    axes[0, 1].set_title("Latency vs. Population Size", fontweight="bold")
    axes[0, 1].set_xlabel("Population Size P")
    axes[0, 1].set_ylabel("Mean Latency (ms)")

    # C: Score vs Generations
    gen_df = df[df["sweep_type"] == "generations"]
    for scen in scenarios:
        sub = gen_df[gen_df["scenario"] == scen].sort_values("generations")
        if sub.empty:
            continue
        axes[1, 0].plot(sub["generations"], sub["score"], marker="o", linewidth=2.2, label=scen, color=colors[scen])
        if "score_std" in sub.columns and (sub["score_std"] > 0).any():
            axes[1, 0].fill_between(
                sub["generations"],
                sub["score"] - sub["score_std"],
                sub["score"] + sub["score_std"],
                alpha=0.15, color=colors[scen]
            )
    axes[1, 0].set_title("Score vs. Generations Count (H=3, P=8)", fontweight="bold")
    axes[1, 0].set_xlabel("Generations G")
    axes[1, 0].set_ylabel("Score (Mean ± Std)")
    axes[1, 0].legend()

    # D: Latency vs Generations
    for scen in scenarios:
        sub = gen_df[gen_df["scenario"] == scen].sort_values("generations")
        if sub.empty:
            continue
        axes[1, 1].plot(sub["generations"], sub["mean_latency_ms"], marker="s", linewidth=2.2, label=scen, color=colors[scen])
    axes[1, 1].set_title("Latency vs. Generations Count", fontweight="bold")
    axes[1, 1].set_xlabel("Generations G")
    axes[1, 1].set_ylabel("Mean Latency (ms)")

    plt.savefig(OUTPUT_DIR / "ga_pop_generations_tradeoff.png", dpi=200)
    plt.close()
    print("Saved -> ga_pop_generations_tradeoff.png")

    # -------------------------------------------------------------------------
    # Plot 3: Fitness Convergence Curves across Generations
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.5))
    if len(scenarios) == 1:
        axes = [axes]

    for idx, scen in enumerate(scenarios):
        ax = axes[idx]
        gen_sub = df[(df["sweep_type"] == "generations") & (df["scenario"] == scen)]
        for _, row in gen_sub.iterrows():
            curve = row["gen_fitness_curve"]
            gens = list(range(len(curve)))
            ax.plot(gens, curve, marker="o", linewidth=2, label=f"G={int(row['generations'])}")
        ax.set_title(f"Convergence: {scen.upper()}", fontweight="bold")
        ax.set_xlabel("Generation Index")
        ax.set_ylabel("Mean Best Fitness")
        ax.legend()

    plt.savefig(OUTPUT_DIR / "ga_fitness_convergence.png", dpi=200)
    plt.close()
    print("Saved -> ga_fitness_convergence.png")

    # -------------------------------------------------------------------------
    # Plot 4: Scenario Performance Comparison & Pareto Frontier
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart comparing Baseline vs Optimal GA config (or multi-horizon comparison if baseline skipped)
    base = df[df["sweep_type"] == "baseline"]

    x = np.arange(len(scenarios))
    if base.empty:
        # User requested no baseline: compare short, medium, and deep lookahead
        h1 = df[(df["sweep_type"] == "horizon") & (df["horizon"] == 1)]
        h3 = df[(df["sweep_type"] == "horizon") & (df["horizon"] == 3)]
        h6 = df[(df["sweep_type"] == "horizon") & (df["horizon"] == 6)]
        w = 0.25
        h1_scores = [h1[h1["scenario"] == s]["score"].values[0] if not h1[h1["scenario"] == s].empty else 0.0 for s in scenarios]
        h1_stds = [h1[h1["scenario"] == s]["score_std"].values[0] if not h1[h1["scenario"] == s].empty and "score_std" in h1.columns else 0.0 for s in scenarios]
        h3_scores = [h3[h3["scenario"] == s]["score"].values[0] if not h3[h3["scenario"] == s].empty else 0.0 for s in scenarios]
        h3_stds = [h3[h3["scenario"] == s]["score_std"].values[0] if not h3[h3["scenario"] == s].empty and "score_std" in h3.columns else 0.0 for s in scenarios]
        h6_scores = [h6[h6["scenario"] == s]["score"].values[0] if not h6[h6["scenario"] == s].empty else 0.0 for s in scenarios]
        h6_stds = [h6[h6["scenario"] == s]["score_std"].values[0] if not h6[h6["scenario"] == s].empty and "score_std" in h6.columns else 0.0 for s in scenarios]

        axes[0].bar(x - w, h1_scores, w, yerr=h1_stds, capsize=3, label="Horizon H=1", color="#1f77b4", alpha=0.85)
        axes[0].bar(x, h3_scores, w, yerr=h3_stds, capsize=3, label="Horizon H=3", color="#2ca02c", alpha=0.85)
        axes[0].bar(x + w, h6_scores, w, yerr=h6_stds, capsize=3, label="Horizon H=6", color="#d62728", alpha=0.85)
        axes[0].set_title("Lookahead Horizon Comparison across Scenarios", fontweight="bold")
    else:
        opt_ga = df[(df["sweep_type"] == "horizon") & (df["horizon"] == 3)]
        width = 0.35
        base_scores = [
            base[base["scenario"] == s]["score"].values[0]
            if not base[base["scenario"] == s].empty else 0.0
            for s in scenarios
        ]
        base_stds = [
            base[base["scenario"] == s]["score_std"].values[0]
            if not base[base["scenario"] == s].empty and "score_std" in base.columns else 0.0
            for s in scenarios
        ]
        ga_scores = [
            opt_ga[opt_ga["scenario"] == s]["score"].values[0]
            if not opt_ga[opt_ga["scenario"] == s].empty else 0.0
            for s in scenarios
        ]
        ga_stds = [
            opt_ga[opt_ga["scenario"] == s]["score_std"].values[0]
            if not opt_ga[opt_ga["scenario"] == s].empty and "score_std" in opt_ga.columns else 0.0
            for s in scenarios
        ]
        axes[0].bar(x - width/2, base_scores, width, yerr=base_stds, capsize=4, label="Double-DQN Baseline", color="#7f7f7f", alpha=0.85)
        axes[0].bar(x + width/2, ga_scores, width, yerr=ga_stds, capsize=4, label="World Model GA (H=3, P=8, G=3)", color="#2ca02c", alpha=0.85)
        axes[0].set_title("Baseline vs. GA Planner across Scenarios", fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios)
    axes[0].set_ylabel("Score / Coins Collected (Mean ± Std)")
    axes[0].legend()

    # Pareto Scatter: Score vs Latency for all trials
    if raw_df is not None and not raw_df.empty:
        # Plot raw individual trials with light opacity
        for scen in scenarios:
            raw_sub = raw_df[raw_df["scenario"] == scen]
            axes[1].scatter(
                raw_sub["mean_latency_ms"], raw_sub["score"],
                s=25, alpha=0.25, color=colors[scen]
            )

    # Plot configuration means prominently
    for scen in scenarios:
        scen_all = df[df["scenario"] == scen]
        axes[1].scatter(
            scen_all["mean_latency_ms"], scen_all["score"],
            s=80, alpha=0.9, edgecolors="black", linewidths=0.7,
            label=f"{scen} (mean)", color=colors[scen]
        )

    axes[1].set_title("Pareto Analysis: Game Score vs. Decision Latency", fontweight="bold")
    axes[1].set_xlabel("Decision Latency (ms / step)")
    axes[1].set_ylabel("Game Score (Mean)")
    axes[1].legend()

    plt.savefig(OUTPUT_DIR / "ga_scenario_comparison_and_pareto.png", dpi=200)
    plt.close()
    print("Saved -> ga_scenario_comparison_and_pareto.png")


def main():
    parser = argparse.ArgumentParser(
        description="Systematic Benchmark and Evaluation Suite for Genetic Algorithm World Model Planning with Multiprocessing"
    )
    parser.add_argument(
        "--n-trials", "-n",
        type=int,
        default=100,
        help="Number of trials per parameter setting for statistical accumulation (default: 3)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=12,
        help="Number of parallel worker processes (default: 4 for stability)",
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=80,
        help="Maximum game steps per trial (default: 80)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed for matched-seed reproducibility (default: 42)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["coin-heaven", "loot-crate", "classic"],
        help="List of scenarios to benchmark (default: coin-heaven loot-crate classic)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test benchmark (1 trial, 40 steps, reduced parameter grid)",
    )
    import torch
    default_dev = "cuda" if torch.cuda.is_available() else "auto"
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=default_dev,
        choices=["auto", "xpu", "cuda", "cpu"],
        help=f"Hardware accelerator device for neural inference (default: {default_dev})",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        default=True,
        help="Skip pure Double-DQN baseline (no GA) trials (default: True per user request)",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        default=False,
        help="Include baseline Double-DQN trials",
    )

    args = parser.parse_args()

    skip_baseline = args.skip_baseline and not args.include_baseline

    df, raw_df = run_all_benchmarks(
        n_trials=args.n_trials,
        workers=args.workers,
        max_steps=args.max_steps,
        base_seed=args.base_seed,
        scenarios=args.scenarios,
        quick=args.quick,
        device=args.device,
        skip_baseline=skip_baseline,
    )
    generate_plots(df, raw_df=raw_df)

    # Print summary table
    print("\n" + "=" * 94)
    print("BENCHMARK SUMMARY TABLE (KEY CONFIGURATIONS ACCUMULATED ACROSS TRIALS)")
    print("=" * 94)

    display_rows = []
    for _, row in df.iterrows():
        score_str = (
            f"{row['score']:.2f} +/- {row['score_std']:.2f}"
            if row.get("score_std", 0.0) > 0
            else f"{row['score']:.1f}"
        )
        crates_str = (
            f"{row['crates']:.2f} +/- {row['crates_std']:.2f}"
            if row.get("crates_std", 0.0) > 0
            else f"{row['crates']:.1f}"
        )
        display_rows.append({
            "scenario": row["scenario"],
            "sweep_type": row["sweep_type"],
            "horizon": int(row["horizon"]),
            "pop_size": int(row["pop_size"]),
            "generations": int(row["generations"]),
            "score (mean+/-std)": score_str,
            "crates (mean+/-std)": crates_str,
            "surv%": f"{row['survival_rate']:.0f}%",
            "latency_ms": f"{row['mean_latency_ms']:.1f}",
            "cache_hit%": f"{row['mean_cache_hit_rate']:.1f}%",
            "trials": int(row["n_trials"]),
        })

    disp_df = pd.DataFrame(display_rows)
    print(disp_df.to_string(index=False))
    print("=" * 94)


if __name__ == "__main__":
    main()
