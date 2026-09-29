#!/usr/bin/env python3
"""
train_curriculum.py -- High-Performance Curriculum Training Runner for my_spatial_qwm_agent.

Features:
  - Multi-processing parallel training: 8-12 concurrent game instance actors on CPU (Ryzen 7 5700X)
    with central GPU batch training on CUDA (RTX 3080) using AMP FP16 and Tensor Cores (~48,000 samples/s).
  - Phased learning:
      * Stage 0: Latent World Model Warmup (DQN backbone frozen, only predictive model trains)
      * Stages 1-4: Curriculum training with QWM test-time Tree Search planning enabled
  - Dedicated self-play stages with randomized scenarios ('classic', 'loot-crate', 'coin-heaven')
    running after each normal stage and periodically (intra-stage) during longer stages.
  - Single pair of self-play checkpoints per stage (best_model_s{st}_self_play.pt / last_model_s{st}_self_play.pt).
  - Zero-discovery during self-play (epsilon=0.0) for full potential utilization.
  - Seamless epsilon state preservation across self-play without resetting.
  - Live progress curves and telemetry updates (training_curves-spatial-qwm.png).

Usage:
    python agent_code/my_spatial_qwm_agent/train_curriculum.py --workers 8
    python agent_code/my_spatial_qwm_agent/train_curriculum.py --start-stage 0 --stages 4
    python agent_code/my_spatial_qwm_agent/train_curriculum.py --s0 100 --workers 8
    python agent_code/my_spatial_qwm_agent/train_curriculum.py --no-self-play
"""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SELF_PLAY_SCENARIOS = ["classic", "loot-crate", "coin-heaven"]

STAGE_CONFIG = {
    0: {
        "name": "s0_wm_warmup",
        "default_rounds": 150,
        "eps_start": 0.05,
        "scenario": "loot-crate",
        "opponents": ["peaceful_agent", "coin_collector_agent"],
        "lr": 1e-4,
        "freeze_dqn": True,
    },
    1: {
        "name": "s1_coin_heaven",
        "default_rounds": 250,
        "eps_start": 0.5,
        "scenario": "coin-heaven",
        "opponents": ["coin_collector_agent"],
        "lr": 1e-4,
        "freeze_dqn": False,
    },
    2: {
        "name": "s2_classic_alone",
        "default_rounds": 400,
        "eps_start": 0.3,
        "scenario": "loot-crate",
        "opponents": ["coin_collector_agent"],
        "lr": 1e-4,
        "freeze_dqn": False,
    },
    3: {
        "name": "s3_vs_passive",
        "default_rounds": 350,
        "eps_start": 0.2,
        "scenario": "loot-crate",
        "opponents": ["peaceful_agent", "coin_collector_agent"],
        "lr": 1e-4,
        "freeze_dqn": False,
    },
    6: {
        "name": "s4_vs_rule_based",
        "default_rounds": 500,
        "eps_start": 0.10,
        "scenario": "classic",
        "opponents": ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
        "lr": 5e-5,
        "freeze_dqn": False,
    },
    5: {
        "name": "s5_vs_rule_based_loot",
        "default_rounds": 500,
        "eps_start": 0.05,
        "scenario": "loot-crate",
        "opponents": ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
        "lr": 5e-5,
        "freeze_dqn": False,
    },
    4: {
        "name": "s6_vs_mixed_loot",
        "default_rounds": 500,
        "eps_start": 0.05,
        "scenario": "loot-crate",
        "opponents": ["my_spatial_dqn_agent", "my_spatial_qwm_agent", "rule_based_agent"],
        "lr": 1e-5,
        "freeze_dqn": False,
    },
    7: {
        "name": "s7_vs_mixed_classic",
        "default_rounds": 500,
        "eps_start": 0.05,
        "scenario": "classic",
        "opponents": ["my_spatial_dqn_agent", "my_spatial_qwm_agent", "rule_based_agent"],
        "lr": 1e-5,
        "freeze_dqn": False,
    },
}


def compute_stage_slices(n_rounds: int, interval: int) -> List[int]:
    """Slices a stage into intervals for intra-stage self-play."""
    if interval <= 0 or n_rounds <= interval:
        return [n_rounds]
    slices = []
    rem = n_rounds
    while rem > interval:
        slices.append(interval)
        rem -= interval
    if rem > 0:
        slices.append(rem)
    return slices


def run_game_session(
    stage_name: str,
    n_rounds: int,
    scenario: str,
    agents: list,
    run_dir: Path,
    model_file: str,
    target_model: Path,
    batch: int = 64,
    no_augment: bool = False,
    lr: float = 1e-4,
    freeze_dqn: bool = False,
    gui: bool = False,
    python_exe: str = sys.executable,
    checkpoint_name: Optional[str] = None,
    save_last: bool = True,
    is_self_play: bool = False,
    eps_start: Optional[float] = None,
    eps_end: float = 0.05,
    eps_decay: Optional[float] = None,
):
    """Run one single-process game session."""
    if n_rounds <= 0:
        return

    ckpt_name = checkpoint_name or stage_name

    env = os.environ.copy()
    env["MY_WM_RUN_DIR"] = str(run_dir)
    env["MY_WM_MODEL_FILE"] = model_file
    env["MY_WM_STAGE"] = ckpt_name
    env["MY_WM_N_ROUNDS"] = str(n_rounds)
    env["MY_WM_FREEZE_DQN"] = "1" if freeze_dqn else "0"
    env["MY_WM_TREE_SEARCH"] = "0" if freeze_dqn else "1"
    env["MY_WM_BATCH"] = str(batch)
    env["MY_WM_AUGMENT"] = "0" if no_augment else "1"
    if lr is not None:
        env["MY_WM_LR"] = str(lr)

    if is_self_play:
        env["MY_WM_SELF_PLAY"] = "1"
        env["MY_WM_EPSILON"] = "0.0"
        env["MY_WM_EPS_START"] = "0.0"
        env["MY_WM_EPS_END"] = "0.0"
        env["MY_WM_EPS_DECAY"] = "1.0"
    else:
        env["MY_WM_SELF_PLAY"] = "0"
        if eps_start is not None:
            env["MY_WM_EPSILON"] = str(eps_start)
            env["MY_WM_EPS_START"] = str(eps_start)
        env["MY_WM_EPS_END"] = str(eps_end)
        if eps_decay is not None:
            env["MY_WM_EPS_DECAY"] = str(eps_decay)

    cmd = [
        python_exe,
        "main.py",
        "play",
        "--train", "1",
        "--scenario", scenario,
        "--n-rounds", str(n_rounds),
    ]
    if not gui:
        cmd.append("--no-gui")
    cmd.extend(["--agents"] + list(agents))

    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    print(f"[curriculum] {stage_name} ({scenario}, {n_rounds} r) finished in {dt:.1f}s (exit {res.returncode})")
    if res.returncode != 0:
        print(f"[curriculum] ERROR in {stage_name}, stopping curriculum.")
        sys.exit(res.returncode)

    if save_last and target_model.is_file():
        shutil.copy2(target_model, run_dir / f"last_model_{ckpt_name}.pt")

    stats_csv = run_dir / "training_stats-spatial-qwm.csv"
    curves_png = run_dir / "training_curves-spatial-qwm.png"
    compare_png = run_dir / "future_value_comparison-spatial-qwm.png"
    eval_json = run_dir / "future_val_eval-spatial-qwm.json"
    try:
        from .plot_training import plot_stats_file, plot_future_value_comparison
    except ImportError:
        try:
            from plot_training import plot_stats_file, plot_future_value_comparison
        except ImportError:
            plot_stats_file = None
            plot_future_value_comparison = None

    if plot_stats_file and stats_csv.is_file():
        if plot_stats_file(str(stats_csv), str(curves_png)):
            shutil.copy2(curves_png, run_dir / "training_curves.png")
            shutil.copy2(curves_png, HERE / "training_curves-spatial-qwm.png")
            shutil.copy2(curves_png, HERE / "training_curves.png")

    if plot_future_value_comparison:
        if plot_future_value_comparison(str(eval_json) if eval_json.is_file() else None, str(compare_png)):
            shutil.copy2(compare_png, HERE / "future_value_comparison-spatial-qwm.png")


def run_self_play_stage(
    stage_name: str,
    total_rounds: int,
    agent_name: str,
    scenario_pool: list,
    run_dir: Path,
    model_file: str,
    target_model: Path,
    batch: int = 64,
    no_augment: bool = False,
    lr: float = 1e-4,
    chunk_size: int = 50,
    gui: bool = False,
    python_exe: str = sys.executable,
    checkpoint_name: Optional[str] = None,
    save_last: bool = True,
):
    """Executes single-process self-play across randomized scenarios."""
    if total_rounds <= 0:
        return

    remaining = total_rounds
    chunk_idx = 0
    while remaining > 0:
        chunk_idx += 1
        n = min(remaining, chunk_size)
        is_last_chunk = (remaining - n <= 0)
        chunk_save_last = (is_last_chunk and save_last)
        scenario = random.choice(scenario_pool)

        print(f"\n------------------------------------------------------------")
        print(f"--- [Self-Play] {stage_name} (Chunk {chunk_idx}: {n} rounds, scenario={scenario}) ---")
        print(f"--- 4 clones of '{agent_name}' competing with ZERO discovery ---")
        print(f"------------------------------------------------------------")

        agents = [agent_name] * 4
        run_game_session(
            stage_name=stage_name,
            n_rounds=n,
            scenario=scenario,
            agents=agents,
            run_dir=run_dir,
            model_file=model_file,
            target_model=target_model,
            batch=batch,
            no_augment=no_augment,
            lr=lr,
            gui=gui,
            python_exe=python_exe,
            checkpoint_name=checkpoint_name,
            save_last=chunk_save_last,
            is_self_play=True,
            eps_start=0.0,
            eps_end=0.0,
            eps_decay=1.0,
        )
        remaining -= n


def build_arg_parser():
    p = argparse.ArgumentParser(description="High-Performance Curriculum Training for my_spatial_qwm_agent")
    p.add_argument("--stages", type=int, default=5, help="Run stages up to N (0..4, default: 4)")
    p.add_argument("--start-stage", type=int, default=0, help="Start at this stage (0..4, default: 0)")
    p.add_argument("--s0", type=int, default=None, help="Rounds for Stage 0 (World Model Warmup)")
    p.add_argument("--s1", type=int, default=None, help="Rounds for Stage 1 (Coin Heaven)")
    p.add_argument("--s2", type=int, default=None, help="Rounds for Stage 2 (Classic Alone)")
    p.add_argument("--s3", type=int, default=None, help="Rounds for Stage 3 (vs Passive)")
    p.add_argument("--s4", type=int, default=None, help="Rounds for Stage 4 (vs Rule-Based)")
    p.add_argument("--scale", type=float, default=1.0, help="Multiply all default round counts")
    p.add_argument("--model-file", type=str, default="my-saved-model-spatial-qwm.pt", help="Model checkpoint file name")
    p.add_argument("--run-dir", type=str, default=None, help="Explicit run directory")
    p.add_argument("--fresh", action="store_true", help="Start training fresh (do not bootstrap from base checkpoint)")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--batch", type=int, default=128, help="Batch size (default: 128, optimal for RTX 3080)")
    p.add_argument("--no-augment", action="store_true", help="Disable D4 dihedral symmetry augmentation")
    p.add_argument("--gui", action="store_true", default=False, help="Render GUI during training")
    p.add_argument("--workers", "--parallel", dest="workers", type=int, default=8, help="Number of parallel game instances / worker processes (default: 8, optimal for Ryzen 7 5700X)")
    p.add_argument("--accumulate", type=int, default=None, help="Gradient accumulation count")
    p.add_argument("--search-depth", "--depth", dest="search_depth", type=int, default=None, help="Lookahead reasoning depth for tree search (default: from config.json)")

    # Self-play options
    p.add_argument("--no-self-play", action="store_true", help="Disable all self-play stages")
    p.add_argument("--sp-rounds", type=int, default=50, help="Default post-stage self-play rounds (default: 100)")
    p.add_argument("--s0-sp", type=int, default=0, help="Post-stage self-play rounds after Stage 0 (default: 0)")
    p.add_argument("--s1-sp", type=int, default=None, help="Post-stage self-play rounds after Stage 1")
    p.add_argument("--s2-sp", type=int, default=None, help="Post-stage self-play rounds after Stage 2")
    p.add_argument("--s3-sp", type=int, default=None, help="Post-stage self-play rounds after Stage 3")
    p.add_argument("--s4-sp", type=int, default=None, help="Post-stage self-play rounds after Stage 4")
    p.add_argument("--intra-sp-interval", type=int, default=600, help="Rounds interval within a stage to trigger intra self-play (0 to disable, default: 600)")
    p.add_argument("--intra-sp-rounds", type=int, default=50, help="Rounds for each intra-stage self-play block (default: 100)")
    p.add_argument("--sp-chunk", type=int, default=50, help="Max chunk size in rounds per random scenario during self-play (default: 100)")
    p.add_argument("--sp-scenarios", nargs="+", default=DEFAULT_SELF_PLAY_SCENARIOS, help="Scenarios pool for self-play")
    return p


def main():
    args = build_arg_parser().parse_args()
    os.chdir(ROOT)

    if args.search_depth is not None:
        os.environ["MY_WM_SEARCH_DEPTH"] = str(args.search_depth)
        print(f"[curriculum] Reasoning search depth set to: {args.search_depth}")

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = HERE / "runs" / f"run_{stamp}_spatial_qwm"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[curriculum] Run directory: {run_dir}")

    target_model = run_dir / args.model_file
    root_model = HERE / args.model_file

    # Bootstrap weights if target model does not exist
    if not target_model.is_file():
        if not args.fresh and root_model.is_file():
            shutil.copy2(root_model, target_model)
            print(f"[curriculum] Inherited existing model weights from {root_model}")
        else:
            cfg = {}
            cfg_p = HERE / "config.json"
            if cfg_p.is_file():
                with open(cfg_p) as f:
                    cfg = json.load(f)
            init_ckpt = cfg.get("initial_checkpoint")
            if init_ckpt:
                resolved = (HERE / init_ckpt).resolve() if not os.path.isabs(init_ckpt) else Path(init_ckpt)
                if resolved.is_file():
                    from agent_code.my_spatial_qwm_agent.model import new_model
                    m = new_model()
                    m.load(str(resolved), partial=True)
                    m.save(str(target_model))
                    print(f"[curriculum] Bootstrapped starting weights from: {resolved.name}")

    shared_model = None
    if args.workers > 1:
        from parallel_train import load_or_create_shared_model, run_parallel_stage, run_parallel_self_play_stage
        shared_model = load_or_create_shared_model("my_spatial_qwm_agent", target_model, lr=args.lr or 1e-4)

    default_post_sp = int(round(args.sp_rounds * args.scale))
    intra_rounds = int(round(args.intra_sp_rounds * args.scale))
    sp_scenarios = list(args.sp_scenarios)

    stages_to_run = [s for s in range(args.start_stage, args.stages + 1) if s in STAGE_CONFIG]

    for st in stages_to_run:
        cfg = STAGE_CONFIG[st]
        override_rounds = getattr(args, f"s{st}", None)
        n_rounds = override_rounds if override_rounds is not None else int(round(cfg["default_rounds"] * args.scale))
        if n_rounds <= 0:
            print(f"[curriculum] Skipping stage {st} (0 rounds)")
            continue

        stage_lr = args.lr if args.lr else cfg["lr"]
        stage_ckpt_name = cfg["name"]
        freeze_dqn = cfg.get("freeze_dqn", False)

        # Self-play config
        override_post_sp = getattr(args, f"s{st}_sp", None)
        post_sp_rounds = override_post_sp if override_post_sp is not None else (0 if st == 0 else default_post_sp)
        has_post_sp = (not args.no_self_play and post_sp_rounds > 0)
        stage_sp_name = f"s{st}_self_play"

        stage_eps_start = cfg.get("eps_start", 1.0)
        eps_end = 0.001
        worker_stage_rounds = max(1, n_rounds // args.workers) if args.workers > 1 else n_rounds
        decay_rounds = max(1, int(round(worker_stage_rounds * 0.8)))
        stage_decay = (eps_end / stage_eps_start) ** (1.0 / decay_rounds) if stage_eps_start > eps_end else 1.0
        current_eps = stage_eps_start

        # 1. Normal stage execution (with intra-stage self-play slicing if applicable)
        slices = compute_stage_slices(n_rounds, args.intra_sp_interval) if not args.no_self_play else [n_rounds]
        for slice_idx, slice_rounds in enumerate(slices, start=1):
            is_last_slice = (slice_idx == len(slices))
            save_last_slice = is_last_slice

            slice_desc = f"Part {slice_idx}/{len(slices)}: {slice_rounds} rounds" if len(slices) > 1 else f"{slice_rounds} rounds"
            print(f"\n============================================================")
            print(f"=== Stage {st}: {cfg['name']} ({slice_desc}, scenario={cfg['scenario']}, eps={current_eps:.4f}->decay={stage_decay:.6f}) ===")
            print(f"=== DQN Frozen: {freeze_dqn} | Parallel Workers: {args.workers} ===")
            print(f"============================================================")

            if args.workers > 1:
                from parallel_train import run_parallel_stage
                run_parallel_stage(
                    stage_name=cfg["name"],
                    total_rounds=slice_rounds,
                    num_workers=args.workers,
                    agent_name="my_spatial_qwm_agent",
                    scenario=cfg["scenario"],
                    opponents=list(cfg["opponents"]),
                    shared_model=shared_model,
                    run_dir=run_dir,
                    model_file=args.model_file,
                    target_model=target_model,
                    batch=args.batch,
                    no_augment=args.no_augment,
                    lr=stage_lr,
                    accumulate_count=args.accumulate,
                    checkpoint_name=stage_ckpt_name,
                    save_best=True,
                    save_last=True,
                    is_self_play=False,
                    freeze_dqn=freeze_dqn,
                    tree_search=not freeze_dqn,
                    eps_start=current_eps,
                    eps_end=eps_end,
                    eps_decay=stage_decay,
                )
            else:
                run_game_session(
                    stage_name=cfg["name"],
                    n_rounds=slice_rounds,
                    scenario=cfg["scenario"],
                    agents=["my_spatial_qwm_agent"] + list(cfg["opponents"]),
                    run_dir=run_dir,
                    model_file=args.model_file,
                    target_model=target_model,
                    batch=args.batch,
                    no_augment=args.no_augment,
                    lr=stage_lr,
                    freeze_dqn=freeze_dqn,
                    gui=args.gui,
                    checkpoint_name=stage_ckpt_name,
                    save_best=True,
                    save_last=True,
                    is_self_play=False,
                    eps_start=current_eps,
                    eps_end=eps_end,
                    eps_decay=stage_decay,
                )

            # Advance stage epsilon progression across this normal slice (preserved across self-play)
            w_rounds_slice = max(1, slice_rounds // args.workers) if args.workers > 1 else slice_rounds
            current_eps = max(eps_end, current_eps * (stage_decay ** w_rounds_slice))

            # Intra-stage self-play between slices
            if not args.no_self_play and slice_idx < len(slices) and intra_rounds > 0 and not freeze_dqn:
                intra_name = f"s{st}_intra_sp_{slice_idx}"
                is_last_intra = (slice_idx == len(slices) - 1)
                save_last_intra = (is_last_intra and not has_post_sp)
                print(f"\n============================================================")
                print(f"=== Intra-Stage Self-Play: {intra_name} ({intra_rounds} rounds) ===")
                print(f"============================================================")
                if args.workers > 1:
                    from parallel_train import run_parallel_self_play_stage
                    run_parallel_self_play_stage(
                        stage_name=intra_name,
                        total_rounds=intra_rounds,
                        num_workers=args.workers,
                        agent_name="my_spatial_qwm_agent",
                        shared_model=shared_model,
                        run_dir=run_dir,
                        model_file=args.model_file,
                        target_model=target_model,
                        scenario_pool=sp_scenarios,
                        batch=args.batch,
                        no_augment=args.no_augment,
                        lr=stage_lr,
                        chunk_size=args.sp_chunk,
                        checkpoint_name=stage_sp_name,
                        save_best=True,
                        save_last=save_last_intra,
                    )
                else:
                    run_self_play_stage(
                        stage_name=intra_name,
                        total_rounds=intra_rounds,
                        agent_name="my_spatial_qwm_agent",
                        scenario_pool=sp_scenarios,
                        run_dir=run_dir,
                        model_file=args.model_file,
                        target_model=target_model,
                        batch=args.batch,
                        no_augment=args.no_augment,
                        lr=stage_lr,
                        chunk_size=args.sp_chunk,
                        gui=args.gui,
                        checkpoint_name=stage_sp_name,
                        save_best=True,
                        save_last=save_last_intra,
                    )

        # 2. Post-stage self-play after every normal stage completes (shares stage_sp_name)
        if has_post_sp:
            post_sp_name = f"s{st}_self_play"
            print(f"\n============================================================")
            print(f"=== Post-Stage {st} Self-Play: {post_sp_name} ({post_sp_rounds} rounds) ===")
            print(f"============================================================")
            if args.workers > 1:
                from parallel_train import run_parallel_self_play_stage
                run_parallel_self_play_stage(
                    stage_name=post_sp_name,
                    total_rounds=post_sp_rounds,
                    num_workers=args.workers,
                    agent_name="my_spatial_qwm_agent",
                    shared_model=shared_model,
                    run_dir=run_dir,
                    model_file=args.model_file,
                    target_model=target_model,
                    scenario_pool=sp_scenarios,
                    batch=args.batch,
                    no_augment=args.no_augment,
                    lr=stage_lr,
                    chunk_size=args.sp_chunk,
                    checkpoint_name=stage_sp_name,
                    save_best=True,
                    save_last=True,
                )
            else:
                run_self_play_stage(
                    stage_name=post_sp_name,
                    total_rounds=post_sp_rounds,
                    agent_name="my_spatial_qwm_agent",
                    scenario_pool=sp_scenarios,
                    run_dir=run_dir,
                    model_file=args.model_file,
                    target_model=target_model,
                    batch=args.batch,
                    no_augment=args.no_augment,
                    lr=stage_lr,
                    chunk_size=args.sp_chunk,
                    gui=args.gui,
                    checkpoint_name=stage_sp_name,
                    save_best=True,
                    save_last=True,
                )

    # Copy best or final model to agent root
    best_s4 = run_dir / "best_model_s4_vs_rule_based.pt"
    best_overall = best_s4 if best_s4.is_file() else target_model
    if best_overall.is_file():
        shutil.copy2(best_overall, root_model)
        print(f"\n[Done] Training curriculum complete! Best model saved to {root_model}")


if __name__ == "__main__":
    main()
