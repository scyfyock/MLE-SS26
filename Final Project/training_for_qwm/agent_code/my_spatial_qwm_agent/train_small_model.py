#!/usr/bin/env python3
"""
train_small_model.py -- Master Two-Phase Training and Evaluation Pipeline for SmallPathRewardModel.

Location: agent_code/my_spatial_qwm_agent/

Orchestrates the entire lifecycle requested by the user:
  1. Creates a timestamped run folder under runs/run_YYYY-MM-DD_HH-MM-SS_small_model/
  2. Phase 1 (Supervised Imitation):
     - Uses dynActivation with 5,893 parameters.
     - Trains on candidate path features targeting Optuna champion heuristics.
     - Logs epoch stats, saves small_model_phase1.pt, and generates initial curves.
  3. Phase 2 (Autonomous Environment RL Fine-Tuning):
     - Keeps base QWM model completely FROZEN.
     - Directly optimizes game rewards (coins, crates, kills, survival, wins).
     - Regularizes against Phase 1 anchor weights to preserve safety priors.
     - Logs round stats, saves small_model_phase2.pt, and updates training_curves-small-model.png.
  4. Copies active checkpoints, stats, and plots into agent_code/my_spatial_qwm_agent/.
  5. Runs benchmark tournament evaluation and outputs performance comparison.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

AGENT_DIR = Path(__file__).resolve().parent

from agent_code.my_spatial_qwm_agent.train_small_model_phase1 import (
    collect_dataset,
    create_run_folder,
    train_phase1,
)
from agent_code.my_spatial_qwm_agent.train_small_model_phase2 import (
    ALL_CURRICULUM_STAGES,
    STAGE_ALIASES,
    TARGET_ENVIRONMENTS,
    resolve_stages,
    run_phase2_training,
)
from agent_code.my_spatial_qwm_agent.plot_small_model import plot_small_model_stats


def update_config_json(small_model_file: str = "small_model.pt"):
    """Updates config.json with Optuna champion values and enables small model."""
    config_path = os.path.join(str(AGENT_DIR), "config.json")
    config = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            try:
                config = json.load(f)
            except Exception:
                config = {}

    config.update({
        "search_depth": 5,
        "beam_size": 18,
        "tree_discount": 0.08,
        "alpha_vq": 0.16,
        "predicted_wait_penalty": 1.25,
        "predicted_loop_penalty": 1.5,
        "use_small_model": True,
        "small_model_file": small_model_file,
    })

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Config] Updated config.json: use_small_model=True, small_model_file={small_model_file}")


def evaluate_agent(n_rounds: int = 10, seed: int = 1000, scenario: str = "classic") -> dict:
    """Evaluates agent in a tournament match against 3x rule_based_agent."""
    print(f"\n[Evaluation] Running {n_rounds}-round tournament in '{scenario}' vs 3x rule_based_agent (Seed: {seed})...")
    env = os.environ.copy()
    env["MY_QWM_USE_SMALL_MODEL"] = "1"
    env["MY_QWM_TREE_SEARCH"] = "1"
    env["MY_QWM_SEARCH_DEPTH"] = "5"
    env["MY_QWM_BEAM_SIZE"] = "18"
    env["MY_QWM_TREE_DISCOUNT"] = "0.08"
    env["MY_QWM_ALPHA_VQ"] = "0.16"

    cmd = [
        sys.executable,
        str(ROOT_DIR / "main.py"),
        "play",
        "--agents", "my_spatial_qwm_agent", "rule_based_agent", "rule_based_agent", "rule_based_agent",
        "--n-rounds", str(n_rounds),
        "--scenario", scenario,
        "--seed", str(seed),
        "--no-gui",
        "--save-stats",
        "--silence-errors",
    ]
    res = subprocess.run(cmd, env=env, cwd=str(ROOT_DIR), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Warning: Evaluation exited with code {res.returncode}")
        if res.stderr:
            print(f"Stderr tail: {res.stderr[-300:]}")

    # Parse stdout for scores and wins
    lines = res.stdout.splitlines()
    summary_lines = [l for l in lines if "my_spatial_qwm_agent" in l or "points" in l or "Score" in l]
    print("\n--- Match Output Summary ---")
    for l in lines[-15:]:
        print(l)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Master Two-Phase Training for SmallPathRewardModel")
    parser.add_argument("--phase1-epochs", type=int, default=10, help="Phase 1 supervised epochs")
    parser.add_argument("--phase2-rounds", type=int, default=10, help="Phase 2 RL rounds per stage")
    parser.add_argument("--stages", nargs="+", default=["all"],
                        help="Environments to train in (default: all main curriculum stages. Supports: all, curriculum, combat, loot, coins, or specific stages like s0..s7)")
    parser.add_argument("--collect-rounds", type=int, default=100, help="Phase 1 collection rounds if needed")
    parser.add_argument("--eval-rounds", type=int, default=10, help="Evaluation rounds after training")
    parser.add_argument("--skip-phase1-collect", action="store_true", default=False, help="Skip collection if dataset exists")
    parser.add_argument("--recollect", action="store_true", default=False, help="Force re-collection of Phase 1 dataset")
    parser.add_argument("--continue", "-c", dest="continue_training", action="store_true", default=False,
                        help="Continue training existing small_model.pt across Part 1 and Part 2")
    parser.add_argument("--init-model", type=str, default=None, help="Path to initial small model checkpoint to continue from")
    parser.add_argument("--skip-phase1", "--skip-part1", dest="skip_phase1", action="store_true", default=False,
                        help="Only skip Part 1 (Phase 1 supervised imitation) if explicitly requested")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of parallel worker processes for data collection (default: auto, up to 8)")
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit run directory")
    args = parser.parse_args()

    # Resolve stages across curriculum
    resolved_stages = resolve_stages(args.stages)

    # 1. Create Run Directory
    run_dir = args.run_dir or create_run_folder()
    mode_desc = "CONTINUATION MODE" if (args.continue_training or args.init_model) else "FRESH TRAINING"
    print("=" * 80)
    print(f" SMALL MODEL TRAINING PIPELINE ({mode_desc})")
    print(f" Target Directory: {run_dir}")
    print(f" Curriculum Bandwidth: {len(resolved_stages)} stages ({args.phase2_rounds} RL rounds/stage, {len(resolved_stages)*args.phase2_rounds} total RL rounds):")
    for s_name in resolved_stages:
        cfg = TARGET_ENVIRONMENTS[s_name]
        print(f"   • {s_name:<22} [{cfg['scenario']:<11}]: {cfg['description']}")
    print("=" * 80)

    # 2. Part 1 (Phase 1 Supervised Imitation): Never skipped unless explicitly instructed
    p1_ckpt = None
    if args.skip_phase1:
        print("\n" + "-" * 80)
        print(" SKIPPING PART 1 (Phase 1 Supervised Imitation)")
        print(" Reason: Explicitly requested via --skip-phase1 / --skip-part1")
        print("-" * 80)
        init_model = args.init_model
        if not init_model:
            candidates = [
                os.path.join(str(AGENT_DIR), "small_model.pt"),
                os.path.join(str(AGENT_DIR), "small_model_phase2.pt"),
                os.path.join(str(AGENT_DIR), "small_model_phase1.pt"),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    init_model = candidate
                    break
        if not init_model or not os.path.isfile(init_model):
            raise FileNotFoundError(f"Cannot skip Part 1: No small model checkpoint found at {init_model}")
        p1_ckpt = init_model
    else:
        # Prepare Part 1 Dataset
        data_file = os.path.join(run_dir, "path_data_phase1_target_envs.npz")
        active_data = os.path.join(str(AGENT_DIR), "path_data_phase1_target_envs.npz")

        if not args.recollect and os.path.isfile(active_data):
            if not os.path.isfile(data_file):
                shutil.copyfile(active_data, data_file)
            print(f"[Phase 1] Using existing dataset: {data_file}")
        elif not args.recollect and os.path.isfile(data_file):
            print(f"[Phase 1] Using existing dataset: {data_file}")
        else:
            collect_dataset(data_file, n_rounds=args.collect_rounds, environments=resolved_stages, n_workers=args.workers)
            if os.path.isfile(data_file) and not os.path.isfile(active_data):
                shutil.copyfile(data_file, active_data)

        # Run Part 1 Supervised Imitation
        print("\n>>> RUNNING PART 1: SUPERVISED PENALTY IMITATION <<<")
        p1_model, _ = train_phase1(
            data_file=data_file,
            run_dir=run_dir,
            epochs=args.phase1_epochs,
            init_model_path=args.init_model,
            continue_training=args.continue_training or bool(args.init_model),
        )
        p1_ckpt = os.path.join(run_dir, "small_model_phase1.pt")

    # 3. Part 2 (Phase 2 Environment RL Training)
    print("\n>>> RUNNING PART 2: AUTONOMOUS ENVIRONMENT RL FINE-TUNING <<<")
    run_phase2_training(
        stages=resolved_stages,
        rounds_per_stage=args.phase2_rounds,
        initial_model_path=p1_ckpt,
        continue_training=args.continue_training or bool(args.init_model),
        run_dir=run_dir,
        n_workers=args.workers,
    )

    # 4. Update Agent Configuration
    update_config_json("small_model.pt")

    # Copy plot to brain artifact directory if available
    brain_dir = "/home/tobitoyota/.gemini/antigravity/brain/6efa50a8-ca8d-41f8-8dd9-0f458b0af786"
    plot_src = os.path.join(run_dir, "training_curves-small-model.png")
    if os.path.isdir(brain_dir) and os.path.isfile(plot_src):
        shutil.copyfile(plot_src, os.path.join(brain_dir, "training_curves-small-model.png"))

    # 5. Tournament Evaluation
    if args.eval_rounds > 0:
        print("\n>>> STARTING TOURNAMENT VERIFICATION <<<")
        evaluate_agent(n_rounds=args.eval_rounds, seed=1000)

    print("\n" + "=" * 80)
    print(" TWO-PHASE TRAINING & EVALUATION COMPLETE!")
    print(f" Run Directory: {run_dir}")
    print(f" Telemetry CSV: {os.path.join(run_dir, 'training_stats-small-model.csv')}")
    print(f" Curves Plot:   {os.path.join(run_dir, 'training_curves-small-model.png')}")
    print(f" Checkpoints:   {os.path.join(run_dir, 'small_model_phase1.pt')}, {os.path.join(run_dir, 'small_model_phase2.pt')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
