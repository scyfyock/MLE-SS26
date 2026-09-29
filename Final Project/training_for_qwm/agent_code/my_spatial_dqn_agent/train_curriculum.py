#!/usr/bin/env python3
"""
train_curriculum.py -- Curriculum training runner for my_spatial_dqn_agent.

Includes dedicated self-play stages with randomized scenarios ('classic', 'loot-crate', 'coin-heaven')
running after each normal stage and periodically (intra-stage) during longer stages.

Usage:
    python agent_code/my_spatial_dqn_agent/train_curriculum.py
    python agent_code/my_spatial_dqn_agent/train_curriculum.py --stages 2
    python agent_code/my_spatial_dqn_agent/train_curriculum.py --s1 100 --s2 200
    python agent_code/my_spatial_dqn_agent/train_curriculum.py --no-self-play
"""

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SELF_PLAY_SCENARIOS = ["classic", "loot-crate", "coin-heaven"]

STAGE_CONFIG = {
    1: {
        "name": "s1_coin_heaven",
        "default_rounds": 300,
        "eps_start": 1.0,
        "scenario": "coin-heaven",
        "opponents": [],
        "lr": 1e-4,
    },
    2: {
        "name": "s2_classic_alone",
        "default_rounds": 600,
        "eps_start": 0.3,
        "scenario": "loot-crate",
        "opponents": [],
        "lr": 1e-4,
    },
    3: {
        "name": "s3_vs_passive",
        "default_rounds": 400,
        "eps_start": 0.2,
        "scenario": "loot-crate",
        "opponents": ["peaceful_agent", "coin_collector_agent"],
        "lr": 1e-4,
    },
    4: {
        "name": "s4_vs_rule_based",
        "default_rounds": 600,
        "eps_start": 0.15,
        "scenario": "classic",
        "opponents": ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
        "lr": 5e-5,
    },
}


def compute_stage_slices(n_rounds: int, interval: int):
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
    batch: int,
    no_augment: bool,
    lr: float,
    eps_rounds: int = None,
    gui: bool = False,
    python_exe: str = sys.executable,
    checkpoint_name: str = None,
    save_last: bool = True,
    is_self_play: bool = False,
    eps_start: float = None,
    eps_end: float = 0.05,
    eps_decay: float = None,
):
    """Run one game session (stage, intra self-play, or post-stage self-play)."""
    if n_rounds <= 0:
        return

    ckpt_name = checkpoint_name or stage_name

    env = os.environ.copy()
    env["MY_WM_RUN_DIR"] = str(run_dir)
    env["MY_WM_MODEL_FILE"] = model_file
    env["MY_WM_STAGE"] = ckpt_name
    env["MY_WM_N_ROUNDS"] = str(n_rounds)
    if is_self_play:
        env["MY_WM_SELF_PLAY"] = "1"
        env["MY_WM_EPSILON"] = "0.0"
        env["MY_WM_EPS_END"] = "0.0"
        env["MY_WM_EPS_ROUNDS"] = "0"
        env["MY_WM_EPS_DECAY"] = "1.0"
    else:
        env["MY_WM_SELF_PLAY"] = "0"
        if eps_start is not None:
            env["MY_WM_EPSILON"] = str(eps_start)
        if eps_end is not None:
            env["MY_WM_EPS_END"] = str(eps_end)
        if eps_decay is not None:
            env["MY_WM_EPS_DECAY"] = str(eps_decay)
        elif eps_rounds is not None:
            env["MY_WM_EPS_ROUNDS"] = str(eps_rounds)
        else:
            env["MY_WM_EPS_ROUNDS"] = str(max(1, int(n_rounds * 0.8)))
    env["MY_WM_BATCH"] = str(batch)
    env["MY_WM_AUGMENT"] = "0" if no_augment else "1"
    if lr is not None:
        env["MY_WM_LR"] = str(lr)

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

    # Archive last model for stage if requested and log best model
    if save_last and target_model.is_file():
        import shutil
        shutil.copy2(target_model, run_dir / f"last_model_{ckpt_name}.pt")

    best_json = run_dir / f"best_model_{ckpt_name}.json"
    if best_json.is_file():
        try:
            import json
            with open(best_json) as fh:
                binfo = json.load(fh)
            print(f"[curriculum] {ckpt_name} Best Model: Round {binfo.get('round')} | Score {binfo.get('score')} | Reward {binfo.get('sum_reward')} | Survived {binfo.get('survived')} | Steps {binfo.get('steps')}")
        except Exception:
            pass

    # Update training curves plot
    stats_csv = run_dir / "training_stats-spatial-dqn.csv"
    curves_png = run_dir / "training_curves-spatial-dqn.png"
    try:
        from .plot_training import plot_stats_file
    except ImportError:
        try:
            from plot_training import plot_stats_file
        except ImportError:
            plot_stats_file = None
    if plot_stats_file and stats_csv.is_file():
        if plot_stats_file(str(stats_csv), str(curves_png)):
            import shutil
            shutil.copy2(curves_png, run_dir / "training_curves.png")
            shutil.copy2(curves_png, HERE / "training_curves-spatial-dqn.png")
            shutil.copy2(curves_png, HERE / "training_curves.png")


def run_self_play_stage(
    stage_name: str,
    total_rounds: int,
    agent_name: str,
    scenario_pool: list,
    run_dir: Path,
    model_file: str,
    target_model: Path,
    batch: int,
    no_augment: bool,
    lr: float,
    chunk_size: int = 50,
    gui: bool = False,
    python_exe: str = sys.executable,
    checkpoint_name: str = None,
    save_last: bool = True,
):
    """
    Executes a self-play stage where 4 clones of the agent compete.
    Rounds are executed in chunks, with each chunk randomly choosing a scenario
    from scenario_pool (e.g. classic, loot-crate, coin-heaven).
    """
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
        print(f"--- Competing against 3 clones of itself ({agent_name}) ---")
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
    p = argparse.ArgumentParser(description="Curriculum training for my_spatial_dqn_agent")
    p.add_argument("--stages", type=int, default=4, help="Run stages 1..N (default: 4)")
    p.add_argument("--start-stage", type=int, default=1, help="Start at this stage (default: 1)")
    p.add_argument("--s1", type=int, default=None, help="Rounds for stage 1")
    p.add_argument("--s2", type=int, default=None, help="Rounds for stage 2")
    p.add_argument("--s3", type=int, default=None, help="Rounds for stage 3")
    p.add_argument("--s4", type=int, default=None, help="Rounds for stage 4")
    p.add_argument("--scale", type=float, default=1.0, help="Multiply all default round counts")
    p.add_argument("--model-file", type=str, default="my-saved-model-spatial-dqn.pt", help="Model file name")
    p.add_argument("--run-dir", type=str, default=None, help="Explicit run directory")
    p.add_argument("--fresh", action="store_true", help="Start training fresh (do not inherit older weights)")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--batch", type=int, default=128, help="Batch size (default: 128, optimal for RTX 3080)")
    p.add_argument("--no-augment", action="store_true", help="Disable D4 dihedral symmetry augmentation")
    p.add_argument("--gui", action="store_true", default=False, help="Render GUI during training")
    p.add_argument("--workers", "--parallel", dest="workers", type=int, default=12, help="Number of parallel game instances / worker processes (default: 8, optimal for Ryzen 7 5700X)")
    p.add_argument("--accumulate", type=int, default=None, help="Number of gradients to accumulate before optimizer step (default: equal to workers)")

    # Self-play options
    p.add_argument("--no-self-play", action="store_true", help="Disable all self-play stages")
    p.add_argument("--sp-rounds", type=int, default=100, help="Default post-stage self-play rounds (default: 100)")
    p.add_argument("--s1-sp", type=int, default=None, help="Post-stage self-play rounds after stage 1")
    p.add_argument("--s2-sp", type=int, default=None, help="Post-stage self-play rounds after stage 2")
    p.add_argument("--s3-sp", type=int, default=None, help="Post-stage self-play rounds after stage 3")
    p.add_argument("--s4-sp", type=int, default=None, help="Post-stage self-play rounds after stage 4")
    p.add_argument("--intra-sp-interval", type=int, default=600, help="Rounds interval within a stage to trigger intra self-play (0 to disable, default: 200)")
    p.add_argument("--intra-sp-rounds", type=int, default=100, help="Rounds for each intra-stage self-play block (default: 30)")
    p.add_argument("--sp-chunk", type=int, default=100, help="Max chunk size in rounds per random scenario during self-play (default: 50)")
    p.add_argument("--sp-scenarios", nargs="+", default=DEFAULT_SELF_PLAY_SCENARIOS, help="Scenarios pool for self-play")
    return p


def main():
    args = build_arg_parser().parse_args()
    os.chdir(ROOT)

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = HERE / "runs" / f"run_{stamp}_spatial_dqn"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[curriculum] Run directory: {run_dir}")

    target_model = run_dir / args.model_file
    root_model = HERE / args.model_file

    if not args.fresh and root_model.is_file() and not target_model.is_file():
        import shutil
        shutil.copy2(root_model, target_model)
        print(f"[curriculum] Inherited weights from {root_model}")

    shared_model = None
    if args.workers > 1:
        from parallel_train import load_or_create_shared_model, run_parallel_stage, run_parallel_self_play_stage
        shared_model = load_or_create_shared_model("my_spatial_dqn_agent", target_model, lr=args.lr or 1e-4)

    default_post_sp = int(round(args.sp_rounds * args.scale))
    intra_rounds = int(round(args.intra_sp_rounds * args.scale))
    sp_scenarios = list(args.sp_scenarios)

    for st in range(args.start_stage, args.stages + 1):
        cfg = STAGE_CONFIG[st]
        override_rounds = getattr(args, f"s{st}", None)
        n_rounds = override_rounds if override_rounds is not None else int(round(cfg["default_rounds"] * args.scale))
        if n_rounds <= 0:
            print(f"[curriculum] Skipping stage {st} (0 rounds)")
            continue

        stage_lr = args.lr if args.lr else cfg["lr"]
        stage_ckpt_name = cfg["name"]

        override_post_sp = getattr(args, f"s{st}_sp", None)
        post_sp_rounds = override_post_sp if override_post_sp is not None else default_post_sp
        has_post_sp = (not args.no_self_play and post_sp_rounds > 0)

        stage_sp_name = f"s{st}_self_play"

        # Compute consistent epsilon decay across this entire stage
        stage_eps_start = cfg.get("eps_start", 1.0)
        eps_end = 0.05
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
            print(f"============================================================")

            if args.workers > 1:
                run_parallel_stage(
                    stage_name=cfg["name"],
                    total_rounds=slice_rounds,
                    num_workers=args.workers,
                    agent_name="my_spatial_dqn_agent",
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
                    save_last=save_last_slice,
                    is_self_play=False,
                    eps_start=current_eps,
                    eps_end=eps_end,
                    eps_decay=stage_decay,
                )
            else:
                run_game_session(
                    stage_name=cfg["name"],
                    n_rounds=slice_rounds,
                    scenario=cfg["scenario"],
                    agents=["my_spatial_dqn_agent"] + list(cfg["opponents"]),
                    run_dir=run_dir,
                    model_file=args.model_file,
                    target_model=target_model,
                    batch=args.batch,
                    no_augment=args.no_augment,
                    lr=stage_lr,
                    gui=args.gui,
                    checkpoint_name=stage_ckpt_name,
                    save_last=save_last_slice,
                    is_self_play=False,
                    eps_start=current_eps,
                    eps_end=eps_end,
                    eps_decay=stage_decay,
                )

            # Advance stage epsilon progression across this normal slice (preserved across self-play)
            w_rounds_slice = max(1, slice_rounds // args.workers) if args.workers > 1 else slice_rounds
            current_eps = max(eps_end, current_eps * (stage_decay ** w_rounds_slice))

            # Intra-stage self-play between slices
            # Intra-stage self-play between slices (shares stage_sp_name with post-stage self-play)
            if not args.no_self_play and slice_idx < len(slices) and intra_rounds > 0:
                intra_name = f"s{st}_intra_sp_{slice_idx}"
                is_last_intra = (slice_idx == len(slices) - 1)
                save_last_intra = (is_last_intra and not has_post_sp)
                print(f"\n============================================================")
                print(f"=== Intra-Stage Self-Play: {intra_name} ({intra_rounds} rounds) ===")
                print(f"============================================================")
                if args.workers > 1:
                    run_parallel_self_play_stage(
                        stage_name=intra_name,
                        total_rounds=intra_rounds,
                        num_workers=args.workers,
                        agent_name="my_spatial_dqn_agent",
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
                        agent_name="my_spatial_dqn_agent",
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
                        save_last=save_last_intra,
                    )

        # 2. Post-stage self-play after every normal stage completes
        # 2. Post-stage self-play after every normal stage completes (shares stage_sp_name)
        if has_post_sp:
            post_sp_name = f"s{st}_self_play"
            print(f"\n============================================================")
            print(f"=== Post-Stage {st} Self-Play: {post_sp_name} ({post_sp_rounds} rounds) ===")
            print(f"============================================================")
            if args.workers > 1:
                run_parallel_self_play_stage(
                    stage_name=post_sp_name,
                    total_rounds=post_sp_rounds,
                    num_workers=args.workers,
                    agent_name="my_spatial_dqn_agent",
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
                    agent_name="my_spatial_dqn_agent",
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
                    save_last=True,
                )

    # After completion, update root model checkpoint & final plots
    if target_model.is_file():
        import shutil
        shutil.copy2(target_model, root_model)
        print(f"[curriculum] Synced trained model to {root_model}")

    stats_csv = run_dir / "training_stats-spatial-dqn.csv"
    curves_png = run_dir / "training_curves-spatial-dqn.png"
    if stats_csv.is_file():
        try:
            from .plot_training import plot_stats_file
        except ImportError:
            try:
                from plot_training import plot_stats_file
            except ImportError:
                plot_stats_file = None
        if plot_stats_file:
            if plot_stats_file(str(stats_csv), str(curves_png)):
                import shutil
                shutil.copy2(curves_png, run_dir / "training_curves.png")
                shutil.copy2(curves_png, HERE / "training_curves-spatial-dqn.png")
                shutil.copy2(curves_png, HERE / "training_curves.png")
                print(f"[curriculum] Saved final training curves to {curves_png} and {HERE / 'training_curves.png'}")


if __name__ == "__main__":
    main()
