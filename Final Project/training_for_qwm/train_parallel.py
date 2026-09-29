#!/usr/bin/env python3
"""
train_parallel.py -- Root CLI entry point for parallel training across multiple CPU workers.

Usage:
    # Train Full-Board Spatial Dueling DQN across 4 parallel game instances:
    python train_parallel.py --agent spatial_dqn --workers 4 --stages 1 2

    # Train World-Model DQN across 4 parallel game instances:
    python train_parallel.py --agent wm --workers 4 --stages 1 2

    # Quick test run on 2 workers for 10 rounds:
    python train_parallel.py --agent spatial_dqn --workers 2 --s1 10 --stages 1 --no-self-play
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Launch parallel multi-worker Bomberman RL training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=["spatial_dqn", "wm", "my_spatial_dqn_agent", "my_wm_agent"],
        default="spatial_dqn",
        help="Which agent to train in parallel.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of concurrent game instances / worker processes (default: 8, matching Ryzen 7 5700X).",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=128,
        help="Batch size for GPU Double-DQN training (default: 128).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=[1, 2, 3, 4],
        help="Stages to run (e.g. --stages 1 2).",
    )
    parser.add_argument("--s1", type=int, default=None, help="Rounds for stage 1")
    parser.add_argument("--s2", type=int, default=None, help="Rounds for stage 2")
    parser.add_argument("--s3", type=int, default=None, help="Rounds for stage 3")
    parser.add_argument("--s4", type=int, default=None, help="Rounds for stage 4")
    parser.add_argument("--scale", type=float, default=1.0, help="Multiply default round counts.")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh.")
    parser.add_argument("--no-self-play", action="store_true", help="Disable self-play stages.")
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit run directory.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")

    args, unknown = parser.parse_known_args()

    agent_key = args.agent
    if agent_key in ("spatial_dqn", "my_spatial_dqn_agent"):
        script = ROOT / "agent_code" / "my_spatial_dqn_agent" / "train_curriculum.py"
        cmd = [sys.executable, str(script), "--workers", str(args.workers)]
        if len(args.stages) == 1 and str(args.stages[0]).isdigit():
            cmd.extend(["--stages", str(args.stages[0])])
    else:
        script = ROOT / "agent_code" / "my_wm_agent" / "train_curriculum.py"
        cmd = [sys.executable, str(script), "--model", "dqn", "--workers", str(args.workers)]
        cmd.extend(["--stages"] + [str(s) for s in args.stages])

    if args.s1 is not None: cmd.extend(["--s1", str(args.s1)])
    if args.s2 is not None: cmd.extend(["--s2", str(args.s2)])
    if args.s3 is not None: cmd.extend(["--s3", str(args.s3)])
    if args.s4 is not None: cmd.extend(["--s4", str(args.s4)])
    if args.scale != 1.0: cmd.extend(["--scale", str(args.scale)])
    if args.batch is not None: cmd.extend(["--batch", str(args.batch)])
    if args.fresh: cmd.append("--fresh")
    if args.no_self_play: cmd.append("--no-self-play")
    if args.run_dir: cmd.extend(["--run-dir", args.run_dir])
    if args.lr is not None: cmd.extend(["--lr", str(args.lr)])
    if unknown: cmd.extend(unknown)

    print(f"[train_parallel] Launching command: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=ROOT)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
