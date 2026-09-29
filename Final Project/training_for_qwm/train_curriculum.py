#!/usr/bin/env python3
"""Convenience entry point for curriculum training of my_wm_agent.

Usage:
    python train_curriculum.py
    python train_curriculum.py --stages 4 --s4 3000
    python train_curriculum.py --model dqn --fresh --scale 2
"""

import argparse
import sys
from pathlib import Path

# Ensure root directory is on python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description="Curriculum training entry point", add_help=False)
    parser.add_argument("--agent", type=str, default="my_spatial_qwm_agent",
                        choices=["my_spatial_qwm_agent", "my_wm_agent"],
                        help="Which agent to train curriculum for (default: my_spatial_qwm_agent)")
    args, unknown = parser.parse_known_args()

    if args.agent == "my_spatial_qwm_agent":
        from agent_code.my_spatial_qwm_agent.train_curriculum import main as agent_main
    else:
        from agent_code.my_wm_agent.train_curriculum import main as agent_main

    # Forward remaining args
    sys.argv = [sys.argv[0]] + unknown
    agent_main()

if __name__ == "__main__":
    main()

