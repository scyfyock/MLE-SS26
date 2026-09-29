#!/usr/bin/env python3
"""
Convenience entry point for running tune_inference.py across agents.
Delegates to agent_code/qwm_agent/tune_inference.py or
agent_code/my_spatial_qwm_agent/tune_inference.py based on --agent flag.
"""
import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main(argv: Optional[List[str]] = None):
    if argv is None:
        argv = sys.argv[1:]

    # Extract --agent if present
    agent = "my_spatial_qwm_agent"
    filtered_argv: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--agent":
            if i + 1 < len(argv):
                agent = argv[i + 1]
                i += 2
                continue
            else:
                agent = "my_spatial_qwm_agent"
                i += 1
                continue
        elif arg.startswith("--agent="):
            agent = arg.split("=", 1)[1]
            i += 1
            continue
        filtered_argv.append(arg)
        i += 1

    if agent in ("qwm", "qwm_agent"):
        agent_dir = ROOT_DIR / "agent_code" / "qwm_agent"
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        from agent_code.qwm_agent.tune_inference import main as qwm_main
        return qwm_main(filtered_argv)
    elif agent in ("my_spatial_qwm", "my_spatial_qwm_agent"):
        agent_dir = ROOT_DIR / "agent_code" / "my_spatial_qwm_agent"
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        from agent_code.my_spatial_qwm_agent.tune_inference import main as spatial_qwm_main
        return spatial_qwm_main(filtered_argv)
    else:
        raise ValueError(
            f"Unknown agent '{agent}'. Choose 'qwm_agent' or 'my_spatial_qwm_agent'."
        )


if __name__ == "__main__":
    main()

