#!/usr/bin/env python3
"""
Convenience entry point for running evaluate.py from inside agent_code/my_spatial_dqn_agent.
Delegates to the root evaluate.py script with default agent set to 'my_spatial_dqn_agent'.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluate import main

if __name__ == "__main__":
    # If --agent was not explicitly provided, default to my_spatial_dqn_agent
    if "--agent" not in sys.argv and "-a" not in sys.argv:
        sys.argv.extend(["--agent", "my_spatial_dqn_agent"])
    main()

