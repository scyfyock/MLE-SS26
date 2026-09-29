#!/usr/bin/env python3
"""
Convenience entry point for running evaluate.py for my_spatial_qwm_agent.
Delegates to the root evaluate.py script with default agent set to 'my_spatial_qwm_agent'.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluate import main

if __name__ == "__main__":
    if "--agent" not in sys.argv and "-a" not in sys.argv:
        sys.argv.extend(["--agent", "my_spatial_qwm_agent" , "--model-file", "my-saved-model-spatial-qwm.pt", "--n-rounds", "10"])
    main()

