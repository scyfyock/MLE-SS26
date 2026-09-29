"""Extract and save only neural network weights from a world-model checkpoint.

Removes large replay buffers (which dominate disk space) while keeping all
trained MLP weights and ensuring is_ready / prediction properties still pass.
"""

import argparse
import os
from pathlib import Path
import pickle
import sys

# Ensure repository root is on sys.path so pickled agent classes can be unpickled
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agent_code.world_model_agent as _wma
sys.modules.setdefault("agent_code.world_model_agent_cf", _wma)
import agent_code.world_model_agent.world_model as _wm_mod
sys.modules.setdefault("agent_code.world_model_agent_cf.world_model", _wm_mod)
import agent_code.world_model_agent.q_network as _qn_mod
sys.modules.setdefault("agent_code.world_model_agent_cf.q_network", _qn_mod)


def strip_checkpoint(input_path: Path, output_path: Path):
    if not input_path.is_file():
        print(f"Error: input file '{input_path}' not found.")
        sys.exit(1)

    initial_bytes = input_path.stat().st_size
    print(f"Loading '{input_path}' ({initial_bytes / (1024 * 1024):.2f} MB)...")

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print(f"Error: Expected dict checkpoint, got {type(data)}")
        sys.exit(1)

    wm = data.get("world_model")
    qn = data.get("q_network")

    # 1. Strip World Model replay buffers
    if wm is not None:
        raw_experiences = len(getattr(wm, "replay", []))
        wm.replay = []
        wm.min_samples = 0  # Allows is_ready to evaluate True without buffer

        wm.death_risk_replay = []
        wm.suicide_risk_replay = []
        wm.bomb_kill_replay = []
        wm.bomb_failure_replay = []
        wm.pending_risk_indices = []
        wm.pending_bomb_outcome = None

        if hasattr(wm, "category_indices"):
            wm.category_indices = {cat: set() for cat in wm.category_indices}

        # Keep at most 32 compact BombOutcome entries (~2 KB) to satisfy is_bomb_outcome_ready
        if hasattr(wm, "bomb_outcome_replay") and len(wm.bomb_outcome_replay) > 32:
            wm.bomb_outcome_replay = wm.bomb_outcome_replay[-32:]

        print(f"Cleared {raw_experiences} world-model experiences.")

    # 2. Strip Q-network replay buffers
    if qn is not None:
        raw_q = len(getattr(qn, "replay", []))
        qn.replay = []
        qn.positive_replay = []
        qn.negative_replay = []
        qn.pending = []
        print(f"Cleared {raw_q} Q-network transitions.")

    # 3. Save stripped checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    final_bytes = output_path.stat().st_size
    reduction = (1.0 - (final_bytes / max(1, initial_bytes))) * 100.0
    print(f"Saved stripped model to '{output_path}'.")
    print(
        f"Size: {initial_bytes / (1024 * 1024):.2f} MB -> "
        f"{final_bytes / (1024 * 1024):.2f} MB ({reduction:.1f}% reduction)"
    )

    # 4. Verify integrity
    with open(output_path, "rb") as f:
        verified = pickle.load(f)

    v_wm = verified.get("world_model")
    if v_wm is not None:
        print(f"Verification: world_model.is_ready = {v_wm.is_ready}")
        if hasattr(v_wm, "is_bomb_outcome_ready"):
            print(f"Verification: world_model.is_bomb_outcome_ready = {v_wm.is_bomb_outcome_ready}")


def main():
    parser = argparse.ArgumentParser(
        description="Strip replay buffers from a world-model checkpoint."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("agent_code/world_model_agent/world-model.pt"),
        help="Path to the original large .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("agent_code/world_model_agent/world-model-weights-only.pt"),
        help="Path where the stripped lightweight .pt file will be saved",
    )
    args = parser.parse_args()
    strip_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
