#!/usr/bin/env python3
"""
restore_optuna_study.py - Reconstructs Optuna SQLite study from a tuning evaluation JSON artifact.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sys

import optuna
from optuna.trial import TrialState

FILE_PATH = Path(__file__).resolve()
AGENT_DIR = FILE_PATH.parent
AGENT_CODE_DIR = AGENT_DIR.parent
ROOT_DIR = AGENT_CODE_DIR.parent


def restore_study_from_json(json_path: Path, db_path: Path, study_name: str = "qwm_agent_inference_tuning"):
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    cands = data.get("eval_res", {}).get("ranked_candidates", [])
    if not cands:
        raise ValueError(f"No ranked_candidates found in {json_path}")

    # Map candidates by trial number
    by_trial = {}
    for c in cands:
        name = c.get("candidate", {}).get("name", "")
        m = re.search(r"Optuna-(\d+)", name)
        if m:
            trial_num = int(m.group(1))
            by_trial[trial_num] = c
        else:
            print(f"[Warning] Could not extract trial number from '{name}', skipping.")

    if not by_trial:
        raise ValueError("Could not extract any valid trials from candidate names.")

    n_trials = len(by_trial)
    min_trial = min(by_trial.keys())
    max_trial = max(by_trial.keys())
    print(f"[Restore] Loaded {n_trials} trials from JSON (Indices: {min_trial}..{max_trial})")

    # Backup existing DB if present
    if db_path.is_file():
        bak_path = db_path.with_suffix(f".db.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(db_path, bak_path)
        print(f"[Restore] Existing DB backed up to: {bak_path}")
        db_path.unlink()

    storage_url = f"sqlite:///{db_path.resolve()}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize"
    )

    # Insert trials sequentially to preserve exact trial numbers
    for t_idx in range(min_trial, max_trial + 1):
        if t_idx not in by_trial:
            continue
        c = by_trial[t_idx]
        p = c["candidate"]
        trial = optuna.trial.create_trial(
            params={
                "search_depth": int(p["search_depth"]),
                "beam_size": int(p["beam_size"]),
                "tree_discount": float(p["tree_discount"]),
                "alpha_vq": float(p["alpha_vq"]),
            },
            distributions={
                "search_depth": optuna.distributions.IntDistribution(3, 12),
                "beam_size": optuna.distributions.IntDistribution(6, 48, step=3),
                "tree_discount": optuna.distributions.FloatDistribution(0.00, 0.40, step=0.01),
                "alpha_vq": optuna.distributions.FloatDistribution(0.00, 0.96, step=0.02),
            },
            value=float(c.get("overall_score", 0.0)),
            user_attrs={
                "candidate_name": p.get("name", f"Optuna-{t_idx:03d}"),
                "overall_score": float(c.get("overall_score", 0.0)),
                "overall_win_rate": float(c.get("overall_win_rate", 0.0)),
                "overall_survival_rate": float(c.get("overall_survival_rate", 0.0)),
                "overall_mean_score": float(c.get("overall_mean_score", 0.0)),
                "overall_coins": float(c.get("overall_coins", 0.0)),
                "overall_kills": float(c.get("overall_kills", 0.0)),
                "overall_think_time_ms": float(c.get("overall_think_time_ms", 0.0)),
            },
            state=TrialState.COMPLETE,
        )
        study.add_trial(trial)

    print(f"\n[Restore] Successfully restored {len(study.trials)} trials into: {db_path.resolve()}")
    print(f"  • Study Name:  {study_name}")
    print(f"  • Best Trial:  #{study.best_trial.number} ({study.best_trial.user_attrs.get('candidate_name', '')})")
    print(f"  • Best Value:  {study.best_value:.2f}")
    print(f"  • Best Params: {study.best_params}\n")

    return study


def main():
    parser = argparse.ArgumentParser(description="Restore Optuna study from JSON.")
    parser.add_argument(
        "--json-path", type=str,
        default=str(ROOT_DIR / "eval_results" / "tune_qwm_agent_results_2026-09-13_11-30-46.json"),
        help="Path to tuning JSON file."
    )
    parser.add_argument(
        "--db-path", type=str,
        default=str(AGENT_DIR / "optuna_study.db"),
        help="Target SQLite DB path."
    )
    parser.add_argument(
        "--study-name", type=str,
        default="qwm_agent_inference_tuning",
        help="Study name."
    )
    args = parser.parse_args()

    restore_study_from_json(
        json_path=Path(args.json_path),
        db_path=Path(args.db_path),
        study_name=args.study_name
    )


if __name__ == "__main__":
    main()

