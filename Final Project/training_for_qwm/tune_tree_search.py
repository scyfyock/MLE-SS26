#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate_model

CHECKPOINT = ROOT / "agent_code/my_spatial_qwm_agent/my-saved-model-spatial-qwm.pt"

CANDIDATE_CONFIGS = [
    {"D": 4, "K": 2, "discount": 0.10, "alpha_vq": 0.50, "name": "D4_K2_disc0.10_alpha0.50_baseline"},
    {"D": 2, "K": 2, "discount": 0.10, "alpha_vq": 0.50, "name": "D2_K2_disc0.10_alpha0.50"},
    {"D": 3, "K": 2, "discount": 0.10, "alpha_vq": 0.50, "name": "D3_K2_disc0.10_alpha0.50"},
    {"D": 4, "K": 1, "discount": 0.10, "alpha_vq": 0.50, "name": "D4_K1_disc0.10_alpha0.50"},
    {"D": 4, "K": 3, "discount": 0.10, "alpha_vq": 0.50, "name": "D4_K3_disc0.10_alpha0.50"},
    {"D": 3, "K": 3, "discount": 0.10, "alpha_vq": 0.50, "name": "D3_K3_disc0.10_alpha0.50"},
    {"D": 4, "K": 2, "discount": 0.15, "alpha_vq": 0.50, "name": "D4_K2_disc0.15_alpha0.50"},
    {"D": 4, "K": 2, "discount": 0.20, "alpha_vq": 0.50, "name": "D4_K2_disc0.20_alpha0.50"},
    {"D": 4, "K": 2, "discount": 0.10, "alpha_vq": 0.40, "name": "D4_K2_disc0.10_alpha0.40"},
    {"D": 4, "K": 2, "discount": 0.10, "alpha_vq": 0.60, "name": "D4_K2_disc0.10_alpha0.60"},
    {"D": 3, "K": 2, "discount": 0.15, "alpha_vq": 0.50, "name": "D3_K2_disc0.15_alpha0.50"},
    {"D": 4, "K": 3, "discount": 0.15, "alpha_vq": 0.50, "name": "D4_K3_disc0.15_alpha0.50"},
]

def run_tuning(rounds_per_config=10, seed=42):
    results = []
    print(f"=== Starting QWM Tree Search Hyperparameter Tuning ===")
    print(f"Rounds per configuration: {rounds_per_config} (Seed: {seed})")
    print(f"Total configurations to evaluate: {len(CANDIDATE_CONFIGS)}\n")

    for idx, cfg in enumerate(CANDIDATE_CONFIGS, 1):
        d = cfg["D"]
        k = cfg["K"]
        beam_size = k * 6
        disc = cfg["discount"]
        alpha = cfg["alpha_vq"]
        name = cfg["name"]

        print(f"[{idx}/{len(CANDIDATE_CONFIGS)}] Evaluating: {name}...")

        os.environ["MY_WM_SEARCH_DEPTH"] = str(d)
        os.environ["MY_WM_BEAM_SIZE"] = str(beam_size)
        os.environ["MY_WM_TREE_DISCOUNT"] = str(disc)
        os.environ["MY_WM_ALPHA_VQ"] = str(alpha)

        t0 = time.time()
        try:
            metrics = evaluate_model(
                agent_name="my_spatial_qwm_agent",
                model_file=str(CHECKPOINT),
                n_rounds=rounds_per_config,
                scenario="classic",
                seed=seed,
                silence_errors=True,
                no_save=True,
            )
            elapsed = time.time() - t0

            ea = metrics["evaluated_agent"]
            rb = metrics["rule_based_avg"]

            res = {
                "name": name,
                "D": d,
                "K": k,
                "beam_size": beam_size,
                "discount": disc,
                "alpha_vq": alpha,
                "win_rate": ea["win_rate_pct"],
                "surv_rate": ea["survival_rate_pct"],
                "score": ea["mean_score"],
                "coins": ea["coins_per_round"],
                "kills": ea["kills_per_round"],
                "suicides": ea["total_suicides"],
                "invalid_pct": ea["invalid_pct"],
                "rb_win_rate": rb["win_rate_pct"],
                "rb_score": rb["mean_score"],
                "elapsed_s": elapsed,
            }
            results.append(res)
            print(f"    --> Win: {ea['win_rate_pct']:.1f}% | Surv: {ea['survival_rate_pct']:.1f}% | "
                  f"Score: {ea['mean_score']:.2f} | Coins: {ea['coins_per_round']:.2f} | Kills: {ea['kills_per_round']:.2f} | "
                  f"Suicides: {ea['total_suicides']} | Time: {elapsed:.1f}s\n")
        except Exception as ex:
            print(f"    --> Error evaluating {name}: {ex}\n")

    results.sort(key=lambda r: (r["win_rate"], r["score"], r["surv_rate"]), reverse=True)

    out_file = ROOT / "tree_search_tuning_results.json"
    with open(out_file, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n==========================================================================================")
    print("                      QWM TREE SEARCH HYPERPARAMETER TUNING RANKING                       ")
    print("==========================================================================================")
    print(f"{'Rank':<5} {'Config Name':<34} {'Win%':<7} {'Surv%':<7} {'Score':<7} {'Coins':<7} {'Kills':<7} {'Suic':<5} {'Inv%':<6}")
    print("-" * 90)
    for rank, r in enumerate(results, 1):
        print(f"#{rank:<4} {r['name']:<34} {r['win_rate']:>5.1f}% {r['surv_rate']:>5.1f}% {r['score']:>6.2f} {r['coins']:>6.2f} {r['kills']:>6.2f} {r['suicides']:>4} {r['invalid_pct']:>5.1f}%")
    print("==========================================================================================")
    print(f"Results saved to {out_file}\n")
    return results

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_tuning(rounds_per_config=n)
