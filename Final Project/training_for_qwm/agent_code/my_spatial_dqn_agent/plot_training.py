"""
plot_training.py -- Live and offline training curve visualization for my_spatial_dqn_agent.

Usage:
    python agent_code/my_spatial_dqn_agent/plot_training.py
    python agent_code/my_spatial_dqn_agent/plot_training.py --csv path/to/stats.csv --out path/to/curves.png
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_STATS = HERE / "training_stats-spatial-dqn.csv"
DEFAULT_OUT = HERE / "training_curves-spatial-dqn.png"


def smooth(series, window: int):
    """Moving average with adaptive window."""
    if len(series) < 2:
        return series
    w = max(1, min(window, len(series)))
    return pd.Series(series).rolling(w, min_periods=1).mean().values


def plot_stats_file(csv_path: str, out_png: str, window: int = 25, title_prefix: str = "") -> bool:
    """Plot training curves from training_stats-spatial-dqn.csv.

    Saves atomically via a temporary file to avoid partial reads by viewers.
    """
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) < 10:
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False

    if df.empty or len(df) < 1:
        return False

    # Standardize column types
    numeric_cols = [
        'round', 'epsilon', 'score', 'coins', 'kills', 'crates', 'bombs', 'survived',
        'steps', 'invalid_acts', 'killed_self', 'got_killed', 'mean_reward', 'sum_reward',
        'mean_td_error', 'unsafe_actions', 'wall_time_s'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    rounds = df['round'].values if 'round' in df.columns else np.arange(1, len(df) + 1)
    w = max(1, min(window, len(df)))

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), dpi=95)
    total_rounds = len(df)
    current_stage = str(df['stage'].iloc[-1]) if 'stage' in df.columns else 'training'
    latest_score = float(df['score'].iloc[-1]) if 'score' in df.columns and not pd.isna(df['score'].iloc[-1]) else 0.0
    latest_eps = float(df['epsilon'].iloc[-1]) if 'epsilon' in df.columns and not pd.isna(df['epsilon'].iloc[-1]) else 0.0

    fig.suptitle(
        f"{title_prefix}Spatial Dueling DQN | Stage: {current_stage} | Round {total_rounds} | "
        f"Latest Score: {latest_score:.1f} | Epsilon: {latest_eps:.3f}",
        fontsize=13,
        fontweight='bold',
    )

    # ------------------------------------------------------------------------
    # Panel 1: Reward & Epsilon
    # ------------------------------------------------------------------------
    ax1 = axes[0, 0]
    if 'mean_reward' in df.columns and not df['mean_reward'].isna().all():
        mr = df['mean_reward'].fillna(0.0).values
        ax1.plot(rounds, mr, color='steelblue', alpha=0.25, label='Mean Reward (raw)')
        ax1.plot(rounds, smooth(mr, w), color='navy', lw=2.0, label=f'Mean Reward (avg {w})')
    ax1.set_title(f'Reward per Round (avg {w})', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Round', fontsize=9)
    ax1.set_ylabel('Reward', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    if 'epsilon' in df.columns and not df['epsilon'].isna().all():
        ax1_twin = ax1.twinx()
        ax1_twin.plot(rounds, df['epsilon'].values, color='crimson', ls='--', lw=1.5, label='Epsilon')
        ax1_twin.set_ylabel('Epsilon', color='crimson', fontsize=9)
        ax1_twin.tick_params(axis='y', labelcolor='crimson')
        ax1_twin.set_ylim(-0.05, 1.05)
    ax1.legend(loc='lower right', fontsize=8)

    # ------------------------------------------------------------------------
    # Panel 2: Score & Steps Survived
    # ------------------------------------------------------------------------
    ax2 = axes[0, 1]
    if 'score' in df.columns and not df['score'].isna().all():
        sc = df['score'].fillna(0.0).values
        ax2.plot(rounds, sc, color='darkorange', alpha=0.25, label='Score (raw)')
        ax2.plot(rounds, smooth(sc, w), color='darkorange', lw=2.0, label=f'Score (avg {w})')
    ax2.set_title(f'Score & Steps per Round (avg {w})', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Round', fontsize=9)
    ax2.set_ylabel('Score', color='darkorange', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.grid(True, linestyle='--', alpha=0.4)

    if 'steps' in df.columns and not df['steps'].isna().all():
        ax2_twin = ax2.twinx()
        st = smooth(df['steps'].fillna(0.0).values, w)
        ax2_twin.plot(rounds, st, color='teal', ls=':', lw=1.8, label=f'Steps (avg {w})')
        ax2_twin.set_ylabel('Steps Survived', color='teal', fontsize=9)
        ax2_twin.tick_params(axis='y', labelcolor='teal')
        ax2_twin.set_ylim(0, 420)
    ax2.legend(loc='upper left', fontsize=8)

    # ------------------------------------------------------------------------
    # Panel 3: Objectives: Coins, Crates, Bombs, Kills
    # ------------------------------------------------------------------------
    ax3 = axes[1, 0]
    if 'coins' in df.columns and not df['coins'].isna().all():
        ax3.plot(rounds, smooth(df['coins'].fillna(0.0).values, w), color='goldenrod', lw=2.0, label='Coins/rd')
    if 'crates' in df.columns and not df['crates'].isna().all():
        ax3.plot(rounds, smooth(df['crates'].fillna(0.0).values, w), color='saddlebrown', lw=2.0, label='Crates/rd')
    if 'bombs' in df.columns and not df['bombs'].isna().all():
        ax3.plot(rounds, smooth(df['bombs'].fillna(0.0).values, w), color='purple', ls=':', lw=1.8, label='Bombs/rd')
    if 'kills' in df.columns and not df['kills'].isna().all():
        ax3.plot(rounds, smooth(df['kills'].fillna(0.0).values, w), color='red', lw=1.8, label='Kills/rd')
    ax3.set_title(f'Game Objectives & Activity (avg {w})', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Round', fontsize=9)
    ax3.set_ylabel('Count / Round', fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='upper left', fontsize=8)

    # ------------------------------------------------------------------------
    # Panel 4: Q-Learning TD Error
    # ------------------------------------------------------------------------
    ax4 = axes[1, 1]
    if 'mean_td_error' in df.columns and not df['mean_td_error'].isna().all():
        td = df['mean_td_error'].dropna().values
        td_rounds = rounds[~df['mean_td_error'].isna()]
        if len(td) > 0:
            ax4.plot(td_rounds, td, color='darkviolet', alpha=0.25, label='TD Error (raw)')
            ax4.plot(td_rounds, smooth(td, w), color='darkviolet', lw=2.0, label=f'TD Error (avg {w})')
    ax4.set_title(f'DQN Bellman TD Error (avg {w})', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Round', fontsize=9)
    ax4.set_ylabel('Loss / TD Error', fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.4)
    ax4.legend(loc='upper right', fontsize=8)

    # ------------------------------------------------------------------------
    # Panel 5: Survival Rate & Suicide Rate
    # ------------------------------------------------------------------------
    ax5 = axes[2, 0]
    if 'survived' in df.columns and not df['survived'].isna().all():
        surv = smooth(df['survived'].fillna(0.0).values * 100.0, w)
        ax5.plot(rounds, surv, color='forestgreen', lw=2.0, label=f'Survival % (avg {w})')
    if 'killed_self' in df.columns and not df['killed_self'].isna().all():
        suic = smooth(df['killed_self'].fillna(0.0).values * 100.0, w)
        ax5.plot(rounds, suic, color='crimson', ls='--', lw=1.8, label=f'Suicide % (avg {w})')
    ax5.set_title(f'Survival vs. Suicide Rate (avg {w})', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Round', fontsize=9)
    ax5.set_ylabel('Percentage (%)', fontsize=9)
    ax5.set_ylim(-5, 105)
    ax5.grid(True, linestyle='--', alpha=0.4)
    ax5.legend(loc='center right', fontsize=8)

    # ------------------------------------------------------------------------
    # Panel 6: Safety & Invalid Actions
    # ------------------------------------------------------------------------
    ax6 = axes[2, 1]
    if 'unsafe_actions' in df.columns and not df['unsafe_actions'].isna().all():
        ax6.plot(rounds, smooth(df['unsafe_actions'].fillna(0.0).values, w), color='orange', lw=1.8, label='Unsafe Moves/rd')
    if 'invalid_acts' in df.columns and not df['invalid_acts'].isna().all():
        ax6.plot(rounds, smooth(df['invalid_acts'].fillna(0.0).values, w), color='grey', ls='--', lw=1.8, label='Invalid Acts/rd')
    ax6.set_title(f'Safety Violations & Invalid Acts (avg {w})', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Round', fontsize=9)
    ax6.set_ylabel('Count / Round', fontsize=9)
    ax6.grid(True, linestyle='--', alpha=0.4)
    ax6.legend(loc='upper right', fontsize=8)

    # ------------------------------------------------------------------------
    # Stage Boundaries
    # ------------------------------------------------------------------------
    if 'stage' in df.columns:
        prev_stage = None
        for i, row in df.iterrows():
            st = str(row['stage'])
            if st and st != prev_stage:
                rnd = rounds[i]
                for ax in axes.flat:
                    ax.axvline(rnd, color='grey', ls=':', lw=0.9, alpha=0.7)
                axes[0, 0].text(
                    rnd, axes[0, 0].get_ylim()[1] * 0.9,
                    f" {st}",
                    fontsize=8,
                    rotation=90,
                    va='top',
                    color='#555',
                    fontweight='bold'
                )
                prev_stage = st

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(out_dir, exist_ok=True)
    tmp_out = os.path.join(out_dir, f".tmp_{os.path.basename(out_png)}")
    try:
        fig.savefig(tmp_out, format='png', dpi=95)
        os.replace(tmp_out, out_png)
        return True
    except Exception:
        if os.path.isfile(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass
        return False
    finally:
        plt.clf()
        plt.close('all')
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves for my_spatial_dqn_agent")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_STATS), help="Path to training_stats CSV")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Path to output PNG")
    parser.add_argument("--window", type=int, default=25, help="Rolling average window")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.is_file():
        # Fallback to latest run if default doesn't exist
        runs_dir = HERE / "runs"
        if runs_dir.is_dir():
            run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
            for rd in run_dirs:
                cand = rd / "training_stats-spatial-dqn.csv"
                if cand.is_file():
                    csv_path = cand
                    break

    if not csv_path.is_file():
        print(f"Error: Stats file {csv_path} not found.")
        sys.exit(1)

    print(f"Plotting {csv_path} -> {args.out} (window={args.window})...")
    ok = plot_stats_file(str(csv_path), str(args.out), window=args.window)
    if ok:
        print(f"Successfully saved training curves to {args.out}")
    else:
        print("Failed to generate plot (empty data or plotting error).")
        sys.exit(1)


if __name__ == "__main__":
    main()
