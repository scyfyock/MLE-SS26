#!/usr/bin/env python3
"""
plot_small_model.py -- Visualizes training curves for the Small Auxiliary Path Reward Model.

Generates a multi-panel telemetry dashboard:
  1. Phase 1 Supervised Loss (Train vs Val Loss)
  2. Phase 1 Accuracy (Val MAE & Val R^2)
  3. Phase 2 RL Episode Return & Loss Progression
  4. Phase 2 Competitive Performance (Win Rate, Survival Rate, Coins, Kills)

Usage:
    python agent_code/my_spatial_qwm_agent/plot_small_model.py
    python agent_code/my_spatial_qwm_agent/plot_small_model.py --csv path/to/stats.csv --out path/to/curves.png
"""

import argparse
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_STATS = HERE / "training_stats-small-model.csv"
DEFAULT_OUT = HERE / "training_curves-small-model.png"


def smooth(series, window: int = 10):
    if len(series) < 2:
        return series
    w = max(1, min(window, len(series)))
    return pd.Series(series).rolling(w, min_periods=1).mean().values


def plot_small_model_stats(csv_path: str, out_png: str, window: int = 10, title_prefix: str = "") -> bool:
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) < 10:
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False

    if df.empty or len(df) < 1:
        return False

    p1_df = df[df['phase'].astype(str) == 'phase1'].copy()
    p2_df = df[df['phase'].astype(str).str.startswith('phase2')].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=120)
    fig.patch.set_facecolor('#0f111a')

    # Color palette
    dark_bg = '#1a1c29'
    grid_col = '#2c3144'
    text_col = '#e0e6ed'
    accent_blue = '#4fc3f7'
    accent_cyan = '#00e5ff'
    accent_green = '#00e676'
    accent_purple = '#b388ff'
    accent_orange = '#ff9100'
    accent_red = '#ff5252'

    title = f"{title_prefix}Small Path Reward Model (dynActivation, ~5.8k params) | Dual-Phase Training"
    fig.suptitle(title, fontsize=14, fontweight='bold', color=text_col, y=0.98)

    for ax in axes.flat:
        ax.set_facecolor(dark_bg)
        ax.grid(True, linestyle='--', color=grid_col, alpha=0.6)
        ax.tick_params(colors=text_col, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(grid_col)

    # ------------------------------------------------------------------------
    # Panel 1: Phase 1 Supervised Loss
    # ------------------------------------------------------------------------
    ax1 = axes[0, 0]
    if not p1_df.empty and 'epoch' in p1_df.columns:
        epochs = p1_df['epoch'].values
        train_l = p1_df['train_loss'].values
        val_l = p1_df['val_loss'].values

        ax1.plot(epochs, train_l, color=accent_blue, lw=1.8, label='Train Loss (SmoothL1)')
        ax1.plot(epochs, val_l, color=accent_orange, lw=2.2, label='Val Loss')

        best_idx = np.argmin(val_l)
        ax1.scatter([epochs[best_idx]], [val_l[best_idx]], color=accent_green, s=70, zorder=5,
                    label=f'Best Val: {val_l[best_idx]:.4f} (ep {epochs[best_idx]})')
        ax1.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=text_col, fontsize=8)
    ax1.set_title('Phase 1: Supervised Penalty Imitation Loss', color=text_col, fontsize=11, fontweight='bold')
    ax1.set_xlabel('Epoch', color=text_col, fontsize=9)
    ax1.set_ylabel('Loss', color=text_col, fontsize=9)

    # ------------------------------------------------------------------------
    # Panel 2: Phase 1 Generalization & Fit (R^2 & MAE)
    # ------------------------------------------------------------------------
    ax2 = axes[0, 1]
    if not p1_df.empty and 'epoch' in p1_df.columns and 'val_r2' in p1_df.columns:
        epochs = p1_df['epoch'].values
        val_r2 = p1_df['val_r2'].values
        val_mae = p1_df['val_mae'].values

        line1 = ax2.plot(epochs, val_r2, color=accent_cyan, lw=2.0, label='Val R^2 Score')
        ax2.set_ylabel('R^2 Score (1.0 = Perfect)', color=accent_cyan, fontsize=9)
        ax2.set_ylim([max(0.7, float(np.min(val_r2)) - 0.05), 1.005])

        ax2_twin = ax2.twinx()
        ax2_twin.tick_params(colors=text_col, labelsize=9)
        for spine in ax2_twin.spines.values():
            spine.set_color(grid_col)
        line2 = ax2_twin.plot(epochs, val_mae, color=accent_purple, lw=1.8, ls='--', label='Val MAE')
        ax2_twin.set_ylabel('Mean Absolute Error', color=accent_purple, fontsize=9)

        best_r2 = np.max(val_r2)
        min_mae = np.min(val_mae)
        ax2.set_title(f'Phase 1: Optuna Target Alignment (Best R^2={best_r2:.4f}, MAE={min_mae:.4f})',
                      color=text_col, fontsize=11, fontweight='bold')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, facecolor=dark_bg, edgecolor=grid_col, labelcolor=text_col, fontsize=8)
    else:
        ax2.set_title('Phase 1: Alignment Accuracy', color=text_col, fontsize=11, fontweight='bold')

    # ------------------------------------------------------------------------
    # Panel 3: Phase 2 RL Returns & Rewards
    # ------------------------------------------------------------------------
    ax3 = axes[1, 0]
    if not p2_df.empty and 'round' in p2_df.columns:
        rounds = p2_df['round'].values
        w_p2 = max(1, min(window, len(p2_df)))

        if 'mean_reward' in p2_df.columns:
            rew = p2_df['mean_reward'].values
            ax3.plot(rounds, rew, color=accent_blue, alpha=0.3, label='Round Reward (raw)')
            ax3.plot(rounds, smooth(rew, w_p2), color=accent_cyan, lw=2.2, label=f'Avg Reward ({w_p2}-rd)')

        if 'score' in p2_df.columns:
            score = p2_df['score'].values
            ax3.plot(rounds, smooth(score, w_p2), color=accent_green, lw=2.0, ls='-.', label=f'Avg Score ({w_p2}-rd)')

        ax3.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=text_col, fontsize=8)
    ax3.set_title('Phase 2: Environment RL Return Progression', color=text_col, fontsize=11, fontweight='bold')
    ax3.set_xlabel('Round', color=text_col, fontsize=9)
    ax3.set_ylabel('Environmental Return / Score', color=text_col, fontsize=9)

    # ------------------------------------------------------------------------
    # Panel 4: Phase 2 Competitive Performance (Win & Survival Rates)
    # ------------------------------------------------------------------------
    ax4 = axes[1, 1]
    if not p2_df.empty and 'round' in p2_df.columns:
        rounds = p2_df['round'].values
        w_p2 = max(1, min(window, len(p2_df)))

        if 'won' in p2_df.columns:
            win_rate = smooth(p2_df['won'].values * 100.0, w_p2)
            ax4.plot(rounds, win_rate, color=accent_green, lw=2.2, label=f'Win Rate % ({w_p2}-rd avg)')

        if 'survived' in p2_df.columns:
            surv_rate = smooth(p2_df['survived'].values * 100.0, w_p2)
            ax4.plot(rounds, surv_rate, color=accent_orange, lw=2.0, ls='--', label=f'Survival Rate % ({w_p2}-rd avg)')

        if 'kills' in p2_df.columns:
            kills = smooth(p2_df['kills'].values, w_p2)
            ax4.plot(rounds, kills * 20.0, color=accent_red, lw=1.5, ls=':', label=f'Kills (scaled x20)')

        ax4.set_ylim([-5, 105])
        ax4.legend(facecolor=dark_bg, edgecolor=grid_col, labelcolor=text_col, fontsize=8)
    ax4.set_title('Phase 2: Competitive Tournament Performance', color=text_col, fontsize=11, fontweight='bold')
    ax4.set_xlabel('Round', color=text_col, fontsize=9)
    ax4.set_ylabel('Rate (%)', color=text_col, fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Atomic save
    tmp_out = str(out_png) + ".tmp.png"
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(tmp_out, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    os.replace(tmp_out, out_png)
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot Small Path Reward Model training curves.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_STATS), help="Path to training_stats-small-model.csv")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Path to output PNG image")
    parser.add_argument("--window", type=int, default=10, help="Smoothing window")
    args = parser.parse_args()

    success = plot_small_model_stats(args.csv, args.out, window=args.window)
    if success:
        print(f"Successfully generated training curves plot: {args.out}")
    else:
        print(f"Could not generate plot from {args.csv}")


if __name__ == "__main__":
    main()
