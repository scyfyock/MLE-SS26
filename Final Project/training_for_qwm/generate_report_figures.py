"""
generate_report_figures.py -- Generates clean, reduced, publication-quality figures
for the Bomberman report (IEEEtran format):
1. figures/dqn_training_curve.pdf (.png) - Spatial Dueling DQN from start to end (12,800 rounds)
2. figures/qwm_training_curve.pdf (.png) - Spatial QWM training timeline (warmup + curriculum)
3. figures/qwm_depth_sweep.pdf (.png) - Planning horizon depth sweep (D=1..10, 1000 rounds/task)
4. figures/world_model_horizon_error.pdf (.png) - Compounding latent error over horizon H=1..8
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure matplotlib for crisp publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 11,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

FIG_DIRS = ['text_files/figures', 'figures']
for d in FIG_DIRS:
    os.makedirs(d, exist_ok=True)

def save_fig(fig, base_name):
    for d in FIG_DIRS:
        pdf_path = os.path.join(d, f"{base_name}.pdf")
        png_path = os.path.join(d, f"{base_name}.png")
        fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"Saved {base_name}.pdf and .png to {FIG_DIRS}")
    plt.close(fig)

def smooth(arr, window=100):
    return pd.Series(arr).rolling(window, min_periods=max(1, window//5)).mean().values


# ============================================================================
# 1. DQN Training Curve (my_spatial_dqn_agent)
# ============================================================================
def plot_dqn_training():
    csv_path = 'agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/training_stats-spatial-dqn.csv'
    df = pd.read_csv(csv_path)
    rounds = df['round'].values
    
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.2), sharex=True, dpi=300)
    
    stage_boundaries = [
        (0, 2000, r'Stage 1: Coin Heaven', '#e8f4f8'),
        (2000, 6200, r'Stage 2: Loot Crate (Solo)', '#fdf6e2'),
        (6200, 8800, r'Stage 3: Passive Opponents', '#f4ecf7'),
        (8800, 12800, r'Stage 4: Rule-Based + Self-Play', '#eafaf1'),
    ]
    
    for ax in axes:
        for x0, x1, label, col in stage_boundaries:
            ax.axvspan(x0, x1, color=col, alpha=0.55, lw=0)
        ax.grid(True)
    
    # Label stages cleanly above panel 0
    short_labels = ['Stage 1\n(Coins)', 'Stage 2\n(Crates)', 'Stage 3\n(Passive)', 'Stage 4\n(Rule-Based & SP)']
    for (x0, x1, _, _), sl in zip(stage_boundaries, short_labels):
        axes[0].text((x0 + x1) / 2, 53, sl, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#2c3e50')
    
    # Panel 1: Score & Reward
    score_sm = smooth(df['score'].values, 100)
    axes[0].plot(rounds, df['score'].values, color='#3498db', alpha=0.15, lw=0.6)
    axes[0].plot(rounds, score_sm, color='#1f618d', label='Score / Round (avg 100)', lw=1.8)
    axes[0].set_ylabel('Score / Round')
    axes[0].set_ylim(-2, 72)
    axes[0].legend(loc='lower left', framealpha=0.85)
    axes[0].set_title('(a) Spatial Dueling DQN: Game Score across Curriculum', pad=12)
    
    # Panel 2: TD Error & Epsilon
    td_sm = smooth(df['mean_td_error'].values, 100)
    axes[1].plot(rounds, df['mean_td_error'].values, color='#e67e22', alpha=0.2, lw=0.6)
    axes[1].plot(rounds, td_sm, color='#b9770e', label='TD Error (avg 100)', lw=1.8)
    axes[1].set_ylabel('TD Error (Bellman)')
    axes[1].set_ylim(0, 1.8)
    
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(rounds, df['epsilon'].values, color='#7f8c8d', ls=':', lw=1.5, label=r'Exploration $\epsilon$')
    ax1_twin.set_ylabel(r'$\epsilon$', color='#5d6d7e')
    ax1_twin.set_ylim(-0.05, 1.05)
    
    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', framealpha=0.85)
    axes[1].set_title('(b) Learning Convergence & Exploration Schedule')
    
    # Panel 3: Survival vs Suicide Rate
    surv_sm = smooth(df['survived'].values, 100) * 100
    suic_sm = smooth(df['killed_self'].values, 100) * 100
    axes[2].plot(rounds, surv_sm, color='#27ae60', lw=1.8, label='Survival Rate % (avg 100)')
    axes[2].plot(rounds, suic_sm, color='#c0392b', lw=1.8, label='Suicide Rate % (avg 100)')
    axes[2].set_ylabel('Rate (%)')
    axes[2].set_ylim(-5, 105)
    axes[2].set_xlabel('Curriculum Training Rounds')
    axes[2].legend(loc='center right', framealpha=0.85)
    axes[2].set_title('(c) Survival vs. Suicide Rate (Showing Vulnerability without Planning)')
    
    fig.tight_layout()
    save_fig(fig, 'dqn_training_curve')


# ============================================================================
# 2. QWM Training Curve (my_spatial_qwm_agent)
# ============================================================================
def plot_qwm_training():
    warmup_csv = 'agent_code/my_spatial_qwm_agent/runs/run_2026-09-06_23-46-34_spatial_qwm/training_stats-spatial-qwm.csv'
    curr_csv = 'agent_code/my_spatial_qwm_agent/runs/run_2026-09-10_19-43-12_spatial_qwm/training_stats-spatial-qwm.csv'
    
    df_warm = pd.read_csv(warmup_csv)
    df_warm0 = df_warm[df_warm['stage'] == 's0_wm_warmup'].copy()
    
    df_curr = pd.read_csv(curr_csv)
    
    # Total rounds: warmup (900) + full curriculum (12,000) = 12,900
    w_len = len(df_warm0)
    c_len = len(df_curr)
    
    combined_rounds = np.arange(1, w_len + c_len + 1)
    
    # Extract losses
    trans_warm = df_warm0['trans_loss'].values
    rew_warm = df_warm0['rew_loss'].values
    surv_warm = df_warm0['survived'].values
    suic_warm = df_warm0['killed_self'].values
    score_warm = df_warm0['score'].values
    win_warm = np.zeros(w_len)
    
    trans_curr = df_curr['trans_loss'].values
    rew_curr = df_curr['rew_loss'].values
    surv_curr = df_curr['survived'].values
    suic_curr = df_curr['killed_self'].values
    score_curr = df_curr['score'].values
    win_curr = df_curr['won'].values if 'won' in df_curr.columns else np.zeros(c_len)
    
    all_trans = np.concatenate([trans_warm, trans_curr])
    all_rew = np.concatenate([rew_warm, rew_curr])
    all_surv = np.concatenate([surv_warm, surv_curr])
    all_suic = np.concatenate([suic_warm, suic_curr])
    all_score = np.concatenate([score_warm, score_curr])
    all_win = np.concatenate([win_warm, win_curr])
    
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.4), sharex=True, dpi=300)
    
    qwm_stages = [
        (0, w_len, r'$s_0$: Warmup (Frozen DQN)', '#fadbd8'),
        (w_len, w_len + 1000, r'$s_1$: Coins', '#e8f4f8'),
        (w_len + 1000, w_len + 2600, r'$s_2$: Crates', '#fdf6e2'),
        (w_len + 2600, w_len + 4000, r'$s_3$: Passive', '#f4ecf7'),
        (w_len + 4000, w_len + 6000, r'$s_4$: Classic RB', '#eafaf1'),
        (w_len + 6000, w_len + 8000, r'$s_5$: Loot RB', '#fef9e7'),
        (w_len + 8000, w_len + 10000, r'$s_6$: Mixed Loot', '#ebf5fb'),
        (w_len + 10000, w_len + 12000, r'$s_7$: Mixed Classic', '#eaeded'),
    ]
    
    for ax in axes:
        for x0, x1, _, col in qwm_stages:
            ax.axvspan(x0, x1, color=col, alpha=0.55, lw=0)
        ax.grid(True)
    
    # Stage labels above panel 0
    axes[0].text(w_len / 2, 0.46, r'$s_0$', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#922b21')
    axes[0].text(w_len + 500, 0.46, r'$s_1$', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[0].text(w_len + 1800, 0.46, r'$s_2$', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[0].text(w_len + 3300, 0.46, r'$s_3$', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[0].text(w_len + 5000, 0.46, r'$s_4$', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[0].text(w_len + 7000, 0.46, r'$s_5$', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[0].text(w_len + 9000, 0.46, r'$s_6$', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[0].text(w_len + 11000, 0.46, r'$s_7$', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Panel 1: World Model Losses
    axes[0].plot(combined_rounds, smooth(all_trans, 80), color='#c0392b', lw=1.8, label=r'Transition Residual $\Delta z$ Loss')
    axes[0].plot(combined_rounds, smooth(all_rew, 80), color='#d35400', lw=1.6, ls='--', label=r'Reward Head Loss')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_ylim(-0.02, 0.56)
    axes[0].legend(loc='upper right', framealpha=0.85)
    axes[0].set_title('(a) Latent World Model Convergence (Warmup & Joint Training)', pad=16)
    
    # Panel 2: Score & Win Rate
    axes[1].plot(combined_rounds, smooth(all_score, 100), color='#1f618d', lw=1.8, label='Score / Round (avg 100)')
    axes[1].set_ylabel('Score / Round')
    axes[1].set_ylim(-2, 55)
    
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(combined_rounds[w_len:], smooth(all_win[w_len:], 100) * 100, color='#2e4053', ls=':', lw=1.5, label='Win Rate % (avg 100)')
    ax1_twin.set_ylabel('Win Rate (%)', color='#2e4053')
    ax1_twin.set_ylim(-5, 105)
    
    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', framealpha=0.85)
    axes[1].set_title('(b) Task Performance & Win Rate Progression')
    
    # Panel 3: Survival vs Suicides
    qwm_surv_sm = smooth(all_surv, 100) * 100
    qwm_suic_sm = smooth(all_suic, 100) * 100
    axes[2].plot(combined_rounds, qwm_surv_sm, color='#27ae60', lw=1.8, label='Survival Rate % (avg 100)')
    axes[2].plot(combined_rounds, qwm_suic_sm, color='#922b21', lw=1.8, label='Suicide Rate % (avg 100)')
    axes[2].set_ylabel('Rate (%)')
    axes[2].set_ylim(-5, 105)
    axes[2].set_xlabel('Curriculum Training Rounds ($s_0$ Warmup through $s_7$ Mixed Classic)')
    axes[2].legend(loc='center right', framealpha=0.85)
    axes[2].set_title('(c) Survival & Suicide Suppression under World Model Grounding')
    
    fig.tight_layout()
    save_fig(fig, 'qwm_training_curve')


# ============================================================================
# 3. Tree Search Depth Sweep (qwm_depth_sweep)
# ============================================================================
def plot_depth_sweep():
    # Data from 1000-round benchmark on matched random seeds (eval_tasks_qwm_agent_veryclean_2026-09-23_14-12-52.md)
    depths = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    task4_score = np.array([3.54, 3.77, 3.85, 3.89, 3.91, 3.86, 3.90, 3.85, 3.93, 3.82])
    task4_win = np.array([35.4, 38.8, 39.4, 39.3, 40.7, 41.1, 40.5, 40.3, 41.1, 40.2])
    task2_suicides = np.array([94, 110, 127, 138, 151, 161, 174, 160, 154, 164])
    latency_ms = np.array([34.5, 39.8, 44.7, 49.8, 55.5, 61.5, 68.0, 74.9, 81.5, 87.9])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7), dpi=300)
    
    # Left: Score & Win rate vs Depth
    color_score = '#1f618d'
    color_win = '#27ae60'
    
    ax1.plot(depths, task4_score, marker='o', color=color_score, lw=1.8, label='Mean Score / Rnd')
    ax1.axvline(x=7, color='#8e44ad', ls='--', lw=1.4, alpha=0.85, label=r'Optuna Chosen ($D=7$)')
    ax1.set_xlabel('Lookahead Search Depth $D$')
    ax1.set_ylabel('Competitive Score', color=color_score)
    ax1.tick_params(axis='y', labelcolor=color_score)
    ax1.set_ylim(3.3, 4.1)
    ax1.grid(True)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(depths, task4_win, marker='s', color=color_win, lw=1.6, ls='-.', label='Win Rate (%)')
    ax1_twin.set_ylabel('Win Rate (%)', color=color_win)
    ax1_twin.tick_params(axis='y', labelcolor=color_win)
    ax1_twin.set_ylim(33, 43)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=7.5, framealpha=0.85)
    ax1.set_title('(a) Competitive Combat vs. Lookahead Depth')
    
    # Right: Latency vs Depth
    bars = ax2.bar(depths, latency_ms, color='#f39c12', alpha=0.75, edgecolor='#d35400', width=0.6, label='Step Latency')
    ax2.axhline(y=500, color='#c0392b', ls=':', lw=1.5, label='Timeout Limit (500 ms)')
    ax2.axvline(x=7, color='#8e44ad', ls='--', lw=1.4, alpha=0.85)
    ax2.set_xlabel('Lookahead Search Depth $D$')
    ax2.set_ylabel('Decision Latency (ms / step)')
    ax2.set_ylim(0, 120)
    ax2.set_title('(b) Inference Decision Latency')
    ax2.grid(True)
    
    # Annotate D=1 (pure backbone) and D=7 (Optuna deployment)
    ax2.text(1, 40, '34.5 ms\n(Backbone)', ha='center', va='bottom', fontsize=7.5, color='#2c3e50')
    ax2.text(7, 75, '68.0 ms\n(Optuna D=7)', ha='center', va='bottom', fontsize=7.5, color='#6c3483', fontweight='bold')
    
    fig.tight_layout()
    save_fig(fig, 'qwm_depth_sweep')


# ============================================================================
# 4. Multi-Step Latent Rollout Horizon Error (world_model_horizon_error)
# ============================================================================
def plot_horizon_error():
    # Data from future_val_eval-spatial-qwm.json (tested on 1000 held-out rollouts)
    horizons = np.arange(1, 9)
    mae = np.array([1.645, 2.230, 2.600, 2.860, 3.095, 3.264, 3.387, 3.452])
    rmse = np.array([2.295, 3.008, 3.574, 3.890, 4.212, 4.398, 4.549, 4.611])
    
    fig, ax = plt.subplots(figsize=(4.5, 2.6), dpi=300)
    ax.plot(horizons, mae, marker='o', color='#2980b9', lw=1.8, label='Mean Absolute Error (MAE)')
    ax.plot(horizons, rmse, marker='^', color='#c0392b', lw=1.8, label='Root Mean Squared Error (RMSE)')
    
    ax.axvspan(1, 5, color='#eafaf1', alpha=0.6, label=r'Accurate Planning Region ($H \leq 5$)')
    ax.axvspan(5, 8.5, color='#fadbd8', alpha=0.4, label=r'Compounding Drift Zone ($H > 5$)')
    
    ax.set_xlabel('Imagined Rollout Horizon $H$ (Steps)')
    ax.set_ylabel('Latent Value Prediction Error')
    ax.set_title('Compounding Prediction Error in Latent World Model')
    ax.set_xlim(0.7, 8.3)
    ax.set_ylim(1.0, 5.5)
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    fig.tight_layout()
    save_fig(fig, 'world_model_horizon_error')


if __name__ == '__main__':
    print("Generating plots...")
    plot_dqn_training()
    plot_qwm_training()
    plot_depth_sweep()
    plot_horizon_error()
    print("Done!")


