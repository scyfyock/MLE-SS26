#!/usr/bin/env python3
"""
train_small_model_phase2.py -- Phase 2 Environment RL Training for SmallPathRewardModel.

Location: agent_code/my_spatial_qwm_agent/

Trains the Small Path Reward Model (~5,893 parameters, dynActivation) via
policy gradient reinforcement learning (REINFORCE with moving baseline and anchor regularization)
directly in the BombeRLe game environment, while keeping the base Spatial QWM backbone 100% frozen.

Supports multi-stage / multi-environment training across:
1. Stage 6 (s4_vs_rule_based):
   Scenario: 'classic'
   Opponents: ['rule_based_agent', 'rule_based_agent', 'rule_based_agent']

2. Stage 7 (s7_vs_mixed_classic):
   Scenario: 'classic'
   Opponents: ['my_spatial_dqn_agent', 'my_spatial_qwm_agent', 'rule_based_agent']
"""

import argparse
import csv
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

CSV_FIELDS = [
    "phase", "epoch", "round",
    "train_loss", "val_loss", "val_mae", "val_r2",
    "mean_reward", "sum_reward", "score", "coins", "crates",
    "kills", "suicides", "survived", "won", "steps", "wall_time_s"
]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Root directory resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

AGENT_DIR = Path(__file__).resolve().parent

import events as e
import settings as s
from environment import BombeRLeWorld, WorldArgs

from agent_code.my_spatial_qwm_agent.path_penalty_model import (
    FEATURE_DIM,
    SmallPathRewardModel,
    load_small_model,
    save_small_model,
)
from agent_code.my_spatial_qwm_agent.plot_small_model import plot_small_model_stats


TARGET_ENVIRONMENTS = {
    # -------------------------------------------------------------------------
    # Main Curriculum Stages (Matching train_curriculum.py STAGE_CONFIG)
    # -------------------------------------------------------------------------
    "s0_wm_warmup": {
        "env_id": 0,
        "name": "s0_wm_warmup",
        "scenario": "loot-crate",
        "opponents": ["peaceful_agent", "coin_collector_agent"],
        "default_rounds": 150,
        "description": "Stage 0: Warmup exploration vs peaceful & coin_collector (Loot Crate)",
    },
    "s1_coin_heaven": {
        "env_id": 1,
        "name": "s1_coin_heaven",
        "scenario": "coin-heaven",
        "opponents": ["coin_collector_agent"],
        "default_rounds": 250,
        "description": "Stage 1: Pure coin collection & navigation vs coin_collector (Coin Heaven)",
    },
    "s2_classic_alone": {
        "env_id": 2,
        "name": "s2_classic_alone",
        "scenario": "loot-crate",
        "opponents": ["coin_collector_agent"],
        "default_rounds": 400,
        "description": "Stage 2: Crate clearance & coin gathering vs coin_collector (Loot Crate)",
    },
    "s3_vs_passive": {
        "env_id": 3,
        "name": "s3_vs_passive",
        "scenario": "loot-crate",
        "opponents": ["peaceful_agent", "coin_collector_agent"],
        "default_rounds": 350,
        "description": "Stage 3: Multi-agent evasion & competition vs peaceful & coin_collector (Loot Crate)",
    },
    "s4_vs_rule_based": {
        "env_id": 4,
        "name": "s4_vs_rule_based",
        "scenario": "classic",
        "opponents": ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
        "default_rounds": 500,
        "description": "Stage 4: Tactical combat & bomb survival vs 3x rule_based_agent (Classic)",
    },
    "s5_vs_rule_based_loot": {
        "env_id": 5,
        "name": "s5_vs_rule_based_loot",
        "scenario": "loot-crate",
        "opponents": ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
        "default_rounds": 500,
        "description": "Stage 5: High-density crate blasting & combat vs 3x rule_based_agent (Loot Crate)",
    },
    "s6_vs_mixed_loot": {
        "env_id": 6,
        "name": "s6_vs_mixed_loot",
        "scenario": "loot-crate",
        "opponents": ["my_spatial_dqn_agent", "my_spatial_qwm_agent", "rule_based_agent"],
        "default_rounds": 500,
        "description": "Stage 6: Mixed adversarial combat vs DQN + QWM clone + rule_based (Loot Crate)",
    },
    "s7_vs_mixed_classic": {
        "env_id": 7,
        "name": "s7_vs_mixed_classic",
        "scenario": "classic",
        "opponents": ["my_spatial_dqn_agent", "my_spatial_qwm_agent", "rule_based_agent"],
        "default_rounds": 500,
        "description": "Stage 7: Full tournament combat vs DQN + QWM clone + rule_based (Classic)",
    },
    # -------------------------------------------------------------------------
    # Dedicated Self-Play Stages
    # -------------------------------------------------------------------------
    "sp_classic": {
        "env_id": 8,
        "name": "sp_classic",
        "scenario": "classic",
        "opponents": ["my_spatial_qwm_agent", "my_spatial_qwm_agent", "my_spatial_qwm_agent"],
        "default_rounds": 100,
        "description": "Self-Play: Symmetric combat vs 3x my_spatial_qwm_agent (Classic)",
    },
    "sp_loot": {
        "env_id": 9,
        "name": "sp_loot",
        "scenario": "loot-crate",
        "opponents": ["my_spatial_qwm_agent", "my_spatial_qwm_agent", "my_spatial_qwm_agent"],
        "default_rounds": 100,
        "description": "Self-Play: Symmetric crate & bomb combat vs 3x my_spatial_qwm_agent (Loot Crate)",
    },
    "sp_coins": {
        "env_id": 10,
        "name": "sp_coins",
        "scenario": "coin-heaven",
        "opponents": ["my_spatial_qwm_agent", "my_spatial_qwm_agent", "my_spatial_qwm_agent"],
        "default_rounds": 100,
        "description": "Self-Play: Symmetric coin race vs 3x my_spatial_qwm_agent (Coin Heaven)",
    },
}

ALL_CURRICULUM_STAGES: List[str] = [
    "s0_wm_warmup",
    "s1_coin_heaven",
    "s2_classic_alone",
    "s3_vs_passive",
    "s4_vs_rule_based",
    "s5_vs_rule_based_loot",
    "s6_vs_mixed_loot",
    "s7_vs_mixed_classic",
]

STAGE_ALIASES: Dict[str, List[str]] = {
    "all": ALL_CURRICULUM_STAGES,
    "curriculum": ALL_CURRICULUM_STAGES,
    "main": ALL_CURRICULUM_STAGES,
    "combat": ["s4_vs_rule_based", "s5_vs_rule_based_loot", "s6_vs_mixed_loot", "s7_vs_mixed_classic"],
    "loot": ["s0_wm_warmup", "s2_classic_alone", "s3_vs_passive", "s5_vs_rule_based_loot", "s6_vs_mixed_loot"],
    "coins": ["s1_coin_heaven"],
    "self_play": ["sp_classic", "sp_loot", "sp_coins"],
    "s0": ["s0_wm_warmup"],
    "s1": ["s1_coin_heaven"],
    "s2": ["s2_classic_alone"],
    "s3": ["s3_vs_passive"],
    "s4": ["s4_vs_rule_based"],
    "s5": ["s5_vs_rule_based_loot"],
    "s6": ["s6_vs_mixed_loot"],
    "s7": ["s7_vs_mixed_classic"],
    "0": ["s0_wm_warmup"],
    "1": ["s1_coin_heaven"],
    "2": ["s2_classic_alone"],
    "3": ["s3_vs_passive"],
    "4": ["s4_vs_rule_based"],
    "5": ["s5_vs_rule_based_loot"],
    "6": ["s6_vs_mixed_loot"],
    "7": ["s7_vs_mixed_classic"],
}


def resolve_stages(stage_inputs: Optional[List[str]]) -> List[str]:
    """Resolves stage names, aliases, and numbers into valid TARGET_ENVIRONMENTS keys."""
    if not stage_inputs:
        return list(ALL_CURRICULUM_STAGES)
    resolved: List[str] = []
    for item in stage_inputs:
        key = str(item).strip().lower()
        if key in STAGE_ALIASES:
            for s_name in STAGE_ALIASES[key]:
                if s_name not in resolved:
                    resolved.append(s_name)
        elif item in TARGET_ENVIRONMENTS:
            if item not in resolved:
                resolved.append(item)
        elif key in TARGET_ENVIRONMENTS:
            if key not in resolved:
                resolved.append(key)
        else:
            # Substring match (e.g. "coin_heaven" -> "s1_coin_heaven")
            matched = False
            for target_k in TARGET_ENVIRONMENTS:
                if key in target_k:
                    if target_k not in resolved:
                        resolved.append(target_k)
                    matched = True
                    break
            if not matched:
                print(f"Warning: Unknown stage '{item}'. Valid stages: {list(TARGET_ENVIRONMENTS.keys()) + list(STAGE_ALIASES.keys())}")
    return resolved if resolved else list(ALL_CURRICULUM_STAGES)


def create_run_folder(base_runs_dir: Optional[str] = None) -> str:
    """Creates a timestamped run folder matching the big model's convention."""
    runs_dir = base_runs_dir or os.path.join(str(AGENT_DIR), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(runs_dir, f"run_{timestamp}_small_model")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_last_completed_round(csv_path: str) -> int:
    """Reads the CSV and returns the maximum integer round found in Phase 2 rows."""
    if not os.path.isfile(csv_path):
        return 0
    max_round = 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_val = row.get("round", "")
                if r_val:
                    try:
                        r_int = int(float(r_val))
                        if r_int > max_round:
                            max_round = r_int
                    except (ValueError, TypeError):
                        pass
    except Exception as err:
        print(f"Warning: Could not parse round from {csv_path}: {err}")
    return max_round


def compute_turn_reward(agent_events: List[str]) -> float:
    """Computes turn rewards focused primarily on point acquisition and aggression."""
    reward = 0.0
    reward_map = {
        e.COIN_COLLECTED: 2.0,      # Direct point gain (+1 pt)
        e.CRATE_DESTROYED: 0.4,     # Unlocks coins and paths
        e.KILLED_OPPONENT: 6.0,     # Direct kill (+5 pts) + elimination
        e.KILLED_SELF: -8.0,        # Suicide deterrent
        e.GOT_KILLED: -4.0,         # Elimination deterrent
        e.SURVIVED_ROUND: 0.0,      # Passive survival alone does NOT award points
        e.INVALID_ACTION: -0.1,
        e.BOMB_DROPPED: 0.05,
    }
    for ev in agent_events:
        reward += reward_map.get(ev, 0.0)
    return reward


def _rollout_worker_process(worker_id: int, env_key: str, seed: int, pipe_conn):
    """Persistent background worker process running BombeRLeWorld rollouts on demand."""
    local_model = SmallPathRewardModel(input_dim=FEATURE_DIM, hidden_dim=64).to("cpu")
    local_model.eval()

    cfg = TARGET_ENVIRONMENTS[env_key]
    scenario = cfg["scenario"]
    opponents = cfg["opponents"]

    args = WorldArgs(
        no_gui=True,
        fps=0,
        turn_based=False,
        update_interval=0.0,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir="logs",
        save_stats=False,
        match_name=None,
        seed=seed,
        silence_errors=True,
        scenario=scenario,
    )

    agent_specs = [("my_spatial_qwm_agent", False)] + [(opp, False) for opp in opponents]
    world = BombeRLeWorld(args, agent_specs)

    our_agent = world.agents[0]
    fake_self = our_agent.backend.runner.fake_self
    fake_self.use_small_model = True
    fake_self.small_model = local_model
    fake_self.tree_search = True
    fake_self.search_depth = 5
    fake_self.beam_size = 18
    fake_self.tree_discount = 0.08
    fake_self.alpha_vq = 0.16

    for a in world.agents[1:]:
        if hasattr(a, 'backend') and hasattr(a.backend, 'runner') and hasattr(a.backend.runner, 'fake_self'):
            f_self = a.backend.runner.fake_self
            if getattr(f_self, 'code_name', '') == 'my_spatial_qwm_agent' or 'qwm' in (getattr(a, 'name', '') or ''):
                f_self.use_small_model = True
                f_self.small_model = local_model
                f_self.tree_search = True
                f_self.search_depth = 5
                f_self.beam_size = 18
                f_self.tree_discount = 0.08
                f_self.alpha_vq = 0.16

    pipe_conn.send(("READY", worker_id))

    while True:
        try:
            cmd, payload = pipe_conn.recv()
        except EOFError:
            break
        if cmd == "STOP":
            break
        elif cmd == "RUN_ROUND":
            weights_cpu, temperature, round_seed = payload
            local_model.load_state_dict(weights_cpu)
            fake_self.exploration_temperature = temperature
            if round_seed is not None:
                world.rng = np.random.default_rng(round_seed)

            world.new_round()
            round_trajectory = []
            round_rewards = []
            step_count = 0
            fake_self._last_step_data = None

            while world.running:
                world.do_step(None)
                step_count += 1
                step_data = getattr(fake_self, '_last_step_data', None)
                if step_data is not None:
                    turn_rew = compute_turn_reward(our_agent.events)
                    round_trajectory.append({
                        'features': step_data['features'],
                        'base_scores': step_data['base_scores'],
                        'valid_mask': step_data['valid_mask'],
                        'action_index': step_data.get('action_index', 0),
                        'reward': turn_rew,
                    })
                    round_rewards.append(turn_rew)
                    fake_self._last_step_data = None

            score = our_agent.score
            coins = our_agent.statistics.get("coins", 0)
            crates = our_agent.statistics.get("crates", 0)
            kills = our_agent.statistics.get("kills", 0)
            suicides = our_agent.statistics.get("suicides", 0)
            survived = int(not our_agent.dead)

            other_scores = [a.score for a in world.agents if a is not our_agent]
            max_other_score = max(other_scores) if other_scores else 0

            point_win = (score > max_other_score and score > 0)
            point_tie = (score == max_other_score and score > 0)
            won = int(point_win or point_tie)

            if point_win:
                point_lead = score - max_other_score
                terminal_reward = 8.0 + 1.0 * min(point_lead, 4)
            elif point_tie:
                terminal_reward = 4.0
            elif score < max_other_score:
                point_deficit = max_other_score - score
                terminal_reward = -4.0 - 0.5 * min(point_deficit, 4)
            elif score == 0 and survived:
                terminal_reward = -2.0
            else:
                terminal_reward = -1.0

            if round_trajectory:
                round_trajectory[-1]['reward'] += terminal_reward
                round_rewards.append(terminal_reward)

            round_stats = {
                "score": score,
                "coins": coins,
                "crates": crates,
                "kills": kills,
                "suicides": suicides,
                "survived": survived,
                "won": won,
                "steps": step_count,
                "sum_reward": float(sum(round_rewards)),
                "mean_reward": float(np.mean(round_rewards)) if round_rewards else 0.0,
            }
            pipe_conn.send(("ROUND_DONE", (worker_id, round_trajectory, round_stats)))

    world.end()
    pipe_conn.close()


def train_single_environment(
    model: SmallPathRewardModel,
    optimizer: torch.optim.Optimizer,
    anchor_weights: Dict[str, torch.Tensor],
    env_key: str,
    n_rounds: int,
    round_offset: int,
    baseline_return: float,
    run_dir: str,
    gamma: float = 0.95,
    anchor_reg: float = 0.005,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_workers: int = 4,
) -> Tuple[float, float]:
    """Trains the small model in a single environment for n_rounds with multi-worker parallelism."""
    cfg = TARGET_ENVIRONMENTS[env_key]
    scenario = cfg["scenario"]
    opponents = cfg["opponents"]
    env_name = cfg["name"]

    actual_workers = max(1, min(n_workers, n_rounds))
    print("\n" + "=" * 85)
    print(f" TRAINING ENVIRONMENT: {env_name} ({cfg['description']})")
    print(f" Scenario: {scenario} | Rounds: {n_rounds} | Cumulative Round Start: {round_offset + 1} | Workers: {actual_workers}")
    print("=" * 85)

    stats_csv_run = os.path.join(run_dir, "training_stats-small-model.csv")
    state_json_run = os.path.join(run_dir, "training_state-small-model.json")
    plot_png_run = os.path.join(run_dir, "training_curves-small-model.png")
    env_ckpt_run = os.path.join(run_dir, f"small_model_{env_name}.pt")
    phase2_ckpt_run = os.path.join(run_dir, "small_model_phase2.pt")
    default_ckpt_run = os.path.join(run_dir, "small_model.pt")

    active_phase2_ckpt = os.path.join(str(AGENT_DIR), "small_model_phase2.pt")
    active_default_ckpt = os.path.join(str(AGENT_DIR), "small_model.pt")
    active_stats_csv = os.path.join(str(AGENT_DIR), "training_stats-small-model.csv")
    active_state_json = os.path.join(str(AGENT_DIR), "training_state-small-model.json")
    active_plot_png = os.path.join(str(AGENT_DIR), "training_curves-small-model.png")

    if not os.path.isfile(stats_csv_run):
        with open(stats_csv_run, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

    recent_wins = []
    recent_survivals = []
    recent_scores = []
    best_eval_metric = -float("inf")
    start_time = time.time()

    print("-" * 88)
    print(f"{'Round':>6} | {'Score':>6} | {'Coins':>5} | {'Crates':>6} | {'Kills':>5} | {'Surv':>5} | {'Won':>4} | {'Return':>8} | {'Loss':>8} | {'Win Rate (10)':>13}")
    print("-" * 88)

    ctx = mp.get_context("spawn")
    workers = []
    conns = []

    try:
        for w_i in range(actual_workers):
            p_conn, c_conn = ctx.Pipe()
            w_p = ctx.Process(
                target=_rollout_worker_process,
                args=(w_i, env_key, seed + w_i * 100, c_conn),
            )
            w_p.start()
            workers.append(w_p)
            conns.append(p_conn)

        for p_conn in conns:
            p_conn.recv()

        completed_rounds = 0
        while completed_rounds < n_rounds:
            batch_size = min(actual_workers, n_rounds - completed_rounds)
            progress = completed_rounds / max(1, n_rounds - 1)
            temperature = max(0.35, 0.80 - 0.45 * progress)
            weights_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            for i in range(batch_size):
                round_seed = seed + (round_offset + completed_rounds + i) * 17
                conns[i].send(("RUN_ROUND", (weights_cpu, temperature, round_seed)))

            batch_trajectories = []
            batch_stats_list = []
            for i in range(batch_size):
                msg, (wid, round_traj, round_stats) = conns[i].recv()
                batch_trajectories.append(round_traj)
                batch_stats_list.append(round_stats)

            # Batched Policy Gradient Update
            valid_trajs = [t for t in batch_trajectories if t]
            loss_val = 0.0

            if valid_trajs:
                model.train()
                optimizer.zero_grad()
                total_batch_loss = torch.tensor(0.0, device=device)

                for round_traj in valid_trajs:
                    T = len(round_traj)
                    returns = np.zeros(T, dtype=np.float32)
                    running_return = 0.0
                    for t in reversed(range(T)):
                        running_return = round_traj[t]['reward'] + gamma * running_return
                        returns[t] = running_return

                    mean_ep_return = float(np.mean(returns))
                    baseline_return = 0.95 * baseline_return + 0.05 * mean_ep_return
                    advantages = returns - baseline_return
                    adv_std = float(np.std(advantages))
                    if adv_std > 1e-4:
                        advantages = (advantages - float(np.mean(advantages))) / (adv_std + 1e-4)

                    traj_loss = torch.tensor(0.0, device=device)
                    for t in range(T):
                        step_item = round_traj[t]
                        feats = torch.from_numpy(step_item['features']).to(device)
                        base_s = torch.from_numpy(step_item['base_scores']).to(device)
                        valid_m = torch.from_numpy(step_item['valid_mask']).to(device)
                        a_idx = step_item['action_index']
                        adv = float(advantages[t])

                        deltas = model(feats).squeeze(-1)
                        adj_scores = base_s + deltas
                        adj_scores_masked = adj_scores.clone()
                        adj_scores_masked[~valid_m] = -1e9

                        log_probs = F.log_softmax(adj_scores_masked / max(0.1, temperature), dim=-1)
                        action_log_prob = log_probs[a_idx]
                        traj_loss = traj_loss + (-action_log_prob * adv)

                    traj_loss = traj_loss / float(T)
                    total_batch_loss = total_batch_loss + traj_loss

                total_batch_loss = total_batch_loss / float(len(valid_trajs))

                anchor_loss = torch.tensor(0.0, device=device)
                for param_name, param in model.named_parameters():
                    if param_name in anchor_weights:
                        anchor_loss = anchor_loss + torch.sum((param - anchor_weights[param_name].to(device)) ** 2)

                full_loss = total_batch_loss + anchor_reg * anchor_loss
                full_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_val = float(full_loss.item())

            for i, r_stats in enumerate(batch_stats_list):
                completed_rounds += 1
                cum_round = round_offset + completed_rounds

                score = r_stats["score"]
                coins = r_stats["coins"]
                crates = r_stats["crates"]
                kills = r_stats["kills"]
                suicides = r_stats["suicides"]
                survived = r_stats["survived"]
                won = r_stats["won"]
                step_count = r_stats["steps"]
                sum_reward = r_stats["sum_reward"]
                mean_reward = r_stats["mean_reward"]

                recent_wins.append(won)
                recent_survivals.append(survived)
                recent_scores.append(score)
                if len(recent_wins) > 10:
                    recent_wins.pop(0)
                    recent_survivals.pop(0)
                    recent_scores.pop(0)

                win_rate_10 = float(np.mean(recent_wins)) * 100.0
                elapsed = time.time() - start_time

                marker = ""
                eval_score = (win_rate_10 * 3.0) + (float(np.mean(recent_scores)) * 4.0)
                if eval_score > best_eval_metric and completed_rounds >= 5:
                    best_eval_metric = eval_score
                    save_small_model(model, env_ckpt_run)
                    save_small_model(model, phase2_ckpt_run)
                    save_small_model(model, default_ckpt_run)
                    shutil.copyfile(phase2_ckpt_run, active_phase2_ckpt)
                    shutil.copyfile(default_ckpt_run, active_default_ckpt)
                    marker = " *"

                print(f"{cum_round:>6d} | {score:>6d} | {coins:>5d} | {crates:>6d} | {kills:>5d} | {survived:>5d} | {won:>4d} | {sum_reward:>8.2f} | {loss_val:>8.4f} | {win_rate_10:>12.1f}%{marker}")

                with open(stats_csv_run, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    writer.writerow({
                        "phase": f"phase2_{env_name}",
                        "epoch": "",
                        "round": cum_round,
                        "train_loss": "",
                        "val_loss": "",
                        "val_mae": "",
                        "val_r2": "",
                        "mean_reward": f"{mean_reward:.4f}",
                        "sum_reward": f"{sum_reward:.4f}",
                        "score": score,
                        "coins": coins,
                        "crates": crates,
                        "kills": kills,
                        "suicides": suicides,
                        "survived": survived,
                        "won": won,
                        "steps": step_count,
                        "wall_time_s": f"{elapsed:.1f}",
                    })

                if completed_rounds % 5 == 0 or completed_rounds == n_rounds:
                    shutil.copyfile(stats_csv_run, active_stats_csv)
                    plot_small_model_stats(stats_csv_run, plot_png_run, window=10)
                    if os.path.isfile(plot_png_run):
                        shutil.copyfile(plot_png_run, active_plot_png)

                    state_data = {
                        "run_dir": os.path.abspath(run_dir),
                        "phase": f"phase2_{env_name}",
                        "completed_rounds": cum_round,
                        "stage_rounds": completed_rounds,
                        "stage_total_rounds": n_rounds,
                        "recent_win_rate": float(np.mean(recent_wins)),
                        "recent_survival_rate": float(np.mean(recent_survivals)),
                        "recent_mean_score": float(np.mean(recent_scores)),
                        "best_eval_metric": float(best_eval_metric),
                        "model_parameters": sum(p.numel() for p in model.parameters()),
                        "model_architecture": str(model.net),
                        "last_updated": datetime.now().isoformat(),
                    }
                    with open(state_json_run, "w") as f:
                        json.dump(state_data, f, indent=2)
                    shutil.copyfile(state_json_run, active_state_json)

    finally:
        for conn in conns:
            try:
                conn.send(("STOP", None))
                conn.close()
            except Exception:
                pass
        for p in workers:
            try:
                p.terminate()
                p.join(timeout=1.0)
            except Exception:
                pass

    print("-" * 88)
    print(f"[{env_name} Training Complete] Finished {n_rounds} rounds.")
    print(f"  -> Final 10-Round Win Rate: {np.mean(recent_wins)*100.0:.1f}%")
    print(f"  -> Final 10-Round Survival Rate: {np.mean(recent_survivals)*100.0:.1f}%")
    print(f"  -> Final 10-Round Mean Score: {np.mean(recent_scores):.2f}")
    print(f"  -> Stage Checkpoint Saved: {env_ckpt_run}")

    return baseline_return, best_eval_metric


def run_phase2_training(
    stages: Optional[List[str]] = None,
    rounds_per_stage: int = 50,
    lr: float = 3e-4,
    gamma: float = 0.95,
    anchor_reg: float = 0.005,
    initial_model_path: Optional[str] = None,
    continue_training: bool = False,
    run_dir: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_workers: Optional[int] = None,
) -> str:
    """Master multi-stage Phase 2 training across requested environments."""
    stages = resolve_stages(stages)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 4)

    print("=" * 88)
    print(" PHASE 2: MULTI-ENVIRONMENT RL FINE-TUNING (SMALL PATH REWARD MODEL)")
    print(f" Backbone: 100% FROZEN | Small Model: dynActivation (~5,893 parameters)")
    print(f" Target Environments: {stages} ({rounds_per_stage} rounds each)")
    print(f" Parallel Workers: {n_workers} | Compute Device: {device}")
    print("=" * 88)

    # Setup Run Folder
    if run_dir is None:
        run_dir = create_run_folder()
    print(f"[Phase 2] Output Run Directory: {run_dir}")

    stats_csv_run = os.path.join(run_dir, "training_stats-small-model.csv")
    state_json_run = os.path.join(run_dir, "training_state-small-model.json")
    phase1_ckpt_run = os.path.join(run_dir, "small_model_phase1.pt")
    phase2_ckpt_run = os.path.join(run_dir, "small_model_phase2.pt")
    default_ckpt_run = os.path.join(run_dir, "small_model.pt")

    active_phase1_ckpt = os.path.join(str(AGENT_DIR), "small_model_phase1.pt")
    active_phase2_ckpt = os.path.join(str(AGENT_DIR), "small_model_phase2.pt")
    active_default_ckpt = os.path.join(str(AGENT_DIR), "small_model.pt")
    active_stats_csv = os.path.join(str(AGENT_DIR), "training_stats-small-model.csv")
    active_state_json = os.path.join(str(AGENT_DIR), "training_state-small-model.json")

    # Preserve previous telemetry, state, and Phase 1 priors in the run folder
    if not os.path.isfile(stats_csv_run) and os.path.isfile(active_stats_csv):
        shutil.copyfile(active_stats_csv, stats_csv_run)
        print(f"[Phase 2] Copied active CSV telemetry from {active_stats_csv}")
    if not os.path.isfile(state_json_run) and os.path.isfile(active_state_json):
        shutil.copyfile(active_state_json, state_json_run)
    if not os.path.isfile(phase1_ckpt_run) and os.path.isfile(active_phase1_ckpt):
        shutil.copyfile(active_phase1_ckpt, phase1_ckpt_run)

    # Resolve initial model checkpoint to load
    init_path = initial_model_path
    if not init_path or not os.path.isfile(init_path):
        if continue_training:
            candidates = [
                default_ckpt_run,
                active_default_ckpt,
                phase2_ckpt_run,
                active_phase2_ckpt,
                phase1_ckpt_run,
                active_phase1_ckpt,
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    init_path = candidate
                    break
        else:
            init_path = phase1_ckpt_run if os.path.isfile(phase1_ckpt_run) else active_phase1_ckpt

    if not init_path or not os.path.isfile(init_path):
        raise FileNotFoundError(f"Initial model checkpoint not found (searched {init_path}). Run Phase 1 first.")

    print(f"[Phase 2] Loading small model weights from: {init_path}")
    model = load_small_model(init_path, device=device)
    model.train()

    save_small_model(model, default_ckpt_run)
    save_small_model(model, phase2_ckpt_run)
    if not continue_training and not os.path.isfile(phase1_ckpt_run):
        save_small_model(model, phase1_ckpt_run)

    # Load anchor weights for regularization:
    # Always prefer Phase 1 checkpoint to preserve anti-looping / anti-waiting priors
    anchor_path = phase1_ckpt_run
    if not os.path.isfile(anchor_path):
        anchor_path = active_phase1_ckpt
    if not os.path.isfile(anchor_path):
        anchor_path = init_path

    anchor_model = load_small_model(anchor_path, device=device)
    anchor_weights = {k: v.clone().detach() for k, v in anchor_model.state_dict().items()}
    print(f"[Phase 2] Policy anchor regularization pegged to: {anchor_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    baseline_return = 0.0

    # Determine starting round offset from existing CSV if continuing
    cum_rounds = 0
    if continue_training:
        cum_rounds = get_last_completed_round(stats_csv_run)
        print(f"[Phase 2 Continuation] Resuming at round {cum_rounds + 1} (previously completed: {cum_rounds} rounds)")

    for idx, env_key in enumerate(stages):
        if env_key not in TARGET_ENVIRONMENTS:
            print(f"Warning: Unknown environment '{env_key}', skipping.")
            continue

        baseline_return, _ = train_single_environment(
            model=model,
            optimizer=optimizer,
            anchor_weights=anchor_weights,
            env_key=env_key,
            n_rounds=rounds_per_stage,
            round_offset=cum_rounds,
            baseline_return=baseline_return,
            run_dir=run_dir,
            gamma=gamma,
            anchor_reg=anchor_reg,
            seed=seed + idx * 100,
            device=device,
            n_workers=n_workers,
        )
        cum_rounds += rounds_per_stage

    # Final save
    save_small_model(model, phase2_ckpt_run)
    save_small_model(model, default_ckpt_run)
    shutil.copyfile(phase2_ckpt_run, active_phase2_ckpt)
    shutil.copyfile(default_ckpt_run, active_default_ckpt)

    print("\n" + "=" * 88)
    print(f"[Phase 2 Training Complete] Finished all {len(stages)} environments ({cum_rounds} cumulative rounds).")
    print(f"  -> Best Checkpoint Saved to: {phase2_ckpt_run}")
    print(f"  -> Active Model Updated: {active_default_ckpt}")
    print("=" * 88)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Environment RL Training for SmallPathRewardModel")
    parser.add_argument("--stages", nargs="+", default=["all"],
                        help="Environment stages to train in (default: all main curriculum stages. Supports: all, curriculum, combat, loot, coins, or specific stages like s0..s7)")
    parser.add_argument("--rounds", type=int, default=50, help="Number of RL rounds per environment")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Number of parallel rollout workers (default: min(CPU count, 4))")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--anchor-reg", type=float, default=0.005, help="Anchor regularization weight")
    parser.add_argument("--init-model", type=str, default=None, help="Path to initial small model checkpoint")
    parser.add_argument("--continue", "-c", dest="continue_training", action="store_true", default=False,
                        help="Continue training from existing small_model.pt and previous round stats")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific run directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.init_model and not args.continue_training:
        args.continue_training = True

    run_phase2_training(
        stages=args.stages,
        rounds_per_stage=args.rounds,
        lr=args.lr,
        gamma=args.gamma,
        anchor_reg=args.anchor_reg,
        initial_model_path=args.init_model,
        continue_training=args.continue_training,
        run_dir=args.run_dir,
        seed=args.seed,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
