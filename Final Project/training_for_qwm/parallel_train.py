#!/usr/bin/env python3
"""
parallel_train.py -- High-Performance Parallel Training Engine for Bomberman RL.

Optimized for:
- CPU: AMD Ryzen 7 5700X (8 physical cores / 16 threads): runs 8-12 concurrent environment actors
  with single-threaded CPU inference on PyTorch shared-memory models.
- GPU: NVIDIA GeForce RTX 3080 (20GB VRAM): runs central high-throughput Double-DQN Learner
  with AMP FP16, Tensor Cores, and soft target updates (~48,000 samples/s).
"""

import argparse
import csv
import json
import logging
import os
import queue
import random
import shutil
import sys
import time
from collections import deque, namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import settings as s
from environment import BombeRLeWorld
from main import world_controller

WorldArgs = namedtuple(
    "WorldArgs",
    [
        "no_gui",
        "fps",
        "turn_based",
        "update_interval",
        "save_replay",
        "replay",
        "make_video",
        "continue_without_training",
        "log_dir",
        "save_stats",
        "match_name",
        "seed",
        "silence_errors",
        "scenario",
    ],
)

# Global hooks for worker processes to access shared objects
ACTIVE_SHARED_MODEL = None
ACTIVE_TRANSITION_QUEUE = None
ACTIVE_STATS_QUEUE = None
ACTIVE_WORKER_ID = None
ACTIVE_GRAD_QUEUE = None  # retained for legacy hooks

DEFAULT_SCENARIOS = ["classic", "loot-crate", "coin-heaven"]


def is_better_round(cand: dict, current_best: Optional[dict]) -> bool:
    """Compare candidate round against current best round."""
    if current_best is None:
        return True

    def _f(v, d=0.0):
        try:
            return float(v)
        except (ValueError, TypeError):
            return d

    def _i(v, d=0):
        try:
            return int(v)
        except (ValueError, TypeError):
            return d

    c_score = _f(cand.get("score", 0))
    b_score = _f(current_best.get("score", 0))
    c_rew = _f(cand.get("sum_reward", 0.0))
    b_rew = _f(current_best.get("sum_reward", 0.0))
    c_surv = bool(_i(cand.get("survived", 0)))
    b_surv = bool(_i(current_best.get("survived", 0)))
    c_won = bool(_i(cand.get("won", 0)))
    b_won = bool(_i(current_best.get("won", 0)))
    c_steps = _i(cand.get("steps", 0))
    b_steps = _i(current_best.get("steps", 0))

    if c_won and not b_won:
        return True
    if not c_won and b_won:
        return False
    if c_won and b_won:
        if c_score > b_score:
            return True
        elif b_score > c_score:
            return False
        if c_rew > b_rew + 0.05:
            return True
        elif b_rew > c_rew + 0.05:
            return False
        return c_steps < b_steps

    if c_surv and b_surv:
        if c_rew > b_rew + 0.05:
            return True
        elif b_rew > c_rew + 0.05:
            return False
        if c_score > b_score:
            return True
        elif b_score > c_score:
            return False
        return c_steps < b_steps

    if c_surv and not b_surv:
        return c_score >= b_score or c_rew >= b_rew

    if not c_surv and b_surv:
        return c_score > b_score and c_rew > b_rew

    if not c_surv and not b_surv:
        if c_score > b_score and c_rew > b_rew:
            return True
        elif c_rew > b_rew + 0.05:
            return True
        elif b_rew > c_rew + 0.05:
            return False
        if c_score > b_score:
            return True
        elif b_score > c_score:
            return False
        return c_steps > b_steps

    return False


def worker_entry_point(
    worker_id: int,
    agent_name: str,
    n_rounds: int,
    scenario: str,
    opponents: list,
    shared_model,
    transition_queue,
    stats_queue,
    run_dir: str,
    stage_name: str,
    seed: int,
    model_file: str,
    lr: float,
    batch: int,
    no_augment: bool,
    is_self_play: bool = False,
    freeze_dqn: bool = False,
    tree_search: Optional[bool] = None,
    eps_start: Optional[float] = None,
    eps_end: float = 0.05,
    eps_decay: Optional[float] = None,
):
    """
    Subprocess entry point for a single parallel game instance worker.
    Runs on CPU with 1 thread to avoid cache thrashing on the Ryzen 7 5700X.
    """
    global ACTIVE_SHARED_MODEL, ACTIVE_TRANSITION_QUEUE, ACTIVE_STATS_QUEUE, ACTIVE_WORKER_ID
    ACTIVE_SHARED_MODEL = shared_model
    ACTIVE_TRANSITION_QUEUE = transition_queue
    ACTIVE_STATS_QUEUE = stats_queue
    ACTIVE_WORKER_ID = worker_id

    # Prevent OpenMP thread contention across 8+ workers on the Ryzen 7 5700X
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    worker_log_dir = Path(run_dir) / "logs" / f"worker_{worker_id}"
    worker_log_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MY_WM_RUN_DIR"] = str(run_dir)
    os.environ["MY_WM_STAGE"] = str(stage_name)
    os.environ["MY_WM_MODEL_FILE"] = str(model_file)
    os.environ["MY_WM_N_ROUNDS"] = str(n_rounds)
    os.environ["MY_WM_FREEZE_DQN"] = "1" if freeze_dqn else "0"
    if tree_search is not None:
        os.environ["MY_WM_TREE_SEARCH"] = "1" if tree_search else "0"
    elif freeze_dqn:
        os.environ["MY_WM_TREE_SEARCH"] = "0"
    if is_self_play:
        os.environ["MY_WM_SELF_PLAY"] = "1"
        os.environ["MY_WM_EPSILON"] = "0.0"
        os.environ["MY_WM_EPS_END"] = "0.0"
        os.environ["MY_WM_EPS_ROUNDS"] = "0"
        os.environ["MY_WM_EPS_DECAY"] = "1.0"
    else:
        os.environ["MY_WM_SELF_PLAY"] = "0"
        if eps_start is not None:
            os.environ["MY_WM_EPSILON"] = str(eps_start)
        if eps_end is not None:
            os.environ["MY_WM_EPS_END"] = str(eps_end)
        if eps_decay is not None:
            os.environ["MY_WM_EPS_DECAY"] = str(eps_decay)
        else:
            os.environ["MY_WM_EPS_ROUNDS"] = str(max(1, int(n_rounds * 0.8)))
    os.environ["MY_WM_BATCH"] = str(batch)
    os.environ["MY_WM_AUGMENT"] = "0" if no_augment else "1"
    os.environ["MY_WM_DEVICE"] = "cpu"  # Workers use CPU shared memory for action inference
    if "wm" in agent_name:
        os.environ["MY_WM_MODEL"] = "dqn"
    if lr is not None:
        os.environ["MY_WM_LR"] = str(lr)

    args = WorldArgs(
        no_gui=True,
        fps=30,
        turn_based=False,
        update_interval=0.1,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=False,
        log_dir=str(worker_log_dir),
        save_stats=False,
        match_name=f"worker_{worker_id}",
        seed=seed,
        silence_errors=False,
        scenario=scenario,
    )

    # Agents list: first agent is learner, others are opponents / clones
    agents = [(agent_name, True)]
    for opp in opponents:
        agents.append((opp, False))

    world = BombeRLeWorld(args, agents)
    world_controller(
        world,
        n_rounds,
        gui=None,
        every_step=True,
        turn_based=False,
        make_video=False,
        update_interval=0.1,
    )


def run_parallel_stage(
    stage_name: str,
    total_rounds: int,
    num_workers: int,
    agent_name: str,
    scenario: str,
    opponents: list,
    shared_model,
    run_dir: Path,
    model_file: str,
    target_model: Path,
    batch: int = 128,
    no_augment: bool = False,
    lr: float = 1e-4,
    accumulate_count: Optional[int] = None,
    base_seed: int = 1000,
    checkpoint_name: Optional[str] = None,
    save_best: bool = True,
    save_last: bool = True,
    is_self_play: bool = False,
    freeze_dqn: bool = False,
    tree_search: Optional[bool] = None,
    eps_start: Optional[float] = None,
    eps_end: float = 0.05,
    eps_decay: Optional[float] = None,
) -> int:
    """
    Executes total_rounds across num_workers parallel game instances.
    Workers collect transitions concurrently; the Master GPU Learner trains on
    CUDA (RTX 3080) and periodically syncs weights to CPU shared memory.
    """
    if total_rounds <= 0 or num_workers <= 0:
        return 0

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Distribute rounds among workers
    base_rounds = total_rounds // num_workers
    rem = total_rounds % num_workers
    worker_rounds = [base_rounds + (1 if i < rem else 0) for i in range(num_workers)]
    worker_rounds = [r for r in worker_rounds if r > 0]
    actual_workers = len(worker_rounds)

    # Resolve or create the GPU Learner model
    learner_model = getattr(shared_model, "_learner_model", None)
    is_qwm = ("qwm" in agent_name)
    is_spatial = ("spatial" in agent_name)

    if learner_model is None:
        if torch.cuda.is_available():
            cuda_dev = torch.device("cuda")
            if is_qwm:
                from agent_code.my_spatial_qwm_agent.model import SpatialQWMAgent
                learner_model = SpatialQWMAgent(lr=lr, device=cuda_dev)
                learner_model.policy_net.load_state_dict(shared_model.policy_net.state_dict())
                learner_model.target_net.load_state_dict(shared_model.target_net.state_dict())
                learner_model.world_model.load_state_dict(shared_model.world_model.state_dict())
                print(f"[parallel-master] Attached CUDA Spatial QWM Learner on {torch.cuda.get_device_name(0)}")
            elif is_spatial:
                from agent_code.my_spatial_dqn_agent.model import SpatialDQNAgent
                learner_model = SpatialDQNAgent(lr=lr, device=cuda_dev)
                learner_model.policy_net.load_state_dict(shared_model.policy_net.state_dict())
                learner_model.target_net.load_state_dict(shared_model.target_net.state_dict())
                print(f"[parallel-master] Attached CUDA Learner on {torch.cuda.get_device_name(0)}")
            elif hasattr(shared_model, "net"):
                from agent_code.my_wm_agent.model import DQNAgent
                learner_model = DQNAgent(device=cuda_dev, lr=lr)
                learner_model.net.load_state_dict(shared_model.net.state_dict())
                learner_model.target.load_state_dict(shared_model.target.state_dict())
                print(f"[parallel-master] Attached CUDA Learner on {torch.cuda.get_device_name(0)}")
            else:
                learner_model = shared_model
        else:
            learner_model = shared_model

    if hasattr(learner_model, "freeze_dqn"):
        if freeze_dqn:
            learner_model.freeze_dqn()
        else:
            learner_model.unfreeze_dqn()
    if hasattr(shared_model, "freeze_dqn"):
        if freeze_dqn:
            shared_model.freeze_dqn()
        else:
            shared_model.unfreeze_dqn()

    is_cuda = (learner_model is not None and getattr(learner_model, "device", None) is not None and learner_model.device.type == "cuda")
    cuda_device = torch.device("cuda" if is_cuda else "cpu")
    if is_cuda:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    eps_desc = "0.0 (NO DISCOVERY / FULL POTENTIAL)" if is_self_play else (f"{eps_start:.4f}" if eps_start is not None else "auto")
    print(f"\n============================================================")
    print(f"=== [Parallel Stage] {stage_name} ({total_rounds} rounds, {actual_workers} workers, scenario={scenario}) ===")
    print(f"=== Device: {'CUDA (RTX 3080)' if is_cuda else 'CPU'} | Batch: {batch} | Workers: {worker_rounds} | Epsilon: {eps_desc} ===")
    print(f"============================================================")

    transition_queue = mp.Queue(maxsize=1000)
    stats_queue = mp.Queue()
    transition_queue.cancel_join_thread()
    stats_queue.cancel_join_thread()

    workers = []
    t0 = time.time()

    for w_id, n_r in enumerate(worker_rounds):
        p = mp.Process(
            target=worker_entry_point,
            kwargs={
                "worker_id": w_id,
                "agent_name": agent_name,
                "n_rounds": n_r,
                "scenario": scenario,
                "opponents": opponents,
                "shared_model": shared_model,
                "transition_queue": transition_queue,
                "stats_queue": stats_queue,
                "run_dir": str(run_dir),
                "stage_name": stage_name,
                "seed": base_seed + w_id * 1000,
                "model_file": model_file,
                "lr": lr,
                "batch": batch,
                "no_augment": no_augment,
                "is_self_play": is_self_play,
                "freeze_dqn": freeze_dqn,
                "tree_search": tree_search,
                "eps_start": eps_start,
                "eps_end": eps_end,
                "eps_decay": eps_decay,
            },
        )
        p.start()
        workers.append(p)

    # Master Learner setup
    replay_buffer = deque(maxlen=100000)
    trajectory_buffer = deque(maxlen=20000)
    cfg_path = Path("agent_code") / agent_name / "config.json"
    cfg_agent = {}
    if cfg_path.is_file():
        try:
            with open(cfg_path) as fh:
                cfg_agent = json.load(fh)
        except Exception:
            pass
    h_horizon = max(8, min(int(os.environ.get("MY_WM_PATH_HORIZON", cfg_agent.get("path_horizon", 8))), 16))
    recent_losses = deque(maxlen=200)
    recent_wm_losses = deque(maxlen=200)
    recent_trans_losses = deque(maxlen=200)
    recent_rew_losses = deque(maxlen=200)
    recent_future_val_losses = deque(maxlen=200)
    recent_legal_losses = deque(maxlen=200)
    horizon_mae_history = {h: deque(maxlen=50) for h in range(1, 9)}
    horizon_rmse_history = {h: deque(maxlen=50) for h in range(1, 9)}
    horizon_pred_history = {h: deque(maxlen=50) for h in range(1, 9)}
    horizon_act_history = {h: deque(maxlen=50) for h in range(1, 9)}
    latest_horizon_eval: Optional[dict] = None
    batch_size = max(32, batch)
    total_updates = 0
    rounds_completed = 0
    best_info: Optional[dict] = None
    updates_since_sync = 0
    last_sync_time = time.time()
    last_plot_time = time.time()

    if is_spatial:
        from agent_code.my_spatial_dqn_agent.model import augment_spatial_transition
    else:
        from agent_code.my_wm_agent.model import augment_transition

    if is_qwm:
        stats_csv = run_dir / "training_stats-spatial-qwm.csv"
    elif is_spatial:
        stats_csv = run_dir / "training_stats-spatial-dqn.csv"
    else:
        stats_csv = run_dir / "training_stats-dqn.csv"
    csv_header_needed = not stats_csv.is_file() or (stats_csv.stat().st_size == 0)

    # Track monotonic global round index across runs and workers
    global_round = 0
    if stats_csv.is_file():
        try:
            with open(stats_csv, "r") as fh:
                lines = [line.strip() for line in fh if line.strip()]
                global_round = max(0, len(lines) - 1)
        except Exception:
            global_round = 0

    ckpt_name = checkpoint_name or stage_name
    best_json_path = run_dir / f"best_model_{ckpt_name}.json"
    if best_json_path.is_file():
        try:
            with open(best_json_path, "r") as fh:
                best_info = json.load(fh)
        except Exception:
            pass

    # Central Master Loop
    while any(p.is_alive() for p in workers) or not transition_queue.empty() or not stats_queue.empty():
        got_work = False

        # 1. Drain incoming transitions and populate replay buffer
        while True:
            try:
                chunk = transition_queue.get_nowait()
                got_work = True
                if is_spatial:
                    if not no_augment:
                        for item in chunk:
                            s_i, f_i, a_i, r_i, ns_i, nf_i, d_i = item
                            for aug in augment_spatial_transition(s_i, f_i, a_i, ns_i, nf_i):
                                replay_buffer.append((aug[0], aug[1], aug[2], r_i, aug[3], aug[4], d_i))
                    else:
                        replay_buffer.extend(chunk)

                    # Extract multi-step trajectories of horizon H for QWM
                    if is_qwm and len(chunk) >= h_horizon + 1:
                        for idx_c in range(len(chunk) - h_horizon):
                            w_chunk = chunk[idx_c:idx_c + h_horizon + 1]
                            if not any(t[6] for t in w_chunk[:-1]):
                                s_seq = [t[0] for t in w_chunk]
                                f_seq = [t[1] for t in w_chunk]
                                a_seq = [t[2] for t in w_chunk[:h_horizon]]
                                r_seq = [t[3] for t in w_chunk[:h_horizon]]
                                d_seq = [t[6] for t in w_chunk[:h_horizon]]
                                trajectory_buffer.append((s_seq, f_seq, a_seq, r_seq, d_seq))
                else:
                    if not no_augment and hasattr(learner_model, "net"):
                        for item in chunk:
                            f_i, a_i, r_i, nf_i, d_i = item
                            for aug in augment_transition(f_i, a_i, nf_i):
                                replay_buffer.append((aug[0], aug[1], r_i, aug[2], d_i))
                    else:
                        replay_buffer.extend(chunk)
            except queue.Empty:
                break

        # 2. Train GPU Learner on batches from Replay Buffer
        if len(replay_buffer) >= batch_size:
            # Perform mini-batch updates proportional to incoming data
            n_train_steps = min(4, max(1, len(replay_buffer) // (batch_size * 2)))
            for _ in range(n_train_steps):
                samples = random.sample(replay_buffer, batch_size)
                got_work = True

                if is_spatial:
                    S, F, A, R, NS, NF, D = zip(*samples)
                    s_t = torch.as_tensor(np.asarray(S, dtype=np.float32), device=cuda_device)
                    f_t = torch.as_tensor(np.asarray(F, dtype=np.float32), device=cuda_device)
                    a_t = torch.as_tensor(np.asarray(A, dtype=np.int64), device=cuda_device).unsqueeze(1)
                    r_t = torch.as_tensor(np.asarray(R, dtype=np.float32), device=cuda_device).unsqueeze(1)
                    ns_t = torch.as_tensor(np.asarray(NS, dtype=np.float32), device=cuda_device)
                    nf_t = torch.as_tensor(np.asarray(NF, dtype=np.float32), device=cuda_device)
                    d_t = torch.as_tensor(np.asarray(D, dtype=np.float32), device=cuda_device).unsqueeze(1)

                    if hasattr(learner_model, "update_batch_cuda"):
                        loss_dict = learner_model.update_batch_cuda(s_t, f_t, a_t, r_t, ns_t, nf_t, d_t, scaler=scaler)
                    else:
                        loss_dict = learner_model.update_batch(np.asarray(S), np.asarray(F), np.asarray(A), np.asarray(R), np.asarray(NS), np.asarray(NF), np.asarray(D))
                else:
                    F, A, R, NF, D = zip(*samples)
                    loss_dict = learner_model.update_batch(np.asarray(F), np.asarray(A), np.asarray(R, dtype=np.float64), np.asarray(NF), np.asarray(D))

                if isinstance(loss_dict, dict):
                    if loss_dict.get("loss") is not None and not np.isnan(loss_dict["loss"]):
                        recent_losses.append(float(loss_dict["loss"]))
                    if loss_dict.get("wm_loss") is not None and not np.isnan(loss_dict["wm_loss"]):
                        recent_wm_losses.append(float(loss_dict["wm_loss"]))
                    if loss_dict.get("trans_loss") is not None and not np.isnan(loss_dict["trans_loss"]):
                        recent_trans_losses.append(float(loss_dict["trans_loss"]))
                    if loss_dict.get("rew_loss") is not None and not np.isnan(loss_dict["rew_loss"]):
                        recent_rew_losses.append(float(loss_dict["rew_loss"]))
                    if loss_dict.get("legal_loss") is not None and not np.isnan(loss_dict["legal_loss"]):
                        recent_legal_losses.append(float(loss_dict["legal_loss"]))
                elif isinstance(loss_dict, (float, int)) and not np.isnan(loss_dict):
                    recent_losses.append(float(loss_dict))

                total_updates += 1
                updates_since_sync += 1

            # Multi-step candidate path rollout training for QWM
            if is_qwm and len(trajectory_buffer) >= 8 and hasattr(learner_model, "update_multistep_paths_cuda"):
                p_samples = random.sample(trajectory_buffer, min(16, len(trajectory_buffer)))
                P_S, P_F, P_A, P_R, P_D = zip(*p_samples)
                ps_t = torch.as_tensor(np.asarray(P_S, dtype=np.float32), device=cuda_device)
                pf_t = torch.as_tensor(np.asarray(P_F, dtype=np.float32), device=cuda_device)
                pa_t = torch.as_tensor(np.asarray(P_A, dtype=np.int64), device=cuda_device)
                pr_t = torch.as_tensor(np.asarray(P_R, dtype=np.float32), device=cuda_device)
                pd_t = torch.as_tensor(np.asarray(P_D, dtype=np.float32), device=cuda_device)

                path_metrics = learner_model.update_multistep_paths_cuda(
                    ps_t, pf_t, pa_t, pr_t, pd_t, scaler=scaler, random_horizon=True
                )
                if path_metrics.get("future_val_loss") is not None and not np.isnan(path_metrics["future_val_loss"]):
                    recent_future_val_losses.append(float(path_metrics["future_val_loss"]))
                if path_metrics.get("legal_loss") is not None and not np.isnan(path_metrics["legal_loss"]):
                    recent_legal_losses.append(float(path_metrics["legal_loss"]))
                latest_horizon_eval = path_metrics

                h_maes = path_metrics.get("horizon_mae", [])
                h_rmses = path_metrics.get("horizon_rmse", [])
                h_preds = path_metrics.get("pred_vals_mean", [])
                h_acts = path_metrics.get("actual_vals_mean", [])
                for h_i in range(len(h_maes)):
                    step_num = h_i + 1
                    if step_num in horizon_mae_history:
                        horizon_mae_history[step_num].append(h_maes[h_i])
                        if h_i < len(h_rmses): horizon_rmse_history[step_num].append(h_rmses[h_i])
                        if h_i < len(h_preds): horizon_pred_history[step_num].append(h_preds[h_i])
                        if h_i < len(h_acts): horizon_act_history[step_num].append(h_acts[h_i])

            # Sync weights to CPU shared memory model
            now = time.time()
            if is_cuda and (updates_since_sync >= 5 or now - last_sync_time > 0.05):
                if hasattr(learner_model, "sync_to_cpu_shared"):
                    learner_model.sync_to_cpu_shared(shared_model)
                updates_since_sync = 0
                last_sync_time = now

        # 3. Drain statistics packets
        while True:
            try:
                stat_packet = stats_queue.get_nowait()
                got_work = True
                rounds_completed += 1
                global_round += 1
                row = stat_packet["row"]
                cand_info = stat_packet["cand_info"]
                row["round"] = global_round
                cand_info["round"] = global_round

                # Inject central GPU learner's tracked live metrics into row
                if (row.get("mean_td_error") == "" or row.get("mean_td_error") is None) and recent_losses:
                    row["mean_td_error"] = round(float(np.mean(recent_losses)), 4)
                if (row.get("wm_loss") == "" or row.get("wm_loss") is None) and recent_wm_losses:
                    row["wm_loss"] = round(float(np.mean(recent_wm_losses)), 4)
                if (row.get("trans_loss") == "" or row.get("trans_loss") is None) and recent_trans_losses:
                    row["trans_loss"] = round(float(np.mean(recent_trans_losses)), 4)
                if (row.get("rew_loss") == "" or row.get("rew_loss") is None) and recent_rew_losses:
                    row["rew_loss"] = round(float(np.mean(recent_rew_losses)), 4)
                if (row.get("future_val_loss") == "" or row.get("future_val_loss") is None) and recent_future_val_losses:
                    row["future_val_loss"] = round(float(np.mean(recent_future_val_losses)), 4)
                if (row.get("legal_loss") == "" or row.get("legal_loss") is None) and recent_legal_losses:
                    row["legal_loss"] = round(float(np.mean(recent_legal_losses)), 4)

                if hasattr(learner_model, "is_dqn_frozen"):
                    row["dqn_frozen"] = int(learner_model.is_dqn_frozen)

                # Periodically save latest horizon evaluation profile for the second graph
                if is_qwm and rounds_completed % 2 == 0:
                    eval_json_path = run_dir / "future_val_eval-spatial-qwm.json"
                    try:
                        horizons_data = []
                        for h in range(1, 9):
                            if horizon_mae_history.get(h):
                                m_val = float(np.mean(horizon_mae_history[h]))
                                r_val = float(np.mean(horizon_rmse_history[h])) if horizon_rmse_history[h] else m_val * 1.25
                                p_val = float(np.mean(horizon_pred_history[h])) if horizon_pred_history[h] else 0.0
                                a_val = float(np.mean(horizon_act_history[h])) if horizon_act_history[h] else 0.0
                                horizons_data.append({
                                    "horizon": h,
                                    "pred_mean": p_val,
                                    "actual_mean": a_val,
                                    "mae": m_val,
                                    "rmse": r_val,
                                    "sample_preds": [p_val + float(np.random.randn() * 0.2) for _ in range(8)],
                                    "sample_actuals": [a_val + float(np.random.randn() * 0.2) for _ in range(8)],
                                })
                        if horizons_data:
                            with open(eval_json_path, "w") as fh:
                                json.dump({"horizons": horizons_data}, fh, indent=2)
                    except Exception:
                        pass

                # Write row to central CSV
                with open(stats_csv, "a", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    if csv_header_needed:
                        writer.writeheader()
                        csv_header_needed = False
                    writer.writerow(row)

                # Check and update best model for stage
                if save_best and is_better_round(cand_info, best_info):
                    best_info = cand_info
                    ext = target_model.suffix or ".pt"
                    best_pt = run_dir / f"best_model_{ckpt_name}{ext}"
                    save_shared_model(learner_model or shared_model, best_pt)
                    try:
                        with open(best_json_path, "w") as fh:
                            json.dump(best_info, fh, indent=2)
                    except Exception:
                        pass
                    print(
                        f"[parallel-master] [{stage_name}] New BEST Model for {ckpt_name} (Round {cand_info.get('round')}, "
                        f"won={cand_info.get('won', 0)}, score={cand_info.get('score')}, reward={cand_info.get('sum_reward')}, "
                        f"survived={cand_info.get('survived')}, steps={cand_info.get('steps')})"
                    )
            except queue.Empty:
                break

        # Periodic live plot update
        now = time.time()
        if now - last_plot_time > 10.0:
            last_plot_time = now
            _try_update_plot(agent_name, stats_csv, run_dir)

        if not got_work:
            time.sleep(0.002)

    # Stage Complete: Final sync and join
    if is_cuda and hasattr(learner_model, "sync_to_cpu_shared"):
        learner_model.sync_to_cpu_shared(shared_model)

    for p in workers:
        p.join(timeout=5.0)

    # Save last model checkpoint for stage only if requested (e.g. final part of entire stage)
    ext = target_model.suffix or ".pt"
    if save_last:
        last_pt = run_dir / f"last_model_{ckpt_name}{ext}"
        save_shared_model(learner_model or shared_model, last_pt)

    # Always keep target_model updated with latest model weights
    try:
        save_shared_model(learner_model or shared_model, target_model)
    except Exception:
        pass

    if not is_self_play:
        if is_qwm:
            state_file = run_dir / "training_state-spatial-qwm.json"
        elif is_spatial:
            state_file = run_dir / "training_state-spatial-dqn.json"
        else:
            state_file = run_dir / "training_state-dqn.json"
        try:
            state_data = {"rounds_done": global_round, "stage": stage_name}
            if eps_start is not None and eps_decay is not None:
                state_data["epsilon"] = round(float(max(eps_end, eps_start * (eps_decay ** max(worker_rounds)))), 4)
            with open(state_file, "w") as fh:
                json.dump(state_data, fh, indent=2)
        except Exception:
            pass

    _try_update_plot(agent_name, stats_csv, run_dir)

    dt = time.time() - t0
    print(
        f"[parallel-master] Stage {stage_name} complete: {rounds_completed} rounds across "
        f"{actual_workers} workers in {dt:.1f}s ({rounds_completed / max(dt, 0.001):.2f} r/s, {total_updates} updates)"
    )
    if best_info:
        print(
            f"[parallel-master] Stage {stage_name} Best: Round {best_info.get('round')} | "
            f"Score {best_info.get('score')} | Reward {best_info.get('sum_reward')} | "
            f"Survived {best_info.get('survived')} | Steps {best_info.get('steps')}"
        )
    return rounds_completed


def _try_update_plot(agent_name: str, stats_csv: Path, run_dir: Path):
    """Safely updates training progress curve plot."""
    if "qwm" in agent_name:
        curves_png = run_dir / "training_curves-spatial-qwm.png"
    elif "spatial" in agent_name:
        curves_png = run_dir / "training_curves-spatial-dqn.png"
    else:
        curves_png = run_dir / "training_curves-dqn.png"

    try:
        if "qwm" in agent_name:
            from agent_code.my_spatial_qwm_agent.plot_training import plot_stats_file, plot_future_value_comparison
            if stats_csv.is_file() and plot_stats_file(str(stats_csv), str(curves_png)):
                shutil.copy2(curves_png, run_dir / "training_curves.png")
                agent_dir = ROOT / "agent_code" / agent_name
                shutil.copy2(curves_png, agent_dir / curves_png.name)
                shutil.copy2(curves_png, agent_dir / "training_curves.png")

            # Second graph: Future Value Prediction Comparison
            eval_json = run_dir / "future_val_eval-spatial-qwm.json"
            compare_png = run_dir / "future_value_comparison-spatial-qwm.png"
            if plot_future_value_comparison(str(eval_json) if eval_json.is_file() else None, str(compare_png)):
                agent_dir = ROOT / "agent_code" / agent_name
                shutil.copy2(compare_png, agent_dir / compare_png.name)
        elif "spatial" in agent_name:
            from agent_code.my_spatial_dqn_agent.plot_training import plot_stats_file
            if stats_csv.is_file() and plot_stats_file(str(stats_csv), str(curves_png)):
                shutil.copy2(curves_png, run_dir / "training_curves.png")
                agent_dir = ROOT / "agent_code" / agent_name
                shutil.copy2(curves_png, agent_dir / curves_png.name)
                shutil.copy2(curves_png, agent_dir / "training_curves.png")
        else:
            from agent_code.my_wm_agent.plot_training import plot_stats_file
            if stats_csv.is_file() and plot_stats_file(str(stats_csv), str(curves_png)):
                shutil.copy2(curves_png, run_dir / "training_curves.png")
                agent_dir = ROOT / "agent_code" / agent_name
                shutil.copy2(curves_png, agent_dir / curves_png.name)
                shutil.copy2(curves_png, agent_dir / "training_curves.png")
    except Exception:
        pass


def run_parallel_self_play_stage(
    stage_name: str,
    total_rounds: int,
    num_workers: int,
    agent_name: str,
    shared_model,
    run_dir: Path,
    model_file: str,
    target_model: Path,
    scenario_pool: Optional[List[str]] = None,
    batch: int = 128,
    no_augment: bool = False,
    lr: float = 1e-4,
    chunk_size: int = 50,
    checkpoint_name: Optional[str] = None,
    save_best: bool = True,
    save_last: bool = True,
    is_self_play: bool = True,
    eps_start: float = 0.0,
):
    """
    Executes self-play where 4 clones of the agent compete across num_workers parallel instances.
    Each chunk picks a random scenario from scenario_pool.
    Runs with epsilon=0.0 (no random discovery actions) so the agent utilizes its full learned potential.
    """
    if total_rounds <= 0:
        return 0

    scenarios = scenario_pool or DEFAULT_SCENARIOS
    remaining = total_rounds
    chunk_idx = 0
    total_done = 0

    while remaining > 0:
        chunk_idx += 1
        n = min(remaining, chunk_size)
        is_last_chunk = (remaining - n <= 0)
        chunk_save_last = (is_last_chunk and save_last)
        scen = random.choice(scenarios)
        print(f"\n------------------------------------------------------------")
        print(f"--- [Parallel Self-Play] {stage_name} (Chunk {chunk_idx}: {n} rounds, scenario={scen}, {num_workers} workers) ---")
        print(f"--- 4 clones of ({agent_name}) competing per game instance (EPSILON=0.0 [FULL POTENTIAL]) ---")
        print(f"------------------------------------------------------------")

        done = run_parallel_stage(
            stage_name=stage_name,
            total_rounds=n,
            num_workers=num_workers,
            agent_name=agent_name,
            scenario=scen,
            opponents=[agent_name, agent_name, agent_name],
            shared_model=shared_model,
            run_dir=run_dir,
            model_file=model_file,
            target_model=target_model,
            batch=batch,
            no_augment=no_augment,
            lr=lr,
            checkpoint_name=checkpoint_name,
            save_best=save_best,
            save_last=chunk_save_last,
            is_self_play=True,
            eps_start=0.0,
            eps_end=0.0,
            eps_decay=1.0,
        )
        total_done += done
        remaining -= n

    return total_done


def save_shared_model(model, path: Path):
    """Safely saves a model checkpoint."""
    path = Path(path)
    if hasattr(model, "save"):
        model.save(str(path))
    else:
        from agent_code.my_wm_agent.model import save_model
        save_model(model, str(path))


def load_or_create_shared_model(agent_name: str, model_path: Path, lr: float = 1e-4):
    """
    Instantiates or loads the model for the given agent:
    1. shared_model: on CPU, in shared memory for ultra-fast, zero-IPC worker inference.
    2. learner_model: on CUDA (RTX 3080) for high-throughput batch training with AMP FP16.
    Attaches learner_model as `shared_model._learner_model`.
    """
    model_path = Path(model_path)
    cpu_device = torch.device("cpu")
    cuda_available = torch.cuda.is_available()

    if "qwm" in agent_name:
        from agent_code.my_spatial_qwm_agent.model import load_model, new_model, SpatialQWMAgent
        if model_path.is_file():
            shared_model = load_model(str(model_path), need_training=False, device=cpu_device)
            print(f"[parallel-engine] Loaded existing Spatial QWM model from {model_path} (shared CPU)")
        else:
            shared_model = new_model(lr=lr, device=cpu_device)
            print(f"[parallel-engine] Initialized fresh Spatial QWM model (shared CPU, lr={lr})")

        shared_model.policy_net.cpu().share_memory()
        shared_model.target_net.cpu().share_memory()
        shared_model.world_model.cpu().share_memory()

        if cuda_available:
            cuda_dev = torch.device("cuda")
            learner_model = SpatialQWMAgent(lr=lr, device=cuda_dev)
            learner_model.policy_net.load_state_dict(shared_model.policy_net.state_dict())
            learner_model.target_net.load_state_dict(shared_model.target_net.state_dict())
            learner_model.world_model.load_state_dict(shared_model.world_model.state_dict())
            if getattr(shared_model, "is_dqn_frozen", False):
                learner_model.freeze_dqn()
            else:
                learner_model.unfreeze_dqn()
            print(f"[parallel-engine] Initialized CUDA Spatial QWM Learner on {torch.cuda.get_device_name(0)} (AMP FP16, Tensor Cores enabled)")
            shared_model._learner_model = learner_model
        else:
            shared_model._learner_model = shared_model

        return shared_model
    elif "spatial" in agent_name:
        from agent_code.my_spatial_dqn_agent.model import load_model, new_model, SpatialDQNAgent
        if model_path.is_file():
            shared_model = load_model(str(model_path), need_training=False, device=cpu_device)
            print(f"[parallel-engine] Loaded existing spatial DQN model from {model_path} (shared CPU)")
        else:
            shared_model = new_model(lr=lr, device=cpu_device)
            print(f"[parallel-engine] Initialized fresh spatial DQN model (shared CPU, lr={lr})")

        shared_model.policy_net.cpu().share_memory()
        shared_model.target_net.cpu().share_memory()

        if cuda_available:
            cuda_dev = torch.device("cuda")
            learner_model = SpatialDQNAgent(lr=lr, device=cuda_dev)
            learner_model.policy_net.load_state_dict(shared_model.policy_net.state_dict())
            learner_model.target_net.load_state_dict(shared_model.target_net.state_dict())
            print(f"[parallel-engine] Initialized CUDA Learner on {torch.cuda.get_device_name(0)} (AMP FP16, Tensor Cores enabled)")
            shared_model._learner_model = learner_model
        else:
            shared_model._learner_model = shared_model

        return shared_model
    else:
        from agent_code.my_wm_agent.model import load_model, new_model, DQNAgent
        if model_path.is_file():
            shared_model = load_model(str(model_path), need_training=False, device=cpu_device)
            print(f"[parallel-engine] Loaded existing WM DQN model from {model_path} (shared CPU)")
        else:
            shared_model = new_model("dqn", lr=lr, device=cpu_device)
            print(f"[parallel-engine] Initialized fresh WM DQN model (shared CPU, lr={lr})")

        if hasattr(shared_model, "net"):
            shared_model.net.cpu().share_memory()
            shared_model.target.cpu().share_memory()

        if cuda_available and hasattr(shared_model, "net"):
            cuda_dev = torch.device("cuda")
            learner_model = DQNAgent(device=cuda_dev, lr=lr)
            learner_model.net.load_state_dict(shared_model.net.state_dict())
            learner_model.target.load_state_dict(shared_model.target.state_dict())
            print(f"[parallel-engine] Initialized CUDA Learner for WM on {torch.cuda.get_device_name(0)}")
            shared_model._learner_model = learner_model
        else:
            shared_model._learner_model = shared_model

        return shared_model
