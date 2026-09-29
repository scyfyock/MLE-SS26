#!/usr/bin/env python3
"""
train_small_model_phase1.py -- Phase 1 Supervised Training for the Small Path Reward Model.

Location: agent_code/my_spatial_qwm_agent/

1. Collects candidate path features (56 dims) and ground-truth heuristic penalty targets
   (using the Optuna champion coefficients: wait=1.25, loop=1.5) by playing matches
   in the BombeRLe environment with the frozen base model.
2. Trains SmallPathRewardModel (using dynActivation) via supervised regression.
3. Evaluates validation loss, MAE, and R^2 score across epochs.
4. Logs training statistics to training_stats-small-model.csv, plots training_curves-small-model.png,
   and saves checkpoints to a dedicated run folder and to agent_code/my_spatial_qwm_agent/.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Iterable, List, Optional, Tuple

CSV_FIELDS = [
    "phase", "epoch", "round",
    "train_loss", "val_loss", "val_mae", "val_r2",
    "mean_reward", "sum_reward", "score", "coins", "crates",
    "kills", "suicides", "survived", "won", "steps", "wall_time_s"
]

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Root directory resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

AGENT_DIR = Path(__file__).resolve().parent

from agent_code.my_spatial_qwm_agent.path_penalty_model import (
    FEATURE_DIM,
    SmallPathRewardModel,
    load_small_model,
    save_small_model,
)
from agent_code.my_spatial_qwm_agent.plot_small_model import plot_small_model_stats


def create_run_folder(base_runs_dir: Optional[str] = None) -> str:
    """Creates a timestamped run folder matching the big model's convention."""
    runs_dir = base_runs_dir or os.path.join(str(AGENT_DIR), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(runs_dir, f"run_{timestamp}_small_model")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_last_completed_epoch(csv_path: str) -> int:
    """Reads the CSV and returns the maximum integer epoch found in Phase 1 rows."""
    if not os.path.isfile(csv_path):
        return 0
    max_epoch = 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("phase", "").strip() == "phase1":
                    e_val = row.get("epoch", "")
                    if e_val:
                        try:
                            e_int = int(float(e_val))
                            if e_int > max_epoch:
                                max_epoch = e_int
                        except (ValueError, TypeError):
                            pass
    except Exception as err:
        print(f"Warning: Could not parse epoch from {csv_path}: {err}")
    return max_epoch


from agent_code.my_spatial_qwm_agent.train_small_model_phase2 import (
    ALL_CURRICULUM_STAGES,
    STAGE_ALIASES,
    TARGET_ENVIRONMENTS,
    resolve_stages,
)


def _run_collection_worker(
    task_id: int,
    env_key: str,
    rounds: int,
    worker_file: str,
    seed: int,
) -> Tuple[int, str, int, int]:
    """Executes a single game batch in an isolated subprocess to collect path data."""
    cfg = TARGET_ENVIRONMENTS[env_key]
    scen = cfg["scenario"]
    opponents = cfg["opponents"]
    agents = ["my_spatial_qwm_agent"] + opponents

    env = os.environ.copy()
    env["MY_QWM_COLLECT_PATH_DATA"] = "1"
    env["MY_QWM_COLLECT_PATH_DATA_FILE"] = os.path.abspath(worker_file)
    env["MY_QWM_USE_SMALL_MODEL"] = "0"  # Use Optuna champion heuristics to generate target values
    env["MY_QWM_TREE_SEARCH"] = "1"
    env["MY_QWM_SEARCH_DEPTH"] = "5"
    env["MY_QWM_BEAM_SIZE"] = "18"
    env["MY_QWM_TREE_DISCOUNT"] = "0.08"
    env["MY_QWM_ALPHA_VQ"] = "0.16"
    env["MY_QWM_WAIT_PENALTY"] = "1.25"
    env["MY_QWM_LOOP_PENALTY"] = "1.5"

    cmd = [
        sys.executable,
        str(ROOT_DIR / "main.py"),
        "play",
        "--agents", *agents,
        "--n-rounds", str(rounds),
        "--scenario", scen,
        "--seed", str(seed),
        "--no-gui",
        "--silence-errors",
    ]
    res = subprocess.run(cmd, env=env, cwd=str(ROOT_DIR), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Warning: Worker {task_id} ({env_key}) exited with code {res.returncode}")
        if res.stderr:
            print(f"Worker {task_id} stderr tail: {res.stderr[-200:]}")

    step_count = 0
    if os.path.isfile(worker_file):
        try:
            with np.load(worker_file) as data:
                step_count = data['x'].shape[0]
        except Exception:
            pass
    return task_id, env_key, rounds, step_count


def collect_dataset(
    output_npz: str,
    n_rounds: int = 40,
    environments: Optional[Iterable[str]] = None,
    n_workers: Optional[int] = None,
    base_seed: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collects candidate path features and heuristic targets via parallel multiprocessing workers.

    Distributes game rounds across worker tasks running simultaneously in isolated subprocesses,
    each generating samples to an independent chunk file, then safely merges them into output_npz.
    """
    abs_output_npz = os.path.abspath(output_npz)
    if os.path.isfile(abs_output_npz):
        os.remove(abs_output_npz)

    # Resolve environment keys and aliases
    resolved_envs = resolve_stages(list(environments) if environments else None)

    # Determine optimal number of parallel workers
    available_cpus = os.cpu_count() or 4
    if n_workers is None or n_workers <= 0:
        n_workers = min(available_cpus, 8)

    # Don't spin up more workers than requested rounds
    n_workers = max(1, min(n_workers, n_rounds))

    print(f"\n[Phase 1 Data Collection] Multiprocessing enabled across {n_workers} parallel workers (Available CPUs: {available_cpus}).")
    print(f"  -> Total target rounds: {n_rounds} across {len(resolved_envs)} environments: {resolved_envs}")

    # Partition rounds among environments and worker chunks
    base_rounds_per_env = n_rounds // len(resolved_envs)
    rem_rounds = n_rounds % len(resolved_envs)
    tasks = []
    task_id = 0

    for env_idx, env_key in enumerate(resolved_envs):
        chunk_target = base_rounds_per_env + (1 if env_idx < rem_rounds else 0)
        chunk_target = max(1, chunk_target)
        workers_for_this_env = max(1, n_workers // len(resolved_envs))
        base_chunk = chunk_target // workers_for_this_env
        remainder = chunk_target % workers_for_this_env

        for w_idx in range(workers_for_this_env):
            chunk_rounds = base_chunk + (1 if w_idx < remainder else 0)
            if chunk_rounds <= 0:
                continue
            worker_file = f"{abs_output_npz}.worker_{task_id}.npz"
            if os.path.isfile(worker_file):
                os.remove(worker_file)
            seed = base_seed + task_id * 97
            tasks.append((task_id, env_key, chunk_rounds, worker_file, seed))
            task_id += 1

    print(f"  -> Dispatched {len(tasks)} parallel worker tasks.")
    start_time = time.time()
    worker_files = [t[3] for t in tasks]

    # Execute workers concurrently
    completed_rounds = 0
    total_steps_collected = 0

    with ThreadPoolExecutor(max_workers=min(n_workers, len(tasks))) as executor:
        futures = {
            executor.submit(_run_collection_worker, t_id, e_key, r_count, w_file, s_val): t_id
            for (t_id, e_key, r_count, w_file, s_val) in tasks
        }
        for future in as_completed(futures):
            t_id, e_key, r_count, s_count = future.result()
            completed_rounds += r_count
            total_steps_collected += s_count
            elapsed = time.time() - start_time
            print(f"  [Worker {t_id:>2d}] Finished {r_count:>2d} rounds in '{e_key}' ({s_count:>5d} steps, {s_count*6:>6d} paths) | Progress: {completed_rounds}/{n_rounds} rds ({elapsed:.1f}s)")

    # Merge all worker npz files
    print(f"\n[Phase 1 Data Collection] Merging data chunks from {len(worker_files)} workers...")
    all_x, all_y = [], []
    for w_file in worker_files:
        if os.path.isfile(w_file):
            try:
                with np.load(w_file) as data:
                    all_x.append(data['x'])
                    all_y.append(data['y'])
                os.remove(w_file)  # Clean up temp worker file
            except Exception as err:
                print(f"Warning: Failed to load worker file {w_file}: {err}")

    if not all_x:
        raise RuntimeError(f"Data collection failed: No worker produced samples for {abs_output_npz}.")

    merged_x = np.concatenate(all_x, axis=0)
    merged_y = np.concatenate(all_y, axis=0)

    os.makedirs(os.path.dirname(abs_output_npz), exist_ok=True)
    np.savez_compressed(abs_output_npz, x=merged_x, y=merged_y)

    total_elapsed = time.time() - start_time
    total_paths = merged_x.shape[0] * 6
    print(f"[Phase 1 Data Collection Complete] Gathered {merged_x.shape[0]} game steps ({total_paths} candidate paths) in {total_elapsed:.1f}s!")
    print(f"  -> Saved unified dataset to: {abs_output_npz}")
    return merged_x, merged_y


def train_phase1(
    data_file: str,
    output_model_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_split: float = 0.15,
    init_model_path: Optional[str] = None,
    continue_training: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Trains SmallPathRewardModel on collected dataset and logs to run directory."""
    if run_dir is None:
        run_dir = create_run_folder()
    print(f"\n[Phase 1 Training] Output Run Directory: {run_dir}")

    stats_csv_run = os.path.join(run_dir, "training_stats-small-model.csv")
    state_json_run = os.path.join(run_dir, "training_state-small-model.json")
    plot_png_run = os.path.join(run_dir, "training_curves-small-model.png")
    phase1_ckpt_run = os.path.join(run_dir, "small_model_phase1.pt")
    default_ckpt_run = os.path.join(run_dir, "small_model.pt")

    active_phase1_ckpt = output_model_path or os.path.join(str(AGENT_DIR), "small_model_phase1.pt")
    active_default_ckpt = os.path.join(str(AGENT_DIR), "small_model.pt")
    active_stats_csv = os.path.join(str(AGENT_DIR), "training_stats-small-model.csv")
    active_state_json = os.path.join(str(AGENT_DIR), "training_state-small-model.json")
    active_plot_png = os.path.join(str(AGENT_DIR), "training_curves-small-model.png")

    if continue_training:
        if not os.path.isfile(stats_csv_run) and os.path.isfile(active_stats_csv):
            shutil.copyfile(active_stats_csv, stats_csv_run)
            print(f"[Phase 1 Continuation] Copied active CSV telemetry from {active_stats_csv}")
        if not os.path.isfile(state_json_run) and os.path.isfile(active_state_json):
            shutil.copyfile(active_state_json, state_json_run)

    # Initialize CSV header if not exists
    if not os.path.isfile(stats_csv_run):
        with open(stats_csv_run, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

    print(f"[Phase 1 Training] Loading dataset from {data_file}...")
    with np.load(data_file) as data:
        x_raw, y_raw = data['x'], data['y']

    N = x_raw.shape[0]
    x_flat = x_raw.reshape(-1, FEATURE_DIM).astype(np.float32)
    y_flat = y_raw.reshape(-1, 1).astype(np.float32)

    total_samples = x_flat.shape[0]
    indices = np.random.permutation(total_samples)
    val_size = int(total_samples * val_split)
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    x_train, y_train = torch.from_numpy(x_flat[train_idx]), torch.from_numpy(y_flat[train_idx])
    x_val, y_val = torch.from_numpy(x_flat[val_idx]), torch.from_numpy(y_flat[val_idx])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    print(f"[Phase 1 Training] Total samples: {total_samples} (Train: {len(train_idx)}, Val: {len(val_idx)})")
    print(f"[Phase 1 Training] Target range: min={y_flat.min():.2f}, max={y_flat.max():.2f}, mean={y_flat.mean():.2f}, std={y_flat.std():.2f}")

    # Model initialization (load checkpoint if continuing, else fresh model)
    init_path = init_model_path
    if not init_path or not os.path.isfile(init_path):
        if continue_training:
            candidates = [default_ckpt_run, active_default_ckpt, phase1_ckpt_run, active_phase1_ckpt]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    init_path = candidate
                    break

    if init_path and os.path.isfile(init_path):
        print(f"[Phase 1] Loading existing model weights from: {init_path}")
        model = load_small_model(init_path, device=device)
    else:
        model = SmallPathRewardModel(input_dim=FEATURE_DIM, hidden_dim=64).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_weights = None
    best_mae = float("inf")
    best_r2 = -float("inf")

    epoch_offset = get_last_completed_epoch(stats_csv_run) if continue_training else 0
    if continue_training and epoch_offset > 0:
        print(f"[Phase 1 Continuation] Resuming at epoch {epoch_offset + 1} (previously completed: {epoch_offset} epochs)")

    print(f"[Phase 1 Training] Architecture: {model.net}")
    print(f"[Phase 1 Training] Total trainable parameters: {sum(p.numel() for p in model.parameters())}")
    print("-" * 75)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val MAE':>10} | {'Val R^2':>10}")
    print("-" * 75)

    start_time = time.time()
    for epoch_idx in range(1, epochs + 1):
        epoch = epoch_offset + epoch_idx
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        val_preds_list, val_targets_list = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                loss = criterion(preds, by)
                val_losses.append(loss.item())
                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(by.cpu().numpy())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_preds = np.concatenate(val_preds_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)

        mae = float(np.mean(np.abs(val_preds - val_targets)))
        ss_res = float(np.sum((val_targets - val_preds) ** 2))
        ss_tot = float(np.sum((val_targets - np.mean(val_targets)) ** 2))
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-6)))
        elapsed = time.time() - start_time

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mae = mae
            best_r2 = r2
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"

        if epoch_idx % 5 == 0 or epoch_idx == 1 or epoch_idx == epochs or marker:
            print(f"{epoch:>6d} | {train_loss:>12.4f} | {val_loss:>12.4f} | {mae:>10.4f} | {r2:>10.4f}{marker}")

        # Append epoch telemetry to CSV
        with open(stats_csv_run, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerow({
                "phase": "phase1",
                "epoch": epoch,
                "round": "",
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_mae": f"{mae:.4f}",
                "val_r2": f"{r2:.4f}",
                "mean_reward": "",
                "sum_reward": "",
                "score": "",
                "coins": "",
                "crates": "",
                "kills": "",
                "suicides": "",
                "survived": "",
                "won": "",
                "steps": "",
                "wall_time_s": f"{elapsed:.1f}",
            })

    print("-" * 75)
    print(f"[Phase 1 Training] Best Validation Loss: {best_val_loss:.4f} (MAE: {best_mae:.4f}, R^2: {best_r2:.4f})")

    # Restore best weights and save checkpoints
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save to run directory
    save_small_model(model, phase1_ckpt_run)
    save_small_model(model, default_ckpt_run)

    # Copy to agent directory
    shutil.copyfile(phase1_ckpt_run, active_phase1_ckpt)
    shutil.copyfile(default_ckpt_run, active_default_ckpt)
    shutil.copyfile(stats_csv_run, active_stats_csv)

    # Plot curves
    plot_small_model_stats(stats_csv_run, plot_png_run, window=10)
    if os.path.isfile(plot_png_run):
        shutil.copyfile(plot_png_run, active_plot_png)

    # Save state JSON
    state_data = {
        "run_dir": os.path.abspath(run_dir),
        "phase": "phase1",
        "epochs": epoch_offset + epochs,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_mae,
        "best_val_r2": best_r2,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_architecture": str(model.net),
        "timestamp": datetime.now().isoformat(),
    }
    with open(state_json_run, "w") as f:
        json.dump(state_data, f, indent=2)
    shutil.copyfile(state_json_run, active_state_json)

    print(f"[Phase 1 Training] Run artifacts saved in: {run_dir}")
    print(f"[Phase 1 Training] Active model updated in: {active_default_ckpt}")
    print(f"[Phase 1 Training] Plot generated: {plot_png_run}")
    return model, run_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Supervised Training for SmallPathRewardModel in my_spatial_qwm_agent")
    parser.add_argument("--collect-rounds", type=int, default=27, help="Number of rounds to play for dataset collection")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data-file", type=str, default="path_data_phase1.npz", help="Path to save/load collected samples")
    parser.add_argument("--output", type=str, default="small_model_phase1.pt", help="Path to save trained small model")
    parser.add_argument("--init-model", type=str, default=None, help="Initial model to continue training from")
    parser.add_argument("--continue", "-c", dest="continue_training", action="store_true", default=False,
                        help="Continue training from existing small_model.pt and CSV stats")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific run directory")
    parser.add_argument("--stages", "--environments", nargs="+", default=None,
                        help="Environments to collect data from (default: all curriculum stages. Supports: all, curriculum, combat, loot, coins, or specific stages: s0..s7)")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of parallel workers for data collection (default: auto, up to 8)")
    parser.add_argument("--skip-collect", action="store_true", help="Skip data collection if dataset already exists")
    parser.add_argument("--recollect", action="store_true", help="Force fresh data collection even if dataset already exists")
    args = parser.parse_args()

    data_path = os.path.join(str(AGENT_DIR), args.data_file)
    output_path = os.path.join(str(AGENT_DIR), args.output)

    if args.recollect and os.path.isfile(data_path):
        os.remove(data_path)

    if not args.skip_collect and (args.recollect or not os.path.isfile(data_path)):
        collect_dataset(data_path, n_rounds=args.collect_rounds, environments=args.stages, n_workers=args.workers)

    if args.init_model and not args.continue_training:
        args.continue_training = True

    train_phase1(
        data_path,
        output_model_path=output_path,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        init_model_path=args.init_model,
        continue_training=args.continue_training,
    )


if __name__ == "__main__":
    main()
