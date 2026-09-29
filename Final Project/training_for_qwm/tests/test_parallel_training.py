#!/usr/bin/env python3
"""
test_parallel_training.py -- Test suite verifying multi-worker parallel training,
shared memory synchronization, and gradient accumulation.
"""

import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parallel_train import (
    is_better_round,
    load_or_create_shared_model,
    save_shared_model,
)
from agent_code.my_spatial_dqn_agent.model import (
    BOARD_H,
    BOARD_W,
    N_FEATURES,
    N_SPATIAL_CHANNELS,
    SpatialDQNAgent,
)
from agent_code.my_wm_agent.model import DQNAgent


class TestParallelTraining(unittest.TestCase):

    def test_shared_memory_spatial_dqn(self):
        """Verify SpatialDQNAgent weights are marked for shared memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = load_or_create_shared_model("my_spatial_dqn_agent", model_path, lr=1e-4)
            self.assertIsInstance(model, SpatialDQNAgent)

            # Check that parameters are in shared memory
            for p in model.policy_net.parameters():
                self.assertTrue(p.is_shared(), "Policy net parameter is not shared")
            for p in model.target_net.parameters():
                self.assertTrue(p.is_shared(), "Target net parameter is not shared")

    def test_shared_memory_wm_dqn(self):
        """Verify my_wm_agent DQNAgent weights are marked for shared memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_wm_model.pt"
            model = load_or_create_shared_model("my_wm_agent", model_path, lr=1e-4)
            self.assertIsInstance(model, DQNAgent)

            for p in model.net.parameters():
                self.assertTrue(p.is_shared(), "WM DQN parameter is not shared")
            for p in model.target.parameters():
                self.assertTrue(p.is_shared(), "WM DQN target parameter is not shared")

    def test_spatial_dqn_gradient_computation_and_accumulation(self):
        """Verify compute_gradients returns detached tensors and apply_accumulated_gradients updates weights."""
        model = SpatialDQNAgent(lr=1e-3)
        initial_weight = next(model.policy_net.parameters()).clone()

        B = 4
        S = np.random.randn(B, N_SPATIAL_CHANNELS, BOARD_W, BOARD_H).astype(np.float32)
        F = np.random.randn(B, N_FEATURES).astype(np.float32)
        A = np.random.randint(0, 6, size=B).astype(np.int64)
        R = np.random.randn(B).astype(np.float32)
        NS = np.random.randn(B, N_SPATIAL_CHANNELS, BOARD_W, BOARD_H).astype(np.float32)
        NF = np.random.randn(B, N_FEATURES).astype(np.float32)
        D = np.zeros(B, dtype=bool)

        td, grads = model.compute_gradients(S, F, A, R, NS, NF, D)
        self.assertIsInstance(td, float)
        self.assertGreater(len(grads), 0)
        for g in grads:
            self.assertIsInstance(g, torch.Tensor)
            self.assertFalse(g.requires_grad)

        # Apply accumulated gradients
        model.apply_accumulated_gradients(grads, scale=1.0)
        updated_weight = next(model.policy_net.parameters())
        self.assertFalse(torch.equal(initial_weight, updated_weight), "Weights were not updated")

    def test_wm_dqn_gradient_computation_and_accumulation(self):
        """Verify DQNAgent compute_gradients and apply_accumulated_gradients."""
        model = DQNAgent(n_features=N_FEATURES, n_actions=6, lr=1e-3)
        initial_weight = next(model.net.parameters()).clone()

        B = 4
        F = np.random.randn(B, N_FEATURES).astype(np.float32)
        A = np.random.randint(0, 6, size=B).astype(np.int64)
        R = np.random.randn(B).astype(np.float32)
        NF = np.random.randn(B, N_FEATURES).astype(np.float32)
        D = np.zeros(B, dtype=bool)

        td, grads = model.compute_gradients(F, A, R, NF, D)
        self.assertIsInstance(td, float)
        self.assertGreater(len(grads), 0)

        model.apply_accumulated_gradients(grads, scale=1.0)
        updated_weight = next(model.net.parameters())
        self.assertFalse(torch.equal(initial_weight, updated_weight), "Weights were not updated")

    def test_is_better_round_logic(self):
        """Test the multi-criteria candidate comparison function."""
        # 1. Higher reward wins
        b1 = {"score": 5, "sum_reward": 10.0, "survived": 1, "steps": 100}
        c1 = {"score": 5, "sum_reward": 15.0, "survived": 1, "steps": 100}
        self.assertTrue(is_better_round(c1, b1))
        self.assertFalse(is_better_round(b1, c1))

        # 2. Survival preference
        b2 = {"score": 5, "sum_reward": 10.0, "survived": 0, "steps": 150}
        c2 = {"score": 5, "sum_reward": 10.0, "survived": 1, "steps": 100}
        self.assertTrue(is_better_round(c2, b2))

        # 3. Tie breaker when survived: lowest step count
        b3 = {"score": 5, "sum_reward": 10.0, "survived": 1, "steps": 200}
        c3 = {"score": 5, "sum_reward": 10.0, "survived": 1, "steps": 120}
        self.assertTrue(is_better_round(c3, b3))

        # 4. Neither survived: highest step count (longer survival)
        b4 = {"score": 5, "sum_reward": 10.0, "survived": 0, "steps": 30}
        c4 = {"score": 5, "sum_reward": 10.0, "survived": 0, "steps": 80}
        self.assertTrue(is_better_round(c4, b4))


if __name__ == "__main__":
    unittest.main()
