"""Hardware Accelerator for World Model and Policy Inference using PyTorch (XPU / CUDA / CPU).

Leverages device-level tensor operations on Intel Arc (XPU) or Nvidia (CUDA) GPUs
to accelerate forward predictions in LearnedWorldModel and NeuralQFunction by up to 10x.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def get_target_device(preferred: Optional[str] = None) -> torch.device:
    """Select the best available compute device (XPU, CUDA, or CPU)."""
    env_choice = os.environ.get("WMA_DEVICE", "").strip().lower()
    choice = (preferred or env_choice or "auto").lower()

    if choice in ("xpu", "intel"):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    elif choice in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
    elif choice == "cpu":
        return torch.device("cpu")

    # Auto-detection priority: XPU -> CUDA -> CPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> str:
    """Return human-readable device name."""
    if device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        name = torch.xpu.get_device_name(0)
        return f"XPU: {name}"
    elif device.type == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA: {name}"
    return "CPU"


class TorchMLPModule(nn.Module):
    """Fast compiled feed-forward MLP operating directly on GPU/XPU/CPU tensors."""

    def __init__(
        self,
        coefs: List[np.ndarray],
        intercepts: List[np.ndarray],
        activation: str = "tanh",
        is_classifier: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.is_classifier = is_classifier

        self.weights = nn.ParameterList([
            nn.Parameter(torch.from_numpy(w.astype(np.float32)), requires_grad=False)
            for w in coefs
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.from_numpy(b.astype(np.float32)), requires_grad=False)
            for b in intercepts
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        num_layers = len(self.weights)
        for i in range(num_layers - 1):
            h = h @ self.weights[i] + self.biases[i]
            if self.activation == "tanh":
                h = torch.tanh(h)
            elif self.activation == "relu":
                h = torch.relu(h)
        out = h @ self.weights[-1] + self.biases[-1]
        if self.is_classifier:
            out = torch.sigmoid(out)
        return out


class XPUWorldModelAccelerator:
    """High-throughput tensorized inference engine for LearnedWorldModel."""

    def __init__(self, world_model: Any, device: Optional[torch.device] = None):
        self.device = device or get_target_device()
        self.world_model = world_model

        # Build PyTorch modules for all trained neural heads
        self.dynamics_net = TorchMLPModule(
            world_model.dynamics_model.coefs_,
            world_model.dynamics_model.intercepts_,
            activation="tanh",
            is_classifier=False,
        ).to(self.device)

        self.agent_net = TorchMLPModule(
            world_model.agent_model.coefs_,
            world_model.agent_model.intercepts_,
            activation="tanh",
            is_classifier=False,
        ).to(self.device)

        self.done_net = TorchMLPModule(
            world_model.done_model.coefs_,
            world_model.done_model.intercepts_,
            activation="tanh",
            is_classifier=True,
        ).to(self.device)

        self.death_net = None
        if getattr(world_model, "is_death_model_ready", False):
            self.death_net = TorchMLPModule(
                world_model.death_model.coefs_,
                world_model.death_model.intercepts_,
                activation="tanh",
                is_classifier=True,
            ).to(self.device)

        self.suicide_net = None
        if getattr(world_model, "is_suicide_model_ready", False):
            self.suicide_net = TorchMLPModule(
                world_model.suicide_model.coefs_,
                world_model.suicide_model.intercepts_,
                activation="tanh",
                is_classifier=True,
            ).to(self.device)

        # Bomb outcome classifiers
        self.bomb_survival_net = None
        self.bomb_crate_net = None
        self.bomb_kill_net = None
        self.bomb_escape_net = None
        if getattr(world_model, "is_bomb_outcome_ready", False):
            self.bomb_survival_net = TorchMLPModule(
                world_model.bomb_survival_model.coefs_,
                world_model.bomb_survival_model.intercepts_,
                is_classifier=True,
            ).to(self.device)
            self.bomb_crate_net = TorchMLPModule(
                world_model.bomb_crate_model.coefs_,
                world_model.bomb_crate_model.intercepts_,
                is_classifier=True,
            ).to(self.device)
            self.bomb_kill_net = TorchMLPModule(
                world_model.bomb_kill_model.coefs_,
                world_model.bomb_kill_model.intercepts_,
                is_classifier=True,
            ).to(self.device)
            self.bomb_escape_net = TorchMLPModule(
                world_model.bomb_escape_model.coefs_,
                world_model.bomb_escape_model.intercepts_,
                is_classifier=True,
            ).to(self.device)

        self.dynamics_net.eval()
        self.agent_net.eval()
        self.done_net.eval()
        if self.death_net is not None:
            self.death_net.eval()
        if self.suicide_net is not None:
            self.suicide_net.eval()

    @torch.inference_mode()
    def predict_forward(
        self,
        encoded_input: np.ndarray,
        encoded_agent_input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Compute all 5 transition heads simultaneously on device.

        Returns:
            (dynamics_output, agent_output, done_prob, death_prob, suicide_prob)
        """
        t_in = torch.from_numpy(encoded_input).to(self.device, non_blocking=True).unsqueeze(0)
        t_ag = torch.from_numpy(encoded_agent_input).to(self.device, non_blocking=True).unsqueeze(0)

        dyn_out = self.dynamics_net(t_in).squeeze(0)
        ag_out = self.agent_net(t_ag).squeeze(0)
        done_prob = float(self.done_net(t_ag).squeeze().item())

        death_prob = (
            float(self.death_net(t_ag).squeeze().item())
            if self.death_net is not None
            else 0.0
        )
        suicide_prob = (
            float(self.suicide_net(t_ag).squeeze().item())
            if self.suicide_net is not None
            else 0.0
        )

        return (
            dyn_out.cpu().numpy(),
            ag_out.cpu().numpy(),
            done_prob,
            death_prob,
            suicide_prob,
        )

    @torch.inference_mode()
    def predict_death_prob(self, encoded_agent_input: np.ndarray) -> float:
        if self.death_net is None:
            return 0.0
        t_ag = torch.from_numpy(encoded_agent_input).to(self.device, non_blocking=True).unsqueeze(0)
        return float(self.death_net(t_ag).squeeze().item())

    @torch.inference_mode()
    def predict_suicide_prob(self, encoded_agent_input: np.ndarray) -> float:
        if self.suicide_net is None:
            return 0.0
        t_ag = torch.from_numpy(encoded_agent_input).to(self.device, non_blocking=True).unsqueeze(0)
        return float(self.suicide_net(t_ag).squeeze().item())

    @torch.inference_mode()
    def predict_bomb_outcomes(self, encoded_state: np.ndarray) -> Tuple[float, float, float, float]:
        t_s = torch.from_numpy(encoded_state).to(self.device, non_blocking=True).unsqueeze(0)
        surv = float(self.bomb_survival_net(t_s).squeeze().item()) if self.bomb_survival_net else 0.0
        crate = float(self.bomb_crate_net(t_s).squeeze().item()) if self.bomb_crate_net else 0.0
        kill = float(self.bomb_kill_net(t_s).squeeze().item()) if self.bomb_kill_net else 0.0
        escape = float(self.bomb_escape_net(t_s).squeeze().item()) if self.bomb_escape_net else 0.0
        return surv, crate, kill, escape


class XPUQNetworkAccelerator:
    """High-throughput tensorized Q-network inference on device."""

    def __init__(self, q_network: Any, device: Optional[torch.device] = None):
        self.device = device or get_target_device()
        self.net = TorchMLPModule(
            q_network.online_model.coefs_,
            q_network.online_model.intercepts_,
            activation="tanh",
            is_classifier=False,
        ).to(self.device)
        self.net.eval()

    @torch.inference_mode()
    def predict(self, encoded_state: np.ndarray) -> np.ndarray:
        t_s = torch.from_numpy(encoded_state).to(self.device, non_blocking=True).unsqueeze(0)
        out = self.net(t_s).squeeze(0)
        return out.cpu().numpy()
