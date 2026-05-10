"""Sequence FDM V2 model: predicts absolute trajectories + binary risk from terrain grid."""

from __future__ import annotations

import torch
from torch import nn


FEATURE_NAMES_V2 = [
    "state_x", "state_y", "state_theta",
    "state_vx", "state_vy", "state_wz",
] + [
    f"cmd_t{i}_vx" for i in range(1, 100)  # dynamically sized by horizon
] + [
    f"cmd_t{i}_vy" for i in range(1, 100)
] + [
    f"cmd_t{i}_wz" for i in range(1, 100)
] + [
    f"terrain_grid_{i}" for i in range(81)
]

TARGET_NAMES_V2 = [
    f"state_x_t{i}" for i in range(1, 100)
] + [
    f"state_y_t{i}" for i in range(1, 100)
] + [
    f"state_theta_t{i}" for i in range(1, 100)
] + [
    f"state_vx_t{i}" for i in range(1, 100)
] + [
    f"state_vy_t{i}" for i in range(1, 100)
] + [
    f"state_wz_t{i}" for i in range(1, 100)
] + [
    f"risk_t{i}" for i in range(1, 100)
]


def build_feature_names_v2(horizon_steps: int) -> list[str]:
    names = ["state_x", "state_y", "state_theta", "state_vx", "state_vy", "state_wz"]
    for step in range(horizon_steps):
        names.extend([f"future_cmd_t{step + 1}_vx", f"future_cmd_t{step + 1}_vy", f"future_cmd_t{step + 1}_wz"])
    names.extend([f"terrain_grid_{i}" for i in range(81)])
    return names


def build_target_names_v2(horizon_steps: int) -> list[str]:
    names: list[str] = []
    for axis in ["x", "y", "theta", "vx", "vy", "wz"]:
        for step in range(horizon_steps):
            names.append(f"state_{axis}_t{step + 1}")
    for step in range(horizon_steps):
        names.append(f"risk_t{step + 1}")
    return names


class SequenceFdmMlpV2(nn.Module):
    """MLP that maps (state, control sequence, terrain grid) -> (state sequence, risk logits)."""

    def __init__(self, horizon_steps: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        self.horizon_steps = int(horizon_steps)
        input_dim = 6 + 3 * self.horizon_steps + 81
        output_dim = 6 * self.horizon_steps + self.horizon_steps

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        controls: torch.Tensor,
        terrain_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, 6)
            controls: (B, H, 3)
            terrain_grid: (B, 81)
        Returns:
            states_pred: (B, H, 6)
            risk_logits: (B, H)
        """
        batch_size = state.shape[0]
        controls_flat = controls.view(batch_size, -1)
        x = torch.cat([state, controls_flat, terrain_grid], dim=1)
        out = self.net(x)
        states_pred = out[:, : 6 * self.horizon_steps].view(batch_size, self.horizon_steps, 6)
        risk_logits = out[:, 6 * self.horizon_steps :]
        return states_pred, risk_logits
