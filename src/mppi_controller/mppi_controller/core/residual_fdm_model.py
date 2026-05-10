"""Shared residual FDM model definition and feature helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = [
    "state_x",
    "state_y",
    "state_theta",
    "state_vx",
    "state_vy",
    "state_wz",
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "terrain_slope_f",
    "terrain_slope_l",
    "terrain_roughness",
    "terrain_friction",
    "terrain_risk",
]
TARGET_NAMES = ["exec_du_vx", "exec_du_vy", "exec_du_wz"]
TARGET_AXES = ("vx", "vy", "wz")


class ResidualFdmMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(TARGET_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _to_int(value) -> int:
    return int(value)


def sequence_fdm_feature_names(
    horizon_steps: int,
    *,
    include_history_controls: bool = True,
    include_history_steps: int = 1,
) -> list[str]:
    """Build deterministic feature names for sequence FDM checkpoints."""
    names: list[str] = [
        "state_x",
        "state_y",
        "state_theta",
        "state_vx",
        "state_vy",
        "state_wz",
    ]
    if include_history_controls:
        for index in range(int(include_history_steps)):
            names.extend(
                [
                    f"history_cmd_t-{index + 1}_vx",
                    f"history_cmd_t-{index + 1}_vy",
                    f"history_cmd_t-{index + 1}_wz",
                ]
            )
    for step in range(_to_int(horizon_steps)):
        names.extend(
            [
                f"future_cmd_t{step + 1}_vx",
                f"future_cmd_t{step + 1}_vy",
                f"future_cmd_t{step + 1}_wz",
                f"terrain_f_t{step + 1}",
                f"terrain_l_t{step + 1}",
                f"terrain_roughness_t{step + 1}",
                f"terrain_friction_t{step + 1}",
                f"terrain_risk_t{step + 1}",
            ]
        )
    return names


def sequence_fdm_target_names(horizon_steps: int) -> list[str]:
    names: list[str] = []
    for step in range(_to_int(horizon_steps)):
        names.extend(
            [
                f"rel_dx_t{step + 1}",
                f"rel_dy_t{step + 1}",
                f"rel_dtheta_t{step + 1}",
                f"terrain_risk_t{step + 1}",
            ]
        )
    return names


class SequenceFdmMlp(nn.Module):
    def __init__(self, input_dim: int, output_horizon: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.output_horizon = _to_int(output_horizon)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_horizon * 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def build_feature_vector(
    state: np.ndarray,
    command: np.ndarray,
    terrain_features: np.ndarray,
    terrain_risk: float,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(state, dtype=np.float32).reshape(6),
            np.asarray(command, dtype=np.float32).reshape(3),
            np.asarray(terrain_features, dtype=np.float32).reshape(4),
            np.asarray([terrain_risk], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
