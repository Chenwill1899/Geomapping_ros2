"""Sequence FDM V2 model: predicts trajectories + risk from a costmap grid."""

from __future__ import annotations

import torch
from torch import nn


COSTMAP_GRID_SIZE = 9
COSTMAP_GRID_DIM = COSTMAP_GRID_SIZE * COSTMAP_GRID_SIZE
GOAL_PATH_FEATURE_NAMES = [
    "goal_rel_x",
    "goal_rel_y",
    "goal_distance",
    "goal_yaw_error",
    "path_lookahead_rel_x",
    "path_lookahead_rel_y",
    "path_end_rel_x",
    "path_end_rel_y",
    "path_length",
    "path_available",
]
GOAL_PATH_FEATURE_DIM = len(GOAL_PATH_FEATURE_NAMES)

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
    f"costmap_grid_{i}" for i in range(COSTMAP_GRID_DIM)
] + [
    *GOAL_PATH_FEATURE_NAMES
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
    names.extend([f"costmap_grid_{i}" for i in range(COSTMAP_GRID_DIM)])
    names.extend(GOAL_PATH_FEATURE_NAMES)
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
    """Dual-branch sequence FDM.

    One branch encodes current state plus future controls with an MLP, another
    branch encodes the local costmap with a compact CNN. Goal/path features are
    concatenated at fusion time before predicting future states and risk logits.
    """

    def __init__(self, horizon_steps: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        self.horizon_steps = int(horizon_steps)
        state_control_dim = 6 + 3 * self.horizon_steps
        state_control_embed_dim = hidden_dims[0]
        costmap_embed_dim = max(16, hidden_dims[0] // 2)
        fusion_input_dim = state_control_embed_dim + costmap_embed_dim + GOAL_PATH_FEATURE_DIM
        output_dim = 6 * self.horizon_steps + self.horizon_steps

        self.state_control_encoder = nn.Sequential(
            nn.Linear(state_control_dim, state_control_embed_dim),
            nn.ReLU(),
            nn.Linear(state_control_embed_dim, state_control_embed_dim),
            nn.ReLU(),
        )
        self.costmap_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * COSTMAP_GRID_SIZE * COSTMAP_GRID_SIZE, costmap_embed_dim),
            nn.ReLU(),
        )
        layers: list[nn.Module] = []
        prev_dim = fusion_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        controls: torch.Tensor,
        costmap_grid: torch.Tensor,
        goal_path_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, 6)
            controls: (B, H, 3)
            costmap_grid: (B, 81) or (B, 1, 9, 9)
            goal_path_features: (B, 10)
        Returns:
            states_pred: (B, H, 6)
            risk_logits: (B, H)
        """
        batch_size = state.shape[0]
        controls_flat = controls.reshape(batch_size, -1)
        state_control = torch.cat([state, controls_flat], dim=1)
        state_control_embedding = self.state_control_encoder(state_control)

        if costmap_grid.ndim == 2:
            costmap_image = costmap_grid.reshape(batch_size, 1, COSTMAP_GRID_SIZE, COSTMAP_GRID_SIZE)
        elif costmap_grid.ndim == 3:
            costmap_image = costmap_grid.unsqueeze(1)
        else:
            costmap_image = costmap_grid
        costmap_embedding = self.costmap_encoder(costmap_image)

        if goal_path_features is None:
            goal_path_features = torch.zeros(
                batch_size,
                GOAL_PATH_FEATURE_DIM,
                dtype=state.dtype,
                device=state.device,
            )
        fusion = torch.cat([state_control_embedding, costmap_embedding, goal_path_features], dim=1)
        out = self.head(fusion)
        states_pred = out[:, : 6 * self.horizon_steps].reshape(batch_size, self.horizon_steps, 6)
        risk_logits = out[:, 6 * self.horizon_steps :]
        return states_pred, risk_logits
