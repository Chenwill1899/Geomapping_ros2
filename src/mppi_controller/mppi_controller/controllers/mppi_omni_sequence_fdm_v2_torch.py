"""Torch MPPI controller using Sequence FDM V2 for direct trajectory + risk prediction."""

from __future__ import annotations

import numpy as np
import torch

from mppi_controller.controllers.mppi_omni_torch import MppiOmniTorch
from mppi_controller.core.sequence_fdm_v2 import COSTMAP_GRID_DIM, COSTMAP_GRID_SIZE, GOAL_PATH_FEATURE_DIM
from mppi_controller.core.sequence_fdm_dynamics import SequenceFdmDynamics
from mppi_controller.core.terrain import TerrainField
from mppi_controller.core.terrain_grid import sample_local_costmap_grid_torch


class MppiOmniSequenceFdmV2Torch(MppiOmniTorch):
    """MPPI controller that uses Sequence FDM V2 to predict trajectories and risks directly."""

    def __init__(
        self,
        *args,
        sequence_dynamics: SequenceFdmDynamics,
        terrain: TerrainField | None = None,
        device: str = "cuda",
        fdm_risk_weight: float = 10.0,
        costmap_grid_size: int = 9,
        costmap_grid_span: float = 18.0,
        profile_enabled: bool = False,
        **kwargs,
    ) -> None:
        shared_terrain = terrain if terrain is not None else getattr(sequence_dynamics, "terrain", TerrainField())
        super().__init__(*args, terrain=shared_terrain, device=device, profile_enabled=profile_enabled, **kwargs)
        self.sequence_dynamics = sequence_dynamics
        self.fdm_risk_weight = float(fdm_risk_weight)
        self.costmap_grid_size = int(costmap_grid_size)
        self.costmap_grid_span = float(costmap_grid_span)
        if self.costmap_grid_size * self.costmap_grid_size != COSTMAP_GRID_DIM:
            raise ValueError(
                f"costmap_grid_size must be {COSTMAP_GRID_SIZE} so the flattened input is {COSTMAP_GRID_DIM}-D"
            )
        # Ensure model is on correct device
        self.sequence_dynamics.model.to(self.torch_device)

    def _trajectory_cost_batch_torch(
        self,
        initial_state: np.ndarray,
        controls: torch.Tensor,
        goal: torch.Tensor,
        obstacles: torch.Tensor,
        path: torch.Tensor | None = None,
        costmap: dict | None = None,
    ) -> torch.Tensor:
        controls = torch.clamp(controls, -self.max_control_t, self.max_control_t)
        num_samples = int(controls.shape[0])
        H = self.horizon_steps

        # 1. Sample costmap grid centered on current state
        profile_start = self._profile_start()
        x0 = float(initial_state[0])
        y0 = float(initial_state[1])
        if costmap and bool(costmap.get("enabled", False)):
            costmap_grid = sample_local_costmap_grid_torch(
                costmap,
                x=x0,
                y=y0,
                size=self.costmap_grid_size,
                span=self.costmap_grid_span,
            )
        else:
            costmap_grid = torch.zeros(
                self.costmap_grid_size * self.costmap_grid_size,
                dtype=torch.float32,
                device=self.torch_device,
            )
        costmap_grid = costmap_grid.unsqueeze(0).expand(num_samples, -1)
        self._profile_stop("costmap_grid_ms", profile_start)
        goal_path_features = (
            self._goal_path_features_torch(initial_state, goal, path)
            .unsqueeze(0)
            .expand(num_samples, -1)
        )

        # 2. Prepare state tensor
        state_t = torch.as_tensor(
            np.asarray(initial_state, dtype=np.float32).reshape(1, 6),
            dtype=torch.float32,
            device=self.torch_device,
        ).expand(num_samples, -1)

        # 3. FDM forward pass (gradients retained)
        profile_start = self._profile_start()
        pred_states, pred_risk_logits = self.sequence_dynamics.predict_torch(
            state_t,
            controls,
            costmap_grid,
            goal_path_features,
        )
        self._profile_stop("fdm_inference_ms", profile_start)

        # 4. Binary risk from logits
        pred_risk = torch.sigmoid(pred_risk_logits)

        # 5. Compute costs
        profile_start = self._profile_start()
        final_states = pred_states[:, -1, :]
        xy_error = final_states[:, :2] - goal[:2]
        yaw_error = self._angle_diff_torch(final_states[:, 2], goal[2])
        goal_cost = self.goal_xy_weight * torch.sum(xy_error * xy_error, dim=1)
        yaw_cost = self.yaw_weight * yaw_error * yaw_error

        # Control smoothness
        control_cost = self.control_weight * torch.sum(controls * controls, dim=(1, 2))
        previous = torch.as_tensor(self.previous_control, dtype=torch.float32, device=self.torch_device).view(1, 1, 3)
        previous = previous.expand(num_samples, 1, 3)
        control_deltas = torch.diff(torch.cat([previous, controls], dim=1), dim=1)
        smooth_cost = self.smooth_weight * torch.sum(control_deltas * control_deltas, dim=(1, 2))

        # Obstacle cost from predicted trajectory positions
        obstacle_cost = self._obstacle_cost_batch_torch(pred_states[:, 1:, :], obstacles)

        # Risk cost from FDM prediction
        risk_cost = self.fdm_risk_weight * torch.sum(pred_risk, dim=1)

        self._profile_stop("cost_terms_ms", profile_start)

        return (
            goal_cost
            + yaw_cost
            + control_cost
            + smooth_cost
            + obstacle_cost
            + risk_cost
        ).to(torch.float32)

    def _goal_path_features_torch(
        self,
        initial_state: np.ndarray,
        goal: torch.Tensor,
        path: torch.Tensor | None,
    ) -> torch.Tensor:
        state = torch.as_tensor(
            np.asarray(initial_state, dtype=np.float32).reshape(6),
            dtype=torch.float32,
            device=self.torch_device,
        )
        goal = goal.to(dtype=torch.float32, device=self.torch_device).reshape(-1)
        rel_goal = self._world_to_body_delta_torch(state, goal[:2])
        goal_distance = torch.linalg.norm(goal[:2] - state[:2])
        goal_yaw_error = self._angle_diff_torch(goal[2], state[2])

        path_available = torch.tensor(0.0, dtype=torch.float32, device=self.torch_device)
        lookahead = torch.zeros(2, dtype=torch.float32, device=self.torch_device)
        path_end = torch.zeros(2, dtype=torch.float32, device=self.torch_device)
        path_length = torch.tensor(0.0, dtype=torch.float32, device=self.torch_device)
        if path is not None and int(path.numel()) >= 4:
            path_t = path.reshape(-1, 2).to(dtype=torch.float32, device=self.torch_device)
            if path_t.shape[0] >= 2:
                path_available = torch.tensor(1.0, dtype=torch.float32, device=self.torch_device)
                lookahead = self._world_to_body_delta_torch(state, path_t[min(1, path_t.shape[0] - 1)])
                path_end = self._world_to_body_delta_torch(state, path_t[-1])
                path_length = torch.sum(torch.linalg.norm(path_t[1:] - path_t[:-1], dim=1))
        return torch.cat(
            [
                rel_goal,
                goal_distance.view(1),
                goal_yaw_error.view(1),
                lookahead,
                path_end,
                path_length.view(1),
                path_available.view(1),
            ],
            dim=0,
        ).reshape(GOAL_PATH_FEATURE_DIM)

    def _world_to_body_delta_torch(self, state: torch.Tensor, point_xy: torch.Tensor) -> torch.Tensor:
        delta = point_xy.to(dtype=torch.float32, device=self.torch_device).reshape(2) - state[:2]
        cos_yaw = torch.cos(state[2])
        sin_yaw = torch.sin(state[2])
        return torch.stack(
            [
                cos_yaw * delta[0] + sin_yaw * delta[1],
                -sin_yaw * delta[0] + cos_yaw * delta[1],
            ]
        )
