"""Torch MPPI controller using Sequence FDM V2 for direct trajectory + risk prediction."""

from __future__ import annotations

import numpy as np
import torch

from mppi_controller.controllers.mppi_omni_torch import MppiOmniTorch
from mppi_controller.core.sequence_fdm_v2 import COSTMAP_GRID_DIM, COSTMAP_GRID_SIZE
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

        # 2. Prepare state tensor
        state_t = torch.as_tensor(
            np.asarray(initial_state, dtype=np.float32).reshape(1, 6),
            dtype=torch.float32,
            device=self.torch_device,
        ).expand(num_samples, -1)

        # 3. FDM forward pass (gradients retained)
        profile_start = self._profile_start()
        pred_states, pred_risk_logits = self.sequence_dynamics.predict_torch(state_t, controls, costmap_grid)
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
