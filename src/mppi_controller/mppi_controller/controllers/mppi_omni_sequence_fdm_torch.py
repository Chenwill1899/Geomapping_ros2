"""Torch MPPI controller using whole-sequence FDM rollout dynamics."""

from __future__ import annotations

import numpy as np
import torch

from mppi_controller.controllers.mppi_omni_torch import MppiOmniTorch
from mppi_controller.core.sequence_fdm_rollout import SequenceFdmRollout
from mppi_controller.core.omni_b2 import OmniB2
from mppi_controller.core.terrain import TerrainField


class MppiOmniSequenceFdmTorch(MppiOmniTorch):
    """Torch MPPI controller that evaluates whole candidate trajectories by learned sequence FDM."""

    def __init__(
        self,
        *args,
        learned_dynamics: SequenceFdmRollout,
        terrain: TerrainField | None = None,
        device: str = "cuda",
        sequence_horizon: int | None = None,
        include_history_controls: bool = True,
        history_steps: int = 1,
        profile_enabled: bool = False,
        **kwargs,
    ) -> None:
        shared_terrain = terrain if terrain is not None else getattr(learned_dynamics, "terrain", TerrainField())
        super().__init__(*args, terrain=shared_terrain, device=device, profile_enabled=profile_enabled, **kwargs)
        self.learned_dynamics = learned_dynamics
        horizon_override = int(sequence_horizon) if sequence_horizon is not None else None
        default_horizon = int(getattr(learned_dynamics, "sequence_horizon", 0))
        self.sequence_horizon = int(horizon_override or default_horizon)
        if self.sequence_horizon <= 0:
            raise ValueError("sequence_horizon must be positive")
        self.include_history_controls = bool(include_history_controls)
        self.history_steps = int(history_steps)
        self._setup_learned_tensors()

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        *,
        learned_dynamics: SequenceFdmRollout | None = None,
        **overrides,
    ) -> "MppiOmniSequenceFdmTorch":
        sim = config["simulation"]
        mppi = config["mppi"]
        robot = config["robot"]
        sampling_rate = float(sim["sampling_rate"])
        dt = 1.0 / sampling_rate
        horizon_steps = int(float(sim["time_horizon"]) * sampling_rate)
        terrain = TerrainField.from_config(config.get("terrain"))
        fdm = config.get("fdm", {})
        device = str(overrides.get("device", fdm.get("device", mppi.get("device", "cuda" if torch.cuda.is_available() else "cpu"))))
        if learned_dynamics is None:
            dynamics_robot = OmniB2(
                dt,
                float(robot["max_vx"]),
                float(robot["max_vy"]),
                float(robot["max_wz"]),
            )
            sequence_horizon = int(fdm.get("sequence_horizon", horizon_steps))
            learned_dynamics = SequenceFdmRollout.from_artifacts(
                fdm["model_dir"],
                robot=dynamics_robot,
                terrain=terrain,
                checkpoint=fdm.get("checkpoint", "best_model.pt"),
                normalization=fdm.get("normalization", "normalization.npz"),
                device=device,
                sequence_horizon=sequence_horizon,
                include_history_controls=bool(fdm.get("sequence_include_history", True)),
                history_steps=int(fdm.get("sequence_history_steps", 1)),
            )
        sequence_horizon = int(
            overrides.get("sequence_horizon", fdm.get("sequence_horizon", int(getattr(learned_dynamics, "sequence_horizon", 0))))
        )
        if sequence_horizon != horizon_steps:
            raise ValueError(
                f"sequence_horizon={sequence_horizon} must match mppi horizon_steps={horizon_steps}"
            )
        return cls(
            dt=dt,
            horizon_steps=horizon_steps,
            num_samples=int(mppi["num_trajectories"]),
            lambda_=float(mppi["lambda"]),
            noise_std=np.asarray(mppi["std_normal"], dtype=np.float32),
            max_vx=float(robot["max_vx"]),
            max_vy=float(robot["max_vy"]),
            max_wz=float(robot["max_wz"]),
            goal_xy_weight=float(overrides.get("goal_xy_weight", mppi["weights"][0])),
            yaw_weight=float(overrides.get("yaw_weight", mppi["weights"][2])),
            control_weight=float(overrides.get("control_weight", mppi.get("control_weight", 0.01))),
            smooth_weight=float(overrides.get("smooth_weight", mppi.get("smooth_weight", 0.2))),
            obstacle_weight=float(overrides.get("obstacle_weight", mppi.get("obstacle_weight", 25.0))),
            obstacle_soft_weight=float(
                overrides.get("obstacle_soft_weight", mppi.get("obstacle_soft_weight", 0.0))
            ),
            obstacle_influence_dist=float(
                overrides.get("obstacle_influence_dist", mppi.get("obstacle_influence_dist", 0.0))
            ),
            max_ax=float(overrides.get("max_ax", robot.get("max_ax", 1000.0))),
            max_ay=float(overrides.get("max_ay", robot.get("max_ay", 1000.0))),
            max_awz=float(overrides.get("max_awz", robot.get("max_awz", 1000.0))),
            velocity_lag_beta=float(overrides.get("velocity_lag_beta", robot.get("velocity_lag_beta", 0.0))),
            lateral_weight=float(overrides.get("lateral_weight", mppi.get("lateral_weight", 0.0))),
            yaw_rate_weight=float(overrides.get("yaw_rate_weight", mppi.get("yaw_rate_weight", 0.0))),
            accel_weight=float(overrides.get("accel_weight", mppi.get("accel_weight", 0.0))),
            jerk_weight=float(overrides.get("jerk_weight", mppi.get("jerk_weight", 0.0))),
            path_tracking_weight=float(
                overrides.get("path_tracking_weight", mppi.get("path_tracking_weight", 0.0))
            ),
            path_tracking_tolerance=float(
                overrides.get("path_tracking_tolerance", mppi.get("path_tracking_tolerance", 0.3))
            ),
            path_progress_weight=float(
                overrides.get("path_progress_weight", mppi.get("path_progress_weight", 0.0))
            ),
            goal_progress_weight=float(
                overrides.get("goal_progress_weight", mppi.get("goal_progress_weight", 0.0))
            ),
            heading_to_goal_weight=float(
                overrides.get("heading_to_goal_weight", mppi.get("heading_to_goal_weight", 0.0))
            ),
            heading_to_goal_min_distance=float(
                overrides.get(
                    "heading_to_goal_min_distance",
                    mppi.get("heading_to_goal_min_distance", 0.3),
                )
            ),
            terrain=terrain,
            terrain_risk_weight=float(overrides.get("terrain_risk_weight", mppi.get("terrain_risk_weight", 0.0))),
            terrain_risk_power=float(overrides.get("terrain_risk_power", mppi.get("terrain_risk_power", 2.0))),
            terrain_risk_threshold=float(
                overrides.get("terrain_risk_threshold", mppi.get("terrain_risk_threshold", 0.0))
            ),
            terrain_risk_mode=str(overrides.get("terrain_risk_mode", mppi.get("terrain_risk_mode", "excess"))),
            robot_radius=float(robot["radius"]),
            safety_dist=float(robot["safety_dist"]),
            draw_num_traj=int(mppi["draw_num_traj"]),
            seed=seed,
            learned_dynamics=learned_dynamics,
            device=device,
            sequence_horizon=sequence_horizon,
            include_history_controls=bool(fdm.get("sequence_include_history", True)),
            history_steps=int(fdm.get("sequence_history_steps", 1)),
            profile_enabled=bool(overrides.get("profile_enabled", fdm.get("profile_enabled", False))),
        )

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
        # Keep control penalties from current response dynamics for stability constraints.
        _, real_controls = self._rollout_batch_torch(initial_state, controls)
        profile_start = self._profile_start()
        rel_traj, terrain_risk_pred = self._predict_sequence_batch_torch(initial_state, controls)
        self._profile_stop("fdm_inference_ms", profile_start)
        pred_states = self._sequence_to_states_torch(initial_state, rel_traj)

        profile_start = self._profile_start()
        final_states = pred_states[:, -1, :]
        xy_error = final_states[:, :2] - goal[:2]
        yaw_error = self._angle_diff_torch(final_states[:, 2], goal[2])
        goal_cost = self.goal_xy_weight * torch.sum(xy_error * xy_error, dim=1)
        yaw_cost = self.yaw_weight * yaw_error * yaw_error
        control_cost = self.control_weight * torch.sum(controls * controls, dim=(1, 2))
        previous = torch.as_tensor(self.previous_control, dtype=torch.float32, device=self.torch_device).view(1, 1, 3)
        previous = previous.expand(controls.shape[0], 1, 3)
        control_deltas = torch.diff(torch.cat([previous, controls], dim=1), dim=1)
        smooth_cost = self.smooth_weight * torch.sum(control_deltas * control_deltas, dim=(1, 2))
        initial_velocity = torch.as_tensor(
            np.asarray(initial_state, dtype=np.float32)[3:],
            dtype=torch.float32,
            device=self.torch_device,
        ).view(1, 1, 3)
        initial_velocity = initial_velocity.expand(controls.shape[0], 1, 3)
        accel = torch.diff(torch.cat([initial_velocity, real_controls], dim=1), dim=1) / self.dt
        accel_cost = self.accel_weight * torch.sum(accel * accel, dim=(1, 2))
        jerk = torch.diff(torch.cat([initial_velocity, real_controls], dim=1), n=2, dim=1)
        jerk_cost = self.jerk_weight * torch.sum(jerk * jerk, dim=(1, 2))
        lateral_cost = self.lateral_weight * torch.sum(real_controls[:, :, 1] * real_controls[:, :, 1], dim=1)
        yaw_rate_cost = self.yaw_rate_weight * torch.sum(real_controls[:, :, 2] * real_controls[:, :, 2], dim=1)
        self._profile_stop("cost_terms_ms", profile_start)
        profile_start = self._profile_start()
        obstacle_cost = self._obstacle_cost_batch_torch(pred_states[:, 1:, :], obstacles)
        self._profile_stop("obstacle_cost_ms", profile_start)
        profile_start = self._profile_start()
        path_tracking_cost = self._path_tracking_cost_batch_torch(pred_states[:, 1:, :], path)
        self._profile_stop("path_tracking_cost_ms", profile_start)
        terrain_risk_cost = self._terrain_risk_cost_from_sequence_torch(terrain_risk_pred, self.terrain_risk_mode)
        goal_progress_cost = self._goal_progress_cost_batch_torch(initial_state, pred_states[:, -1, :], goal)
        heading_to_goal_cost = self._heading_to_goal_cost_batch_torch(pred_states[:, 1:, :], goal)
        local_costmap_cost = self._local_costmap_cost_batch_torch(pred_states[:, 1:, :], initial_state, costmap)
        return (
            goal_cost
            + yaw_cost
            + control_cost
            + smooth_cost
            + accel_cost
            + jerk_cost
            + lateral_cost
            + yaw_rate_cost
            + obstacle_cost
            + path_tracking_cost
            + goal_progress_cost
            + heading_to_goal_cost
            + terrain_risk_cost
            + local_costmap_cost
        ).to(torch.float32)

    def _terrain_risk_cost_from_sequence_torch(self, risks: torch.Tensor, mode: str) -> torch.Tensor:
        if (
            self.terrain_risk_weight <= 0.0
            or mode == "none"
            or risks is None
            or not self.terrain.enabled
        ):
            return torch.zeros(risks.shape[0], dtype=torch.float32, device=self.torch_device)
        if risks.shape[1] != self.sequence_horizon:
            raise ValueError(
                f"sequence risk length {risks.shape[1]} does not match controller sequence_horizon {self.sequence_horizon}"
            )
        terms = self._terrain_risk_terms_torch(risks)
        return self.terrain_risk_weight * torch.sum(terms, dim=1)

    def _predict_sequence_batch_torch(self, initial_state: np.ndarray, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history = torch.as_tensor(self.previous_control, dtype=torch.float32, device=self.torch_device).view(1, 3)
        profile_start = self._profile_start()
        rel_pose, pred_risk = self.learned_dynamics.predict_sequence_batch_torch(
            states=torch.as_tensor(np.asarray(initial_state, dtype=np.float32).reshape(1, 6), device=self.torch_device).expand(
                len(controls), -1
            ),
            command_sequences=torch.clamp(controls, -self.max_control_t, self.max_control_t),
            history=history,
        )
        self._profile_stop("fdm_inference_ms", profile_start)
        rel_pose = rel_pose.to(torch.float32, non_blocking=True)
        pred_risk = pred_risk.to(torch.float32, non_blocking=True)
        return rel_pose, pred_risk

    def _sequence_to_states_torch(self, initial_state: np.ndarray, rel_traj: torch.Tensor) -> torch.Tensor:
        num_samples = int(rel_traj.shape[0])
        if rel_traj.shape[1] != self.sequence_horizon:
            raise ValueError(
                f"predicted trajectory horizon {rel_traj.shape[1]} does not match controller horizon {self.sequence_horizon}"
            )
        initial = torch.as_tensor(np.asarray(initial_state, dtype=np.float32), device=self.torch_device).view(1, 6)
        if initial.shape[0] == 1 and num_samples > 1:
            initial = initial.expand(num_samples, 6)
        states = torch.zeros((num_samples, self.sequence_horizon + 1, 6), dtype=torch.float32, device=self.torch_device)
        states[:, 0, :3] = initial[:, :3]
        rel_xy = rel_traj[:, :, :2]
        rel_yaw = rel_traj[:, :, 2]
        states[:, 1:, 0] = initial[:, :1] + rel_xy[:, :, 0]
        states[:, 1:, 1] = initial[:, 1:2] + rel_xy[:, :, 1]
        states[:, 1:, 2] = initial[:, 2:3] + rel_yaw
        states[:, 1:, 3:] = torch.zeros((num_samples, self.sequence_horizon, 3), dtype=torch.float32, device=self.torch_device)
        return states

    def _setup_learned_tensors(self) -> None:
        model = getattr(self.learned_dynamics, "model", None)
        if model is not None:
            model.to(self.torch_device)
            model.eval()
        if hasattr(self.learned_dynamics, "device"):
            self.learned_dynamics.device = self.torch_device
        if hasattr(self.learned_dynamics, "feature_mean"):
            self.feature_mean_t = torch.as_tensor(
                self.learned_dynamics.feature_mean,
                dtype=torch.float32,
                device=self.torch_device,
            )
            self.feature_std_t = torch.as_tensor(
                self.learned_dynamics.feature_std,
                dtype=torch.float32,
                device=self.torch_device,
            )
            self.target_mean_t = torch.as_tensor(
                self.learned_dynamics.target_mean,
                dtype=torch.float32,
                device=self.torch_device,
            )
            self.target_std_t = torch.as_tensor(
                self.learned_dynamics.target_std,
                dtype=torch.float32,
                device=self.torch_device,
            )
