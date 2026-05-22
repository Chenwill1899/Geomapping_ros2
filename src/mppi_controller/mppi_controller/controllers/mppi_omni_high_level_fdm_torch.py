"""Torch MPPI controller using high_level_fdm TorchScript rollout predictions."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from mppi_controller.controllers.mppi_omni_torch import MppiOmniTorch
from mppi_controller.core.high_level_fdm_runtime import HighLevelFdmRuntime
from mppi_controller.core.terrain import TerrainField


class MppiOmniHighLevelFdmTorch(MppiOmniTorch):
    """MPPI controller that replaces nominal trajectory rollout with high_level_fdm."""

    def __init__(
        self,
        *args,
        learned_runtime: HighLevelFdmRuntime,
        terrain: TerrainField | None = None,
        device: str = "cuda",
        risk_weight: float = 10.0,
        risk_threshold: float = 0.45,
        risk_power: float = 2.0,
        profile_enabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, terrain=terrain, device=device, profile_enabled=profile_enabled, **kwargs)
        self.learned_runtime = learned_runtime
        self.history_len = int(learned_runtime.metadata.history_len)
        self.sequence_horizon = int(learned_runtime.metadata.horizon)
        self.risk_weight = float(risk_weight)
        self.risk_threshold = float(risk_threshold)
        self.risk_power = float(risk_power)
        self._state_history: deque[np.ndarray] = deque(maxlen=self.history_len)
        self._command_history: deque[np.ndarray] = deque(maxlen=self.history_len)
        if self.sequence_horizon != self.horizon_steps:
            raise ValueError(
                f"high_level_fdm horizon={self.sequence_horizon} must match MPPI horizon_steps={self.horizon_steps}"
            )
        if int(learned_runtime.metadata.map_channels) != 1:
            raise ValueError("Geomapping high_level_fdm runtime currently supports one map channel")

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        *,
        learned_runtime: HighLevelFdmRuntime | None = None,
        **overrides,
    ) -> "MppiOmniHighLevelFdmTorch":
        sim = config["simulation"]
        mppi = config["mppi"]
        robot = config["robot"]
        fdm = config.get("fdm", {})
        sampling_rate = float(sim["sampling_rate"])
        dt = 1.0 / sampling_rate
        horizon_steps = int(float(sim["time_horizon"]) * sampling_rate)
        terrain = TerrainField.from_config(config.get("terrain"))
        device = str(overrides.get("device", fdm.get("device", mppi.get("device", "cuda" if torch.cuda.is_available() else "cpu"))))
        if learned_runtime is None:
            learned_runtime = HighLevelFdmRuntime.from_model_dir(
                fdm["model_dir"],
                device=device,
                model_file=str(fdm.get("model_file", "fdm_ts.pt")),
                metadata_file=str(fdm.get("metadata_file", "fdm_metadata.json")),
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
            min_vx=float(overrides.get("min_vx", robot.get("min_vx", -float(robot["max_vx"])))),
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
            update_smoothing_alpha=overrides.get("update_smoothing_alpha", mppi.get("update_smoothing_alpha", 0.0)),
            goal_change_reset_distance=float(
                overrides.get("goal_change_reset_distance", mppi.get("goal_change_reset_distance", 0.75))
            ),
            goal_change_reset_yaw=float(
                overrides.get("goal_change_reset_yaw", mppi.get("goal_change_reset_yaw", 1.0))
            ),
            terrain=terrain,
            terrain_risk_weight=0.0,
            terrain_risk_mode="none",
            robot_radius=float(robot["radius"]),
            safety_dist=float(robot["safety_dist"]),
            draw_num_traj=int(mppi["draw_num_traj"]),
            seed=seed,
            learned_runtime=learned_runtime,
            device=device,
            risk_weight=float(overrides.get("risk_weight", fdm.get("risk_weight", mppi.get("learned_risk_weight", 10.0)))),
            risk_threshold=float(
                overrides.get("risk_threshold", fdm.get("risk_threshold", mppi.get("learned_risk_threshold", 0.45)))
            ),
            risk_power=float(overrides.get("risk_power", fdm.get("risk_power", mppi.get("learned_risk_power", 2.0)))),
            profile_enabled=bool(overrides.get("profile_enabled", fdm.get("profile_enabled", False))),
        )

    def compute_control(self, state: np.ndarray, cost_params):
        self._record_history_context(state)
        return super().compute_control(state, cost_params)

    def _record_history_context(self, state: np.ndarray) -> None:
        self._state_history.append(np.asarray(state, dtype=np.float32).reshape(6).copy())
        self._command_history.append(np.asarray(self.previous_control, dtype=np.float32).reshape(3).copy())

    def _trajectory_cost_batch_torch(
        self,
        initial_state: np.ndarray,
        controls: torch.Tensor,
        goal: torch.Tensor,
        obstacles: torch.Tensor,
        path: torch.Tensor | None = None,
        costmap: dict | None = None,
    ) -> torch.Tensor:
        del obstacles
        controls = torch.clamp(controls, self.min_control_t, self.max_control_t)
        _, real_controls = self._rollout_batch_torch(initial_state, controls)
        profile_start = self._profile_start()
        rel_pose, risk_pred, applied_twist = self._predict_high_level_batch_torch(initial_state, controls, costmap)
        self._profile_stop("fdm_inference_ms", profile_start)
        pred_states = self.relative_pose_to_world_states(initial_state, rel_pose, applied_twist)

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

        path_tracking_cost = self._path_tracking_cost_batch_torch(pred_states[:, 1:, :], path)
        path_progress_cost = self._path_progress_cost_batch_torch(initial_state, pred_states[:, -1, :], path)
        goal_progress_cost = self._goal_progress_cost_batch_torch(initial_state, pred_states[:, -1, :], goal)
        heading_to_goal_cost = self._heading_to_goal_cost_batch_torch(pred_states[:, 1:, :], goal)
        learned_risk_cost = self._learned_risk_cost_torch(risk_pred)
        return (
            goal_cost
            + yaw_cost
            + control_cost
            + smooth_cost
            + accel_cost
            + jerk_cost
            + lateral_cost
            + yaw_rate_cost
            + path_tracking_cost
            + path_progress_cost
            + goal_progress_cost
            + heading_to_goal_cost
            + learned_risk_cost
        ).to(torch.float32)

    def _predict_high_level_batch_torch(
        self,
        initial_state: np.ndarray,
        controls: torch.Tensor,
        costmap: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = self._history_tensor(initial_state)
        local_map = self._local_map_tensor(initial_state, costmap)
        candidates = int(controls.shape[0])
        history_batch = history.unsqueeze(0).expand(candidates, -1, -1).contiguous()
        map_batch = local_map.expand(candidates, -1, -1, -1).contiguous()
        pose, risk, applied_twist = self.learned_runtime.predict(history_batch, map_batch, controls)
        if pose.shape[1] != self.horizon_steps:
            raise ValueError(f"high_level_fdm returned horizon {pose.shape[1]}, expected {self.horizon_steps}")
        return pose, risk, applied_twist

    def _history_tensor(self, initial_state: np.ndarray) -> torch.Tensor:
        current = np.asarray(initial_state, dtype=np.float32).reshape(6)
        states = list(self._state_history)
        commands = list(self._command_history)
        if not states:
            states = [current.copy()]
            commands = [np.zeros(3, dtype=np.float32)]
        while len(states) < self.history_len:
            states.insert(0, states[0].copy())
            commands.insert(0, commands[0].copy())
        states_arr = np.asarray(states[-self.history_len :], dtype=np.float32)
        commands_arr = np.asarray(commands[-self.history_len :], dtype=np.float32)
        rel_pose = self._relative_history_pose(current, states_arr)
        history = np.concatenate([rel_pose, states_arr[:, 3:6], commands_arr], axis=1)
        return torch.as_tensor(history, dtype=torch.float32, device=self.torch_device)

    @staticmethod
    def _relative_history_pose(origin_state: np.ndarray, states: np.ndarray) -> np.ndarray:
        origin = np.asarray(origin_state, dtype=np.float32).reshape(6)
        states = np.asarray(states, dtype=np.float32).reshape(-1, 6)
        dx = states[:, 0] - origin[0]
        dy = states[:, 1] - origin[1]
        cos_yaw = np.cos(origin[2])
        sin_yaw = np.sin(origin[2])
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        dyaw = (states[:, 2] - origin[2] + np.pi) % (2.0 * np.pi) - np.pi
        return np.stack([local_x, local_y, np.sin(dyaw), np.cos(dyaw)], axis=1).astype(np.float32)

    def _local_map_tensor(self, initial_state: np.ndarray, costmap: dict | None) -> torch.Tensor:
        meta = self.learned_runtime.metadata
        map_size = int(meta.map_size)
        if not costmap or not bool(costmap.get("enabled", False)):
            return torch.zeros((1, int(meta.map_channels), map_size, map_size), dtype=torch.float32, device=self.torch_device)
        width = int(costmap.get("width", 0))
        height = int(costmap.get("height", 0))
        resolution = float(costmap.get("resolution", 0.0))
        data = costmap.get("data")
        if width <= 0 or height <= 0 or resolution <= 0.0 or data is None or int(data.numel()) != width * height:
            return torch.zeros((1, int(meta.map_channels), map_size, map_size), dtype=torch.float32, device=self.torch_device)
        origin = costmap["origin"].to(dtype=torch.float32, device=self.torch_device)
        state = torch.as_tensor(initial_state, dtype=torch.float32, device=self.torch_device).reshape(6)
        cells = torch.arange(map_size, dtype=torch.float32, device=self.torch_device) - float(map_size // 2)
        local_x, local_y = torch.meshgrid(cells * resolution, cells * resolution, indexing="ij")
        cos_yaw = torch.cos(state[2])
        sin_yaw = torch.sin(state[2])
        world_x = state[0] + cos_yaw * local_x - sin_yaw * local_y
        world_y = state[1] + sin_yaw * local_x + cos_yaw * local_y
        ix = torch.floor((world_x - origin[0]) / resolution).to(torch.long)
        iy = torch.floor((world_y - origin[1]) / resolution).to(torch.long)
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        unknown = float(costmap.get("unknown_cost", meta.map_max_cost))
        sampled = torch.full((map_size, map_size), unknown, dtype=torch.float32, device=self.torch_device)
        if bool(torch.any(valid).item()):
            flat_idx = iy[valid] * width + ix[valid]
            sampled[valid] = data[flat_idx]
        max_cost = max(float(meta.map_max_cost), 1e-6)
        patch = torch.clamp(sampled, 0.0, max_cost) / max_cost
        return patch.view(1, 1, map_size, map_size)

    @staticmethod
    def relative_pose_to_world_states(
        initial_state: np.ndarray | torch.Tensor,
        rel_pose: torch.Tensor,
        applied_twist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = rel_pose.device
        initial = torch.as_tensor(initial_state, dtype=torch.float32, device=device).reshape(1, 6)
        batch = int(rel_pose.shape[0])
        horizon = int(rel_pose.shape[1])
        if batch > 1:
            initial = initial.expand(batch, 6)
        states = torch.zeros((batch, horizon + 1, 6), dtype=torch.float32, device=device)
        states[:, 0, :] = initial
        rel_x = rel_pose[:, :, 0]
        rel_y = rel_pose[:, :, 1]
        rel_yaw = torch.atan2(rel_pose[:, :, 2], rel_pose[:, :, 3])
        cos_yaw = torch.cos(initial[:, 2]).view(batch, 1)
        sin_yaw = torch.sin(initial[:, 2]).view(batch, 1)
        states[:, 1:, 0] = initial[:, 0:1] + cos_yaw * rel_x - sin_yaw * rel_y
        states[:, 1:, 1] = initial[:, 1:2] + sin_yaw * rel_x + cos_yaw * rel_y
        states[:, 1:, 2] = initial[:, 2:3] + rel_yaw
        if applied_twist is not None and applied_twist.shape[:2] == rel_pose.shape[:2]:
            states[:, 1:, 3:] = applied_twist.to(dtype=torch.float32, device=device)
        return states

    def _learned_risk_cost_torch(self, risks: torch.Tensor) -> torch.Tensor:
        if self.risk_weight <= 0.0:
            return torch.zeros(risks.shape[0], dtype=torch.float32, device=self.torch_device)
        risk_any = risks[:, :, self.learned_runtime.risk_any_index]
        excess = torch.clamp(risk_any - self.risk_threshold, min=0.0)
        return self.risk_weight * torch.sum(torch.pow(excess, self.risk_power), dim=1).to(torch.float32)
