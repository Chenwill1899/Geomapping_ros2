"""Torch batched MPPI controller for nominal B2 omnidirectional SE(2) rollout."""

from __future__ import annotations

import time

import numpy as np
import torch

from mppi_controller.controllers.mppi_omni_numpy import MppiOmniNumpy
from mppi_controller.core.terrain import TerrainField


class MppiOmniTorch(MppiOmniNumpy):
    """Nominal MPPI backend with batched Torch rollout and cost terms."""

    def __init__(
        self,
        *args,
        terrain: TerrainField | None = None,
        device: str = "cuda",
        profile_enabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, terrain=terrain, **kwargs)
        self.torch_device = torch.device(device)
        self.profile_enabled = bool(profile_enabled)
        self._profile_totals_ms: dict[str, float] = {}
        self._profile_counts: dict[str, int] = {}
        self._profile_total_calls = 0
        if self.torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Torch MPPI backend requested CUDA, but torch.cuda.is_available() is false")
        self.max_control_t = torch.as_tensor(self.max_control, dtype=torch.float32, device=self.torch_device)
        self.min_control_t = torch.as_tensor(self.min_control, dtype=torch.float32, device=self.torch_device)
        self.max_accel_t = torch.as_tensor(self.max_accel, dtype=torch.float32, device=self.torch_device)
        self.noise_std_t = torch.as_tensor(self.noise_std, dtype=torch.float32, device=self.torch_device)
        self.previous_control_t = torch.zeros(3, dtype=torch.float32, device=self.torch_device)
        self.generator = torch.Generator(device=self.torch_device)
        self.generator.manual_seed(int(kwargs.get("seed", 0) or 0))
        self._setup_terrain_tensors()

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        **overrides,
    ) -> "MppiOmniTorch":
        sim = config["simulation"]
        mppi = config["mppi"]
        robot = config["robot"]
        sampling_rate = float(sim["sampling_rate"])
        dt = 1.0 / sampling_rate
        horizon_steps = int(float(sim["time_horizon"]) * sampling_rate)
        terrain = TerrainField.from_config(config.get("terrain"))
        device = str(overrides.get("device", mppi.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
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
            device=device,
            profile_enabled=bool(overrides.get("profile_enabled", mppi.get("profile_enabled", False))),
        )

    def compute_control(self, state: np.ndarray, cost_params):
        with torch.inference_mode():
            return self._compute_control_impl(state, cost_params)

    def _compute_control_impl(self, state: np.ndarray, cost_params):
        if self.profile_enabled:
            self._profile_total_calls += 1
        goal_np = np.asarray(cost_params[3], dtype=np.float32)
        self._reset_nominal_on_goal_change(goal_np)
        goal = torch.as_tensor(goal_np, device=self.torch_device)
        obstacles = torch.as_tensor(
            np.asarray(cost_params[4], dtype=np.float32).reshape(-1, 7),
            device=self.torch_device,
        )
        path = torch.as_tensor(self._path_from_cost_params(cost_params), dtype=torch.float32, device=self.torch_device)
        costmap = self._costmap_to_torch(self._costmap_from_cost_params(cost_params))
        profile_start = self._profile_start()
        nominal = torch.as_tensor(self.nominal_u, dtype=torch.float32, device=self.torch_device)
        previous_nominal = nominal.clone()
        noise = torch.randn(
            (self.num_samples, self.horizon_steps, 3),
            dtype=torch.float32,
            device=self.torch_device,
            generator=self.generator,
        ) * self.noise_std_t
        candidates = torch.clamp(nominal.unsqueeze(0) + noise, self.min_control_t, self.max_control_t)
        self._profile_stop("sample_candidates_ms", profile_start)
        costs = self._trajectory_cost_batch_torch(state, candidates, goal, obstacles, path, costmap)
        profile_start = self._profile_start()
        min_cost_t = torch.min(costs)
        weights = torch.exp(-(costs - min_cost_t) / max(self.lambda_, 1e-6))
        normalizer_t = torch.sum(weights)
        if not bool(torch.isfinite(normalizer_t).item()) or float(normalizer_t.detach().cpu()) <= 0.0:
            weights = torch.full((self.num_samples,), 1.0 / self.num_samples, dtype=torch.float32, device=self.torch_device)
            normalizer_t = torch.tensor(1.0, dtype=torch.float32, device=self.torch_device)
        else:
            weights = weights / normalizer_t
        nominal = torch.sum(weights[:, None, None] * candidates, dim=0)
        if self._has_nominal_update and bool(np.any(self.update_smoothing_alpha > 0.0)):
            alpha = torch.as_tensor(
                self.update_smoothing_alpha.reshape(1, 3),
                dtype=torch.float32,
                device=self.torch_device,
            )
            nominal = alpha * previous_nominal + (1.0 - alpha) * nominal
        self._has_nominal_update = True
        nominal = torch.clamp(nominal, self.min_control_t, self.max_control_t)
        self._profile_stop("update_distribution_ms", profile_start)
        profile_start = self._profile_start()
        self.nominal_u = nominal.detach().cpu().numpy().astype(np.float32)
        command = self.nominal_u[0].copy()
        control = self._apply_velocity_response(state, command).astype(np.float32)
        optimal_u = self.nominal_u.copy()
        sample_u = candidates[: self.draw_num_traj].detach().cpu().numpy().astype(np.float32)
        normalizer = float(normalizer_t.detach().cpu())
        min_cost = float(min_cost_t.detach().cpu())
        self._profile_stop("cpu_transfer_ms", profile_start)
        self.previous_control = command.copy()
        self.previous_control_t = torch.as_tensor(self.previous_control, dtype=torch.float32, device=self.torch_device)
        self._shift_nominal_controls()
        return control, optimal_u, sample_u, normalizer, min_cost

    def trajectory_cost_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        path: np.ndarray | None = None,
        costmap: dict | None = None,
    ) -> np.ndarray:
        controls_t = torch.as_tensor(np.asarray(controls, dtype=np.float32), device=self.torch_device)
        controls_t = torch.clamp(controls_t, self.min_control_t, self.max_control_t)
        goal_t = torch.as_tensor(np.asarray(goal, dtype=np.float32), device=self.torch_device)
        obstacles_t = torch.as_tensor(np.asarray(obstacles, dtype=np.float32).reshape(-1, 7), device=self.torch_device)
        path_t = torch.as_tensor(np.asarray(path if path is not None else [], dtype=np.float32).reshape(-1, 2), device=self.torch_device)
        costmap_t = self._costmap_to_torch(costmap)
        costs = self._trajectory_cost_batch_torch(initial_state, controls_t, goal_t, obstacles_t, path_t, costmap_t)
        return costs.detach().cpu().numpy().astype(np.float32)

    def _rollout_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        return_controls: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        controls_t = torch.as_tensor(np.asarray(controls, dtype=np.float32), device=self.torch_device)
        states_t, real_controls_t = self._rollout_batch_torch(initial_state, controls_t)
        states = states_t.detach().cpu().numpy().astype(np.float32)
        real_controls = real_controls_t.detach().cpu().numpy().astype(np.float32)
        if return_controls:
            return states, real_controls
        return states

    def _trajectory_cost_batch_torch(
        self,
        initial_state: np.ndarray,
        controls: torch.Tensor,
        goal: torch.Tensor,
        obstacles: torch.Tensor,
        path: torch.Tensor | None = None,
        costmap: dict | None = None,
    ) -> torch.Tensor:
        controls = torch.clamp(controls, self.min_control_t, self.max_control_t)
        profile_start = self._profile_start()
        states, real_controls = self._rollout_batch_torch(initial_state, controls)
        self._profile_stop("rollout_total_ms", profile_start)
        profile_start = self._profile_start()
        final_states = states[:, -1, :]
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
        obstacle_cost = self._obstacle_cost_batch_torch(states[:, 1:, :], obstacles)
        self._profile_stop("obstacle_cost_ms", profile_start)
        profile_start = self._profile_start()
        terrain_risk_cost = self._terrain_risk_cost_batch_torch(states[:, 1:, :])
        self._profile_stop("terrain_risk_cost_ms", profile_start)
        path_tracking_cost = self._path_tracking_cost_batch_torch(states[:, 1:, :], path)
        path_progress_cost = self._path_progress_cost_batch_torch(initial_state, states[:, -1, :], path)
        goal_progress_cost = self._goal_progress_cost_batch_torch(initial_state, states[:, -1, :], goal)
        heading_to_goal_cost = self._heading_to_goal_cost_batch_torch(states[:, 1:, :], goal)
        local_costmap_cost = self._local_costmap_cost_batch_torch(states[:, 1:, :], initial_state, costmap)
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
            + path_progress_cost
            + goal_progress_cost
            + heading_to_goal_cost
            + terrain_risk_cost
            + local_costmap_cost
        ).to(torch.float32)

    def _rollout_batch_torch(
        self,
        initial_state: np.ndarray,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        controls = torch.clamp(controls.to(torch.float32), self.min_control_t, self.max_control_t)
        num_samples, horizon_steps, _ = controls.shape
        states = torch.zeros((num_samples, horizon_steps + 1, 6), dtype=torch.float32, device=self.torch_device)
        states[:, 0, :] = torch.as_tensor(np.asarray(initial_state, dtype=np.float32), device=self.torch_device)
        real_controls = torch.zeros((num_samples, horizon_steps, 3), dtype=torch.float32, device=self.torch_device)
        prev_real = states[:, 0, 3:].clone()
        max_delta = self.max_accel_t * self.dt
        with torch.no_grad():
            for step in range(horizon_steps):
                prev = states[:, step, :]
                command = controls[:, step, :]
                profile_start = self._profile_start()
                lagged = self.velocity_lag_beta * prev_real + (1.0 - self.velocity_lag_beta) * command
                delta = torch.clamp(lagged - prev_real, -max_delta, max_delta)
                response_command = torch.clamp(prev_real + delta, self.min_control_t, self.max_control_t)
                self._profile_stop("response_update_ms", profile_start)
                residual = self._predict_residual_torch(prev, response_command)
                profile_start = self._profile_start()
                control = torch.clamp(response_command + residual, self.min_control_t, self.max_control_t)
                real_controls[:, step, :] = control
                theta = prev[:, 2]
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)
                vx = control[:, 0]
                vy = control[:, 1]
                wz = control[:, 2]
                states[:, step + 1, 0] = prev[:, 0] + (vx * cos_theta - vy * sin_theta) * self.dt
                states[:, step + 1, 1] = prev[:, 1] + (vx * sin_theta + vy * cos_theta) * self.dt
                states[:, step + 1, 2] = prev[:, 2] + wz * self.dt
                states[:, step + 1, 3] = vx
                states[:, step + 1, 4] = vy
                states[:, step + 1, 5] = wz
                prev_real = control
                self._profile_stop("state_integrate_ms", profile_start)
        return states, real_controls

    def _predict_residual_torch(self, states: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(commands, dtype=torch.float32, device=self.torch_device)

    def _obstacle_cost_batch_torch(self, states: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
        costs = torch.zeros(states.shape[0], dtype=torch.float32, device=self.torch_device)
        if obstacles.numel() == 0:
            return costs
        centers = obstacles[:, :2].to(torch.float32)
        radii = obstacles[:, 2].to(torch.float32)
        deltas = states[:, :, None, :2] - centers[None, None, :, :]
        clearance = torch.linalg.norm(deltas, dim=3) - radii[None, None, :] - self.robot_radius
        margin = torch.clamp(self.safety_dist - clearance, min=0.0)
        costs += self.obstacle_weight * torch.sum(margin * margin, dim=(1, 2))
        if self.obstacle_soft_weight > 0.0 and self.obstacle_influence_dist > self.safety_dist:
            soft_margin = torch.clamp(self.obstacle_influence_dist - clearance, min=0.0)
            soft_margin = torch.where(clearance > self.safety_dist, soft_margin, torch.zeros_like(soft_margin))
            costs += self.obstacle_soft_weight * torch.sum(soft_margin * soft_margin, dim=(1, 2))
        return costs

    def _costmap_to_torch(self, costmap: dict | None) -> dict | None:
        if not costmap or not bool(costmap.get("enabled", False)):
            return None
        return {
            "enabled": True,
            "origin": torch.as_tensor(
                np.asarray(costmap.get("origin", [0.0, 0.0]), dtype=np.float32).reshape(2),
                dtype=torch.float32,
                device=self.torch_device,
            ),
            "resolution": float(costmap.get("resolution", 0.0)),
            "width": int(costmap.get("width", 0)),
            "height": int(costmap.get("height", 0)),
            "data": torch.as_tensor(
                np.asarray(costmap.get("data", []), dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self.torch_device,
            ),
            "unknown_mask": torch.as_tensor(
                np.asarray(costmap.get("unknown_mask", []), dtype=bool).reshape(-1),
                dtype=torch.bool,
                device=self.torch_device,
            ),
            "weight": float(costmap.get("weight", 0.0)),
            "power": float(costmap.get("power", 2.0)),
            "unknown_cost": float(costmap.get("unknown_cost", 100.0)),
            "max_cost": float(costmap.get("max_cost", 100.0)),
            "unknown_clear_radius": float(costmap.get("unknown_clear_radius", 0.0)),
            "unknown_clear_value": float(costmap.get("unknown_clear_value", 0.0)),
            "footprint_enabled": bool(costmap.get("footprint_enabled", False)),
            "footprint_radius": float(costmap.get("footprint_radius", 0.0)),
            "footprint_safety_margin": float(costmap.get("footprint_safety_margin", 0.0)),
            "footprint_sample_count": int(costmap.get("footprint_sample_count", 8)),
        }

    def _local_costmap_cost_batch_torch(
        self,
        states: torch.Tensor,
        initial_state: torch.Tensor | np.ndarray,
        costmap: dict | None,
    ) -> torch.Tensor:
        costs = torch.zeros(states.shape[0], dtype=torch.float32, device=self.torch_device)
        if not costmap or not bool(costmap.get("enabled", False)):
            return costs
        width = int(costmap.get("width", 0))
        height = int(costmap.get("height", 0))
        resolution = float(costmap.get("resolution", 0.0))
        data = costmap.get("data")
        if width <= 0 or height <= 0 or resolution <= 0.0 or data is None or int(data.numel()) != width * height:
            return costs
        origin = costmap["origin"]
        points = self._local_costmap_sample_points_torch(states, costmap)
        ix = torch.floor((points[..., 0] - origin[0]) / resolution).to(torch.long)
        iy = torch.floor((points[..., 1] - origin[1]) / resolution).to(torch.long)
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        sampled = torch.full(
            points.shape[:-1],
            float(costmap.get("unknown_cost", 100.0)),
            dtype=torch.float32,
            device=self.torch_device,
        )
        sampled_unknown = torch.ones(points.shape[:-1], dtype=torch.bool, device=self.torch_device)
        if bool(torch.any(valid).item()):
            flat_idx = iy[valid] * width + ix[valid]
            sampled[valid] = data[flat_idx]
            unknown_mask = costmap.get("unknown_mask")
            if unknown_mask is not None and int(unknown_mask.numel()) == int(data.numel()):
                sampled_unknown[valid] = unknown_mask[flat_idx]
            else:
                sampled_unknown[valid] = False
        clear_radius = float(costmap.get("unknown_clear_radius", 0.0))
        if clear_radius > 0.0 and bool(torch.any(sampled_unknown).item()):
            start_xy = torch.as_tensor(initial_state, dtype=torch.float32, device=self.torch_device).reshape(6)[:2]
            distance_from_start = torch.linalg.norm(points - start_xy, dim=-1)
            clear_mask = sampled_unknown & (distance_from_start <= clear_radius)
            if bool(torch.any(clear_mask).item()):
                sampled[clear_mask] = float(costmap.get("unknown_clear_value", 0.0))
        max_cost = max(float(costmap.get("max_cost", 100.0)), 1e-6)
        normalized = torch.clamp(sampled, min=0.0, max=max_cost) / max_cost
        terms = torch.pow(normalized, max(float(costmap.get("power", 2.0)), 0.1))
        if terms.ndim == 3:
            terms = torch.max(terms, dim=2).values
        return float(costmap.get("weight", 0.0)) * torch.sum(terms, dim=1)

    def _local_costmap_sample_points_torch(self, states: torch.Tensor, costmap: dict) -> torch.Tensor:
        points = states[:, :, :2]
        if not bool(costmap.get("footprint_enabled", False)):
            return points
        radius = float(costmap.get("footprint_radius", 0.0)) + float(costmap.get("footprint_safety_margin", 0.0))
        if radius <= 0.0:
            return points
        sample_count = max(int(costmap.get("footprint_sample_count", 8)), 4)
        angles = torch.linspace(
            0.0,
            2.0 * torch.pi,
            sample_count + 1,
            dtype=torch.float32,
            device=self.torch_device,
        )[:-1]
        unit_offsets = torch.cat(
            [
                torch.zeros((1, 2), dtype=torch.float32, device=self.torch_device),
                torch.stack([torch.cos(angles), torch.sin(angles)], dim=1),
            ],
            dim=0,
        )
        offsets = unit_offsets * radius
        theta = states[:, :, 2]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        ox = offsets[:, 0]
        oy = offsets[:, 1]
        world_x = points[:, :, 0, None] + cos_t[:, :, None] * ox - sin_t[:, :, None] * oy
        world_y = points[:, :, 1, None] + sin_t[:, :, None] * ox + cos_t[:, :, None] * oy
        return torch.stack([world_x, world_y], dim=-1)

    def _path_tracking_cost_batch_torch(self, states: torch.Tensor, path: torch.Tensor | None) -> torch.Tensor:
        costs = torch.zeros(states.shape[0], dtype=torch.float32, device=self.torch_device)
        if self.path_tracking_weight <= 0.0 or path is None or path.numel() < 4:
            return costs
        path = path.reshape(-1, 2).to(dtype=torch.float32, device=self.torch_device)
        if path.shape[0] < 2:
            return costs
        points = states[:, :, :2]
        min_sq = torch.full(points.shape[:2], float("inf"), dtype=torch.float32, device=self.torch_device)
        for idx in range(path.shape[0] - 1):
            start = path[idx]
            end = path[idx + 1]
            segment = end - start
            denom = torch.dot(segment, segment)
            rel = points - start
            if float(denom.detach().cpu()) <= 1e-9:
                diff = rel
            else:
                t = torch.clamp(torch.sum(rel * segment, dim=2) / denom, 0.0, 1.0)
                projection = start + t[:, :, None] * segment
                diff = points - projection
            min_sq = torch.minimum(min_sq, torch.sum(diff * diff, dim=2))
        distance = torch.sqrt(torch.clamp(min_sq, min=0.0))
        excess = torch.clamp(distance - self.path_tracking_tolerance, min=0.0)
        return self.path_tracking_weight * torch.sum(excess * excess, dim=1)

    def _path_progress_cost_batch_torch(
        self,
        initial_state: torch.Tensor,
        final_states: torch.Tensor,
        path: torch.Tensor | None,
    ) -> torch.Tensor:
        costs = torch.zeros(final_states.shape[0], dtype=torch.float32, device=self.torch_device)
        if self.path_progress_weight <= 0.0 or path is None or path.numel() < 4:
            return costs
        path = path.reshape(-1, 2).to(dtype=torch.float32, device=self.torch_device)
        if path.shape[0] < 2:
            return costs
        initial_state_t = torch.as_tensor(initial_state, dtype=torch.float32, device=self.torch_device).reshape(1, -1)
        start_progress = self._path_progress_values_torch(initial_state_t[:, :2], path)[0]
        final_progress = self._path_progress_values_torch(final_states[:, :2], path)
        return -self.path_progress_weight * (final_progress - start_progress)

    def _path_progress_values_torch(self, points: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        points = points.reshape(-1, 2).to(dtype=torch.float32, device=self.torch_device)
        segment_lengths = torch.linalg.norm(path[1:] - path[:-1], dim=1)
        cumulative = torch.cat(
            [
                torch.zeros(1, dtype=torch.float32, device=self.torch_device),
                torch.cumsum(segment_lengths, dim=0),
            ]
        )
        best_sq = torch.full((points.shape[0],), float("inf"), dtype=torch.float32, device=self.torch_device)
        best_progress = torch.zeros(points.shape[0], dtype=torch.float32, device=self.torch_device)
        for idx in range(path.shape[0] - 1):
            start = path[idx]
            end = path[idx + 1]
            segment = end - start
            denom = torch.dot(segment, segment)
            rel = points - start
            if float(denom.detach().cpu()) <= 1e-9:
                projection = start.expand_as(points)
                t = torch.zeros(points.shape[0], dtype=torch.float32, device=self.torch_device)
            else:
                t = torch.clamp(torch.sum(rel * segment, dim=1) / denom, 0.0, 1.0)
                projection = start + t[:, None] * segment
            sq = torch.sum((points - projection) ** 2, dim=1)
            update = sq < best_sq
            best_sq = torch.where(update, sq, best_sq)
            best_progress = torch.where(update, cumulative[idx] + t * segment_lengths[idx], best_progress)
        return best_progress

    def _goal_progress_cost_batch_torch(
        self,
        initial_state: torch.Tensor | np.ndarray,
        final_states: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        costs = torch.zeros(final_states.shape[0], dtype=torch.float32, device=self.torch_device)
        if self.goal_progress_weight <= 0.0:
            return costs
        initial_state_t = torch.as_tensor(initial_state, dtype=torch.float32, device=self.torch_device).reshape(1, -1)
        start_xy = initial_state_t[:, :2]
        goal_xy = goal[:2].reshape(1, 2).to(dtype=torch.float32, device=self.torch_device)
        start_distance = torch.linalg.norm(goal_xy - start_xy, dim=1)[0]
        final_distance = torch.linalg.norm(goal_xy - final_states[:, :2], dim=1)
        return -self.goal_progress_weight * (start_distance - final_distance)

    def _heading_to_goal_cost_batch_torch(self, states: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        costs = torch.zeros(states.shape[0], dtype=torch.float32, device=self.torch_device)
        if self.heading_to_goal_weight <= 0.0:
            return costs
        goal_xy = goal[:2].reshape(1, 1, 2).to(dtype=torch.float32, device=self.torch_device)
        vectors = goal_xy - states[:, :, :2]
        distances = torch.linalg.norm(vectors, dim=2)
        active = distances > self.heading_to_goal_min_distance
        if not bool(torch.any(active).item()):
            return costs
        target_yaw = torch.atan2(vectors[:, :, 1], vectors[:, :, 0])
        heading_error = self._angle_diff_torch(states[:, :, 2], target_yaw)
        terms = torch.where(active, heading_error * heading_error, torch.zeros_like(heading_error))
        return self.heading_to_goal_weight * torch.sum(terms, dim=1)

    def _terrain_risk_cost_batch_torch(self, states: torch.Tensor) -> torch.Tensor:
        costs = torch.zeros(states.shape[0], dtype=torch.float32, device=self.torch_device)
        if self.terrain_risk_weight <= 0.0 or self.terrain_risk_mode == "none" or not self.terrain.enabled:
            return costs
        num_samples, horizon_steps, state_dim = states.shape
        flat_states = states.reshape(num_samples * horizon_steps, state_dim)
        _features, risks = self._terrain_features_torch(flat_states)
        risks = risks.reshape(num_samples, horizon_steps)
        terms = self._terrain_risk_terms_torch(risks)
        return self.terrain_risk_weight * torch.sum(terms, dim=1)

    def _terrain_risk_terms_torch(self, risks: torch.Tensor) -> torch.Tensor:
        mode = str(self.terrain_risk_mode).lower()
        if mode == "cumulative":
            values = torch.clamp(risks, min=0.0)
        elif mode == "excess":
            values = torch.clamp(risks - self.terrain_risk_threshold, min=0.0)
        elif mode == "none":
            return torch.zeros_like(risks, dtype=torch.float32, device=self.torch_device)
        else:
            raise ValueError(f"Unsupported terrain_risk_mode: {self.terrain_risk_mode}")
        return torch.pow(values, self.terrain_risk_power).to(torch.float32)

    def _terrain_features_torch(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = states[:, 0]
        y = states[:, 1]
        if not self.terrain.enabled:
            features = torch.zeros((states.shape[0], 4), dtype=torch.float32, device=self.torch_device)
            risks = torch.zeros(states.shape[0], dtype=torch.float32, device=self.torch_device)
            return features, risks
        slope_f = self.terrain.slope_scale * (torch.sin(self.terrain.slope_wave * x) + 0.35 * torch.cos(0.5 * y))
        slope_l = self.terrain.slope_scale * (torch.cos(self.terrain.slope_wave * y) + 0.35 * torch.sin(0.5 * x))
        roughness = self.terrain.roughness_scale * (0.5 + 0.5 * torch.sin(self.terrain.roughness_wave * (x + y)))
        friction = self.terrain.friction_base - self.terrain.friction_slope_scale * (
            0.5 * (torch.abs(slope_f) + torch.abs(slope_l))
        )
        friction = friction - self.terrain.friction_roughness_scale * roughness
        if self.noise_grid_t is not None:
            noise, grad_x, grad_y = self._bilinear_sample_many_torch(self.noise_fields_t, x, y)
            roughness = roughness + self.terrain.noise_roughness_weight * noise
            friction = friction - self.terrain.noise_friction_weight * noise
            slope_f = slope_f + self.terrain.noise_slope_weight * grad_x
            slope_l = slope_l + self.terrain.noise_slope_weight * grad_y
        roughness = torch.clamp(roughness, 0.0, 1.0)
        friction = torch.clamp(friction, 0.2, 1.0)
        attenuation = self._goal_relief_attenuation_torch(x, y)
        slope_f = slope_f * attenuation
        slope_l = slope_l * attenuation
        roughness = roughness * attenuation
        friction = 1.0 - attenuation * (1.0 - friction)
        slope_f, slope_l, roughness, friction = self._apply_terrain_patches_torch(
            x,
            y,
            slope_f,
            slope_l,
            roughness,
            friction,
        )
        features = torch.stack([slope_f, slope_l, roughness, friction], dim=1).to(torch.float32)
        w0, w1, w2, w3 = self.terrain.risk_weights
        risks = w0 * torch.abs(slope_f) + w1 * torch.abs(slope_l) + w2 * roughness + w3 * (1.0 - friction)
        return features, risks.to(torch.float32)

    def _apply_terrain_patches_torch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        slope_f: torch.Tensor,
        slope_l: torch.Tensor,
        roughness: torch.Tensor,
        friction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for patch in getattr(self.terrain, "patches", []):
            influence = self._terrain_patch_influence_torch(patch, x, y)
            slope_f = slope_f + influence * float(patch.get("slope_f_delta", 0.0))
            slope_l = slope_l + influence * float(patch.get("slope_l_delta", 0.0))
            roughness = roughness + influence * float(patch.get("roughness_delta", 0.0))
            friction = friction + influence * float(patch.get("friction_delta", 0.0))
        roughness = torch.clamp(roughness, 0.0, 1.0)
        friction = torch.clamp(friction, 0.2, 1.0)
        return slope_f, slope_l, roughness, friction

    def _terrain_patch_influence_torch(self, patch: dict, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        local_x, local_y = self._terrain_patch_local_xy_torch(patch, x, y)
        size = patch["size"]
        patch_type = str(patch.get("type", "ellipse")).lower()
        if patch_type == "ellipse":
            half_x = max(1e-6, 0.5 * float(size[0]))
            half_y = max(1e-6, 0.5 * float(size[1]))
            normalized_distance = torch.sqrt((local_x / half_x) ** 2 + (local_y / half_y) ** 2)
            outside_distance = (normalized_distance - 1.0) * min(half_x, half_y)
            return self._smooth_patch_influence_torch(outside_distance, float(patch["edge_width"]))
        if patch_type == "band":
            half_length = max(1e-6, 0.5 * float(size[0]))
            half_width = max(1e-6, 0.5 * float(size[1]))
            outside_distance = torch.maximum(torch.abs(local_x) - half_length, torch.abs(local_y) - half_width)
            return self._smooth_patch_influence_torch(outside_distance, float(patch["edge_width"]))
        raise ValueError(f"Unsupported terrain patch type: {patch_type}")

    def _terrain_patch_local_xy_torch(
        self,
        patch: dict,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dx = x - float(patch["center"][0])
        dy = y - float(patch["center"][1])
        angle = float(np.deg2rad(float(patch.get("angle", 0.0))))
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        return cos_a * dx + sin_a * dy, -sin_a * dx + cos_a * dy

    def _smooth_patch_influence_torch(self, outside_distance: torch.Tensor, edge_width: float) -> torch.Tensor:
        if edge_width <= 1e-6:
            return torch.where(
                outside_distance <= 0.0,
                torch.ones_like(outside_distance, dtype=torch.float32, device=self.torch_device),
                torch.zeros_like(outside_distance, dtype=torch.float32, device=self.torch_device),
            )
        t = torch.clamp(outside_distance / edge_width, 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        influence = 1.0 - smooth
        return torch.where(outside_distance <= 0.0, torch.ones_like(influence), influence).to(torch.float32)

    def _bilinear_sample_torch(self, grid: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_min, x_max = self.terrain.noise_x_range
        y_min, y_max = self.terrain.noise_y_range
        cols = int(grid.shape[1])
        rows = int(grid.shape[0])
        gx = torch.clamp((x - x_min) / max(1e-6, x_max - x_min), 0.0, 1.0) * (cols - 1)
        gy = torch.clamp((y - y_min) / max(1e-6, y_max - y_min), 0.0, 1.0) * (rows - 1)
        x0 = torch.floor(gx).to(torch.long)
        y0 = torch.floor(gy).to(torch.long)
        x1 = torch.clamp(x0 + 1, max=cols - 1)
        y1 = torch.clamp(y0 + 1, max=rows - 1)
        tx = (gx - x0.to(torch.float32)).to(torch.float32)
        ty = (gy - y0.to(torch.float32)).to(torch.float32)
        top = (1.0 - tx) * grid[y0, x0] + tx * grid[y0, x1]
        bottom = (1.0 - tx) * grid[y1, x0] + tx * grid[y1, x1]
        return ((1.0 - ty) * top + ty * bottom).to(torch.float32)

    def _bilinear_sample_many_torch(self, grids: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_min, x_max = self.terrain.noise_x_range
        y_min, y_max = self.terrain.noise_y_range
        cols = int(grids.shape[2])
        rows = int(grids.shape[1])
        gx = torch.clamp((x - x_min) / max(1e-6, x_max - x_min), 0.0, 1.0) * (cols - 1)
        gy = torch.clamp((y - y_min) / max(1e-6, y_max - y_min), 0.0, 1.0) * (rows - 1)
        x0 = torch.floor(gx).to(torch.long)
        y0 = torch.floor(gy).to(torch.long)
        x1 = torch.clamp(x0 + 1, max=cols - 1)
        y1 = torch.clamp(y0 + 1, max=rows - 1)
        tx = (gx - x0.to(torch.float32)).to(torch.float32).unsqueeze(0)
        ty = (gy - y0.to(torch.float32)).to(torch.float32).unsqueeze(0)
        top = (1.0 - tx) * grids[:, y0, x0] + tx * grids[:, y0, x1]
        bottom = (1.0 - tx) * grids[:, y1, x0] + tx * grids[:, y1, x1]
        return ((1.0 - ty) * top + ty * bottom).to(torch.float32)

    def _goal_relief_attenuation_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        relief = self.terrain.goal_relief
        if not bool(relief.get("enabled", False)):
            return torch.ones_like(x, dtype=torch.float32, device=self.torch_device)
        center = relief.get("center", [0.0, 0.0])
        sigma = relief.get("sigma", [2.0, 1.2])
        sigma_x = max(1e-6, float(sigma[0]))
        sigma_y = max(1e-6, float(sigma[1]))
        strength = float(np.clip(relief.get("strength", 0.75), 0.0, 1.0))
        floor = float(np.clip(relief.get("floor", 0.25), 0.0, 1.0))
        dx = (x - float(center[0])) / sigma_x
        dy = (y - float(center[1])) / sigma_y
        gaussian = torch.exp(-0.5 * (dx * dx + dy * dy))
        return torch.clamp(1.0 - strength * gaussian, min=floor)

    def _setup_terrain_tensors(self) -> None:
        if getattr(self.terrain, "_noise_grid", None) is None:
            self.noise_grid_t = None
            self.noise_grad_x_t = None
            self.noise_grad_y_t = None
            self.noise_fields_t = None
            return
        self.noise_grid_t = torch.as_tensor(self.terrain._noise_grid, dtype=torch.float32, device=self.torch_device)
        self.noise_grad_x_t = torch.as_tensor(
            self.terrain._noise_grad_x,
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.noise_grad_y_t = torch.as_tensor(
            self.terrain._noise_grad_y,
            dtype=torch.float32,
            device=self.torch_device,
        )
        self.noise_fields_t = torch.stack(
            [self.noise_grid_t, self.noise_grad_x_t, self.noise_grad_y_t],
            dim=0,
        )

    @staticmethod
    def _angle_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.remainder(a - b + torch.pi, 2.0 * torch.pi) - torch.pi

    def reset_profile(self) -> None:
        self._profile_totals_ms = {}
        self._profile_counts = {}
        self._profile_total_calls = 0

    def profile_summary(self) -> dict:
        totals = {key: float(value) for key, value in sorted(self._profile_totals_ms.items())}
        means = {
            key: float(totals[key] / max(1, self._profile_counts.get(key, 0)))
            for key in totals
        }
        return {
            "enabled": self.profile_enabled,
            "total_calls": int(self._profile_total_calls),
            "totals_ms": totals,
            "means_ms": means,
            "counts": {key: int(value) for key, value in sorted(self._profile_counts.items())},
        }

    def _profile_start(self) -> float | None:
        if not self.profile_enabled:
            return None
        self._profile_sync()
        return time.perf_counter()

    def _profile_stop(self, bucket: str, started_at: float | None) -> None:
        if started_at is None:
            return
        self._profile_sync()
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        self._profile_totals_ms[bucket] = self._profile_totals_ms.get(bucket, 0.0) + float(elapsed_ms)
        self._profile_counts[bucket] = self._profile_counts.get(bucket, 0) + 1

    def _profile_sync(self) -> None:
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize(self.torch_device)
