"""NumPy MPPI controller for the B2 omnidirectional SE(2) nominal model."""

from __future__ import annotations

import numpy as np

from mppi_controller.core.omni_b2 import OmniB2
from mppi_controller.core.terrain import TerrainField


class MppiOmniNumpy:
    def __init__(
        self,
        *,
        dt: float,
        horizon_steps: int,
        num_samples: int,
        lambda_: float,
        noise_std: np.ndarray,
        max_vx: float,
        max_vy: float,
        max_wz: float,
        min_vx: float | None = None,
        goal_xy_weight: float = 5.0,
        yaw_weight: float = 0.2,
        control_weight: float = 0.01,
        backward_weight: float = 0.0,
        smooth_weight: float = 0.2,
        obstacle_weight: float = 25.0,
        obstacle_soft_weight: float = 0.0,
        obstacle_influence_dist: float = 0.0,
        max_ax: float = 1000.0,
        max_ay: float = 1000.0,
        max_awz: float = 1000.0,
        velocity_lag_beta: float = 0.0,
        lateral_weight: float = 0.0,
        yaw_rate_weight: float = 0.0,
        accel_weight: float = 0.0,
        jerk_weight: float = 0.0,
        path_tracking_weight: float = 0.0,
        path_tracking_tolerance: float = 0.3,
        path_progress_weight: float = 0.0,
        goal_progress_weight: float = 0.0,
        heading_to_goal_weight: float = 0.0,
        heading_to_goal_min_distance: float = 0.3,
        update_smoothing_alpha: float = 0.0,
        goal_change_reset_distance: float = 0.75,
        goal_change_reset_yaw: float = 1.0,
        terrain: TerrainField | None = None,
        terrain_risk_weight: float = 0.0,
        terrain_risk_power: float = 2.0,
        terrain_risk_threshold: float = 0.0,
        terrain_risk_mode: str = "excess",
        robot_radius: float = 0.6,
        safety_dist: float = 0.3,
        draw_num_traj: int = 50,
        seed: int | None = None,
    ) -> None:
        self.dt = float(dt)
        self.horizon_steps = int(horizon_steps)
        self.num_samples = int(num_samples)
        self.lambda_ = float(lambda_)
        self.noise_std = np.asarray(noise_std, dtype=np.float32)
        self.max_control = np.asarray([max_vx, max_vy, max_wz], dtype=np.float32)
        self.min_control = np.asarray(
            [min_vx if min_vx is not None else -max_vx, -max_vy, -max_wz], dtype=np.float32
        )
        self.goal_xy_weight = float(goal_xy_weight)
        self.yaw_weight = float(yaw_weight)
        self.control_weight = float(control_weight)
        self.backward_weight = float(backward_weight)
        self.smooth_weight = float(smooth_weight)
        self.obstacle_weight = float(obstacle_weight)
        self.obstacle_soft_weight = float(obstacle_soft_weight)
        self.obstacle_influence_dist = float(obstacle_influence_dist)
        self.max_accel = np.asarray([max_ax, max_ay, max_awz], dtype=np.float32)
        self.velocity_lag_beta = float(np.clip(velocity_lag_beta, 0.0, 1.0))
        self.lateral_weight = float(lateral_weight)
        self.yaw_rate_weight = float(yaw_rate_weight)
        self.accel_weight = float(accel_weight)
        self.jerk_weight = float(jerk_weight)
        self.path_tracking_weight = float(path_tracking_weight)
        self.path_tracking_tolerance = float(path_tracking_tolerance)
        self.path_progress_weight = float(path_progress_weight)
        self.goal_progress_weight = float(goal_progress_weight)
        self.heading_to_goal_weight = float(heading_to_goal_weight)
        self.heading_to_goal_min_distance = float(max(heading_to_goal_min_distance, 0.0))
        self.update_smoothing_alpha = self._control_axis_alpha(update_smoothing_alpha)
        self.goal_change_reset_distance = float(max(goal_change_reset_distance, 0.0))
        self.goal_change_reset_yaw = float(max(goal_change_reset_yaw, 0.0))
        self.terrain = terrain if terrain is not None else TerrainField()
        self.terrain_risk_weight = float(terrain_risk_weight)
        self.terrain_risk_power = float(terrain_risk_power)
        self.terrain_risk_threshold = float(terrain_risk_threshold)
        self.terrain_risk_mode = str(terrain_risk_mode).lower()
        self.robot_radius = float(robot_radius)
        self.safety_dist = float(safety_dist)
        self.draw_num_traj = min(int(draw_num_traj), self.num_samples)
        self.rng = np.random.default_rng(seed)
        self.nominal_u = np.zeros((self.horizon_steps, 3), dtype=np.float32)
        self.previous_control = np.zeros(3, dtype=np.float32)
        self._has_nominal_update = False
        self._last_goal_for_nominal_reset: np.ndarray | None = None
        self.model = OmniB2(self.dt, max_vx, max_vy, max_wz)

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        **overrides,
    ) -> "MppiOmniNumpy":
        sim = config["simulation"]
        mppi = config["mppi"]
        robot = config["robot"]
        terrain = TerrainField.from_config(config.get("terrain"))
        sampling_rate = float(sim["sampling_rate"])
        dt = 1.0 / sampling_rate
        horizon_steps = int(float(sim["time_horizon"]) * sampling_rate)
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
            backward_weight=float(overrides.get("backward_weight", mppi.get("backward_weight", 0.0))),
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
        )

    def compute_control(self, state: np.ndarray, cost_params):
        goal = np.asarray(cost_params[3], dtype=np.float32)
        progress_goal = self._progress_goal_from_cost_params(cost_params, goal)
        self._reset_nominal_on_goal_change(goal)
        obstacles = np.asarray(cost_params[4], dtype=np.float32).reshape(-1, 7)
        path = self._path_from_cost_params(cost_params)
        costmap = self._costmap_from_cost_params(cost_params)
        previous_nominal = self.nominal_u.copy()
        noise = self.rng.normal(
            loc=0.0,
            scale=self.noise_std,
            size=(self.num_samples, self.horizon_steps, 3),
        ).astype(np.float32)
        candidates = np.clip(
            self.nominal_u[None, :, :] + noise,
            self.min_control,
            self.max_control,
        )
        costs = self.trajectory_cost_batch(
            state,
            candidates,
            goal,
            obstacles,
            path,
            costmap=costmap,
            progress_goal=progress_goal,
        )
        min_cost = float(np.min(costs))
        weights = np.exp(-(costs - min_cost) / max(self.lambda_, 1e-6))
        normalizer = float(np.sum(weights))
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            weights = np.full(self.num_samples, 1.0 / self.num_samples, dtype=np.float32)
            normalizer = 1.0
        else:
            weights = weights / normalizer
        updated_nominal = np.tensordot(weights, candidates, axes=(0, 0)).astype(np.float32)
        self.nominal_u = self._smooth_nominal_update(updated_nominal, previous_nominal)
        self.nominal_u = np.clip(self.nominal_u, self.min_control, self.max_control)
        command = self.nominal_u[0].copy()
        control = self._apply_velocity_response(state, command).astype(np.float32)
        optimal_u = self.nominal_u.copy()
        sample_u = candidates[: self.draw_num_traj].copy()
        self.previous_control = command.copy()
        self._shift_nominal_controls()
        return control, optimal_u, sample_u, normalizer, min_cost

    def _reset_nominal_on_goal_change(self, goal: np.ndarray) -> None:
        goal_pose = np.asarray(goal, dtype=np.float32).reshape(-1)[:3].copy()
        if goal_pose.shape[0] < 3:
            return
        previous_goal = self._last_goal_for_nominal_reset
        self._last_goal_for_nominal_reset = goal_pose
        if previous_goal is None:
            return
        xy_delta = float(np.linalg.norm(goal_pose[:2] - previous_goal[:2]))
        yaw_delta = abs(self._angle_diff(float(goal_pose[2]), float(previous_goal[2])))
        should_reset = (
            self.goal_change_reset_distance > 0.0
            and xy_delta > self.goal_change_reset_distance
        ) or (
            self.goal_change_reset_yaw > 0.0
            and yaw_delta > self.goal_change_reset_yaw
        )
        if should_reset:
            self.nominal_u.fill(0.0)
            self._has_nominal_update = False

    def _smooth_nominal_update(self, updated_nominal: np.ndarray, previous_nominal: np.ndarray) -> np.ndarray:
        updated = np.asarray(updated_nominal, dtype=np.float32)
        if not self._has_nominal_update or not np.any(self.update_smoothing_alpha > 0.0):
            self._has_nominal_update = True
            return updated
        alpha = self.update_smoothing_alpha.reshape(1, 3)
        self._has_nominal_update = True
        return (alpha * np.asarray(previous_nominal, dtype=np.float32) + (1.0 - alpha) * updated).astype(np.float32)

    @staticmethod
    def _control_axis_alpha(value: float | list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        alpha = np.asarray(value, dtype=np.float32).reshape(-1)
        if alpha.size == 1:
            alpha = np.full(3, float(alpha[0]), dtype=np.float32)
        elif alpha.size != 3:
            raise ValueError("update_smoothing_alpha must be a scalar or a three-element control-axis list")
        return np.clip(alpha, 0.0, 0.95).astype(np.float32)

    def trajectory_cost(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        path: np.ndarray | None = None,
        costmap: dict | None = None,
        progress_goal: np.ndarray | None = None,
    ) -> float:
        states, real_controls = self._rollout_batch(
            initial_state,
            controls[None, :, :],
            return_controls=True,
        )
        states = states[0]
        real_controls = real_controls[0]
        final_state = states[-1]
        xy_error = final_state[:2] - goal[:2]
        yaw_error = self._angle_diff(float(final_state[2]), float(goal[2]))
        goal_cost = self.goal_xy_weight * float(np.dot(xy_error, xy_error))
        yaw_cost = self.yaw_weight * yaw_error * yaw_error
        control_cost = self.control_weight * float(np.sum(controls * controls))
        smooth_cost = self.smooth_weight * float(
            np.sum(np.diff(np.vstack([self.previous_control, controls]), axis=0) ** 2)
        )
        accel = np.diff(np.vstack([np.asarray(initial_state, dtype=np.float32)[3:], real_controls]), axis=0) / self.dt
        accel_cost = self.accel_weight * float(np.sum(accel * accel))
        jerk = np.diff(np.vstack([np.asarray(initial_state, dtype=np.float32)[3:], real_controls]), n=2, axis=0)
        jerk_cost = self.jerk_weight * float(np.sum(jerk * jerk))
        lateral_cost = self.lateral_weight * float(np.sum(real_controls[:, 1] * real_controls[:, 1]))
        yaw_rate_cost = self.yaw_rate_weight * float(np.sum(real_controls[:, 2] * real_controls[:, 2]))
        obstacle_cost = self._obstacle_cost(states[1:], obstacles)
        path_tracking_cost = self._path_tracking_cost(states[1:], path)
        path_progress_cost = float(self._path_progress_cost_batch(initial_state, final_state[None, :], path)[0])
        progress_target = goal if progress_goal is None else progress_goal
        goal_progress_cost = float(self._goal_progress_cost_batch(initial_state, final_state[None, :], progress_target)[0])
        heading_to_goal_cost = float(self._heading_to_goal_cost_batch(states[None, 1:, :], progress_target)[0])
        terrain_risk_cost = self._terrain_risk_cost(states[1:])
        local_costmap_cost = float(self._local_costmap_cost_batch(initial_state, states[None, 1:, :], costmap)[0])
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
        )

    def trajectory_cost_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        path: np.ndarray | None = None,
        costmap: dict | None = None,
        progress_goal: np.ndarray | None = None,
    ) -> np.ndarray:
        states, real_controls = self._rollout_batch(initial_state, controls, return_controls=True)
        final_states = states[:, -1, :]
        xy_error = final_states[:, :2] - goal[:2]
        yaw_error = self._angle_diff_array(final_states[:, 2], float(goal[2]))
        goal_cost = self.goal_xy_weight * np.sum(xy_error * xy_error, axis=1)
        yaw_cost = self.yaw_weight * yaw_error * yaw_error
        control_cost = self.control_weight * np.sum(controls * controls, axis=(1, 2))
        backward_cost = np.zeros(controls.shape[0], dtype=np.float32)
        if self.backward_weight > 0.0 and self.min_control[0] < 0.0:
            backward_vx = np.maximum(-controls[:, :, 0], 0.0)
            backward_cost = self.backward_weight * np.sum(backward_vx * backward_vx, axis=1)
        previous = np.broadcast_to(self.previous_control, (controls.shape[0], 1, 3))
        control_deltas = np.diff(np.concatenate([previous, controls], axis=1), axis=1)
        smooth_cost = self.smooth_weight * np.sum(control_deltas * control_deltas, axis=(1, 2))
        initial_velocity = np.broadcast_to(
            np.asarray(initial_state, dtype=np.float32)[3:],
            (controls.shape[0], 1, 3),
        )
        accel = np.diff(np.concatenate([initial_velocity, real_controls], axis=1), axis=1) / self.dt
        accel_cost = self.accel_weight * np.sum(accel * accel, axis=(1, 2))
        jerk = np.diff(np.concatenate([initial_velocity, real_controls], axis=1), n=2, axis=1)
        jerk_cost = self.jerk_weight * np.sum(jerk * jerk, axis=(1, 2))
        lateral_cost = self.lateral_weight * np.sum(real_controls[:, :, 1] * real_controls[:, :, 1], axis=1)
        yaw_rate_cost = self.yaw_rate_weight * np.sum(real_controls[:, :, 2] * real_controls[:, :, 2], axis=1)
        obstacle_cost = self._obstacle_cost_batch(states[:, 1:, :], obstacles)
        path_tracking_cost = self._path_tracking_cost_batch(states[:, 1:, :], path)
        path_progress_cost = self._path_progress_cost_batch(initial_state, states[:, -1, :], path)
        progress_target = goal if progress_goal is None else progress_goal
        goal_progress_cost = self._goal_progress_cost_batch(initial_state, states[:, -1, :], progress_target)
        heading_to_goal_cost = self._heading_to_goal_cost_batch(states[:, 1:, :], progress_target)
        terrain_risk_cost = self._terrain_risk_cost_batch(states[:, 1:, :])
        local_costmap_cost = self._local_costmap_cost_batch(initial_state, states[:, 1:, :], costmap)
        return (
            goal_cost
            + yaw_cost
            + control_cost
            + backward_cost
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
        ).astype(np.float32)

    def _rollout_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        return_controls: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        controls = np.clip(np.asarray(controls, dtype=np.float32), self.min_control, self.max_control)
        num_samples, horizon_steps, _ = controls.shape
        states = np.zeros((num_samples, horizon_steps + 1, 6), dtype=np.float32)
        states[:, 0, :] = np.asarray(initial_state, dtype=np.float32)
        real_controls = np.zeros((num_samples, horizon_steps, 3), dtype=np.float32)
        prev_real = np.broadcast_to(states[:, 0, 3:], (num_samples, 3)).copy()
        max_delta = self.max_accel * self.dt
        for step in range(horizon_steps):
            prev = states[:, step, :]
            command = controls[:, step, :]
            lagged = self.velocity_lag_beta * prev_real + (1.0 - self.velocity_lag_beta) * command
            delta = np.clip(lagged - prev_real, -max_delta, max_delta)
            control = np.clip(prev_real + delta, self.min_control, self.max_control)
            real_controls[:, step, :] = control
            theta = prev[:, 2]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
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
        if return_controls:
            return states, real_controls
        return states

    def _obstacle_cost(self, states: np.ndarray, obstacles: np.ndarray) -> float:
        if obstacles.size == 0:
            return 0.0
        total = 0.0
        for obstacle in obstacles:
            ox, oy, radius = obstacle[:3]
            clearance = (
                np.linalg.norm(states[:, :2] - np.array([ox, oy], dtype=np.float32), axis=1)
                - float(radius)
                - self.robot_radius
            )
            margin = self.safety_dist - clearance
            violations = margin[margin > 0.0]
            if violations.size:
                total += self.obstacle_weight * float(np.sum(violations * violations))
            if self.obstacle_soft_weight > 0.0 and self.obstacle_influence_dist > self.safety_dist:
                far_mask = (clearance > self.safety_dist) & (clearance < self.obstacle_influence_dist)
                if np.any(far_mask):
                    soft_margin = self.obstacle_influence_dist - clearance[far_mask]
                    total += self.obstacle_soft_weight * float(np.sum(soft_margin * soft_margin))
        return total

    def _obstacle_cost_batch(self, states: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        costs = np.zeros(states.shape[0], dtype=np.float32)
        if obstacles.size == 0:
            return costs
        for obstacle in obstacles:
            center = obstacle[:2].astype(np.float32)
            clearance = (
                np.linalg.norm(states[:, :, :2] - center[None, None, :], axis=2)
                - float(obstacle[2])
                - self.robot_radius
            )
            margin = np.maximum(self.safety_dist - clearance, 0.0)
            costs += self.obstacle_weight * np.sum(margin * margin, axis=1)
            if self.obstacle_soft_weight > 0.0 and self.obstacle_influence_dist > self.safety_dist:
                soft_margin = np.maximum(self.obstacle_influence_dist - clearance, 0.0)
                soft_margin = np.where(clearance > self.safety_dist, soft_margin, 0.0)
                costs += self.obstacle_soft_weight * np.sum(soft_margin * soft_margin, axis=1)
        return costs

    def _path_tracking_cost(self, states: np.ndarray, path: np.ndarray | None) -> float:
        return float(self._path_tracking_cost_batch(np.asarray(states, dtype=np.float32)[None, :, :], path)[0])

    def _path_tracking_cost_batch(self, states: np.ndarray, path: np.ndarray | None) -> np.ndarray:
        costs = np.zeros(states.shape[0], dtype=np.float32)
        if self.path_tracking_weight <= 0.0:
            return costs
        path_arr = np.asarray(path if path is not None else [], dtype=np.float32).reshape(-1, 2)
        if len(path_arr) < 2:
            return costs
        points = np.asarray(states, dtype=np.float32)[:, :, :2]
        min_sq = np.full(points.shape[:2], np.inf, dtype=np.float32)
        for start, end in zip(path_arr[:-1], path_arr[1:]):
            segment = end - start
            denom = float(np.dot(segment, segment))
            if denom <= 1e-9:
                diff = points - start
            else:
                rel = points - start
                t = np.clip(np.sum(rel * segment, axis=2) / denom, 0.0, 1.0)
                projection = start + t[:, :, None] * segment
                diff = points - projection
            min_sq = np.minimum(min_sq, np.sum(diff * diff, axis=2))
        distance = np.sqrt(np.maximum(min_sq, 0.0))
        excess = np.maximum(distance - self.path_tracking_tolerance, 0.0)
        return (self.path_tracking_weight * np.sum(excess * excess, axis=1)).astype(np.float32)

    def _path_progress_cost_batch(
        self,
        initial_state: np.ndarray,
        final_states: np.ndarray,
        path: np.ndarray | None,
    ) -> np.ndarray:
        final_states = np.asarray(final_states, dtype=np.float32).reshape(-1, 6)
        costs = np.zeros(final_states.shape[0], dtype=np.float32)
        if self.path_progress_weight <= 0.0:
            return costs
        path_arr = np.asarray(path if path is not None else [], dtype=np.float32).reshape(-1, 2)
        if len(path_arr) < 2:
            return costs
        start_progress = self._path_progress_values(
            np.asarray(initial_state, dtype=np.float32).reshape(6)[None, :2],
            path_arr,
        )[0]
        final_progress = self._path_progress_values(final_states[:, :2], path_arr)
        return (-self.path_progress_weight * (final_progress - start_progress)).astype(np.float32)

    @staticmethod
    def _path_progress_values(points: np.ndarray, path: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
        if len(path) < 2:
            return np.zeros(points.shape[0], dtype=np.float32)
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1).astype(np.float32)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)]).astype(np.float32)
        best_sq = np.full(points.shape[0], np.inf, dtype=np.float32)
        best_progress = np.zeros(points.shape[0], dtype=np.float32)
        for idx, (start, end) in enumerate(zip(path[:-1], path[1:])):
            segment = end - start
            denom = float(np.dot(segment, segment))
            if denom <= 1e-9:
                projection = np.broadcast_to(start, points.shape)
                t = np.zeros(points.shape[0], dtype=np.float32)
            else:
                rel = points - start
                t = np.clip(np.sum(rel * segment, axis=1) / denom, 0.0, 1.0).astype(np.float32)
                projection = start + t[:, None] * segment
            sq = np.sum((points - projection) ** 2, axis=1)
            update = sq < best_sq
            best_sq[update] = sq[update]
            best_progress[update] = cumulative[idx] + t[update] * segment_lengths[idx]
        return best_progress.astype(np.float32)

    def _goal_progress_cost_batch(
        self,
        initial_state: np.ndarray,
        final_states: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        final_states = np.asarray(final_states, dtype=np.float32).reshape(-1, 6)
        costs = np.zeros(final_states.shape[0], dtype=np.float32)
        if self.goal_progress_weight <= 0.0:
            return costs
        start_xy = np.asarray(initial_state, dtype=np.float32).reshape(6)[:2]
        goal_xy = np.asarray(goal, dtype=np.float32).reshape(-1)[:2]
        start_distance = float(np.linalg.norm(goal_xy - start_xy))
        final_distance = np.linalg.norm(goal_xy[None, :] - final_states[:, :2], axis=1)
        return (-self.goal_progress_weight * (start_distance - final_distance)).astype(np.float32)

    def _heading_to_goal_cost_batch(self, states: np.ndarray, goal: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        costs = np.zeros(states.shape[0], dtype=np.float32)
        if self.heading_to_goal_weight <= 0.0:
            return costs
        goal_xy = np.asarray(goal, dtype=np.float32).reshape(-1)[:2]
        vectors = goal_xy[None, None, :] - states[:, :, :2]
        distances = np.linalg.norm(vectors, axis=2)
        active = distances > self.heading_to_goal_min_distance
        if not np.any(active):
            return costs
        target_yaw = np.arctan2(vectors[:, :, 1], vectors[:, :, 0])
        heading_error = self._angle_diff_array(states[:, :, 2], target_yaw)
        terms = np.where(active, heading_error * heading_error, 0.0)
        return (self.heading_to_goal_weight * np.sum(terms, axis=1)).astype(np.float32)

    @staticmethod
    def _path_from_cost_params(cost_params) -> np.ndarray:
        if len(cost_params) <= 6 or cost_params[6] is None:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(cost_params[6], dtype=np.float32).reshape(-1, 2)

    @staticmethod
    def _costmap_from_cost_params(cost_params) -> dict | None:
        if len(cost_params) <= 7:
            return None
        candidate = cost_params[7]
        if not isinstance(candidate, dict) or not bool(candidate.get("enabled", False)):
            return None
        return candidate

    @staticmethod
    def _progress_goal_from_cost_params(cost_params, fallback: np.ndarray) -> np.ndarray:
        if len(cost_params) <= 8 or cost_params[8] is None:
            return np.asarray(fallback, dtype=np.float32)
        return np.asarray(cost_params[8], dtype=np.float32).reshape(-1)

    def _local_costmap_cost_batch(
        self,
        initial_state: np.ndarray,
        states: np.ndarray,
        costmap: dict | None,
    ) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        costs = np.zeros(states.shape[0], dtype=np.float32)
        if not costmap or not bool(costmap.get("enabled", False)):
            return costs
        width = int(costmap.get("width", 0))
        height = int(costmap.get("height", 0))
        resolution = float(costmap.get("resolution", 0.0))
        if width <= 0 or height <= 0 or resolution <= 0.0:
            return costs
        data = np.asarray(costmap.get("data", []), dtype=np.float32).reshape(-1)
        if data.size != width * height:
            return costs
        origin = np.asarray(costmap.get("origin", [0.0, 0.0]), dtype=np.float32).reshape(2)
        points = self._local_costmap_sample_points(states, costmap)
        ix = np.floor((points[..., 0] - origin[0]) / resolution).astype(np.int64)
        iy = np.floor((points[..., 1] - origin[1]) / resolution).astype(np.int64)
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        sampled = np.full(points.shape[:-1], float(costmap.get("unknown_cost", 100.0)), dtype=np.float32)
        sampled_unknown = np.ones(points.shape[:-1], dtype=bool)
        if np.any(valid):
            flat_idx = iy[valid] * width + ix[valid]
            sampled[valid] = data[flat_idx]
            unknown_mask = np.asarray(costmap.get("unknown_mask", np.zeros_like(data, dtype=bool)), dtype=bool).reshape(-1)
            if unknown_mask.size == data.size:
                sampled_unknown[valid] = unknown_mask[flat_idx]
            else:
                sampled_unknown[valid] = False
        clear_radius = float(costmap.get("unknown_clear_radius", 0.0))
        if clear_radius > 0.0 and np.any(sampled_unknown):
            start_xy = np.asarray(initial_state, dtype=np.float32).reshape(6)[:2]
            distance_from_start = np.linalg.norm(points - start_xy, axis=-1)
            clear_mask = sampled_unknown & (distance_from_start <= clear_radius)
            if np.any(clear_mask):
                sampled[clear_mask] = float(costmap.get("unknown_clear_value", 0.0))
        max_cost = max(float(costmap.get("max_cost", 100.0)), 1e-6)
        normalized = np.clip(sampled, 0.0, max_cost) / max_cost
        terms = np.power(normalized, max(float(costmap.get("power", 2.0)), 0.1))
        if terms.ndim == 3:
            terms = np.max(terms, axis=2)
        return (float(costmap.get("weight", 0.0)) * np.sum(terms, axis=1)).astype(np.float32)

    @staticmethod
    def _local_costmap_sample_points(states: np.ndarray, costmap: dict) -> np.ndarray:
        points = states[:, :, :2]
        if not bool(costmap.get("footprint_enabled", False)):
            return points
        radius = float(costmap.get("footprint_radius", 0.0)) + float(costmap.get("footprint_safety_margin", 0.0))
        if radius <= 0.0:
            return points
        sample_count = max(int(costmap.get("footprint_sample_count", 8)), 4)
        angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False, dtype=np.float32)
        unit_offsets = np.vstack(
            [
                np.zeros((1, 2), dtype=np.float32),
                np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32),
            ]
        )
        offsets = unit_offsets * np.float32(radius)
        theta = states[:, :, 2]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        ox = offsets[:, 0]
        oy = offsets[:, 1]
        world_x = points[:, :, 0, None] + cos_t[:, :, None] * ox - sin_t[:, :, None] * oy
        world_y = points[:, :, 1, None] + sin_t[:, :, None] * ox + cos_t[:, :, None] * oy
        return np.stack([world_x, world_y], axis=-1).astype(np.float32)

    def _terrain_risk_cost(self, states: np.ndarray) -> float:
        return float(self._terrain_risk_cost_batch(np.asarray(states, dtype=np.float32)[None, :, :])[0])

    def _terrain_risk_cost_batch(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        costs = np.zeros(states.shape[0], dtype=np.float32)
        if self.terrain_risk_weight <= 0.0 or self.terrain_risk_mode == "none" or not self.terrain.enabled:
            return costs
        risks = self._terrain_risk_values_batch(states)
        terms = self._terrain_risk_terms(risks)
        return (self.terrain_risk_weight * np.sum(terms, axis=1)).astype(np.float32)

    def _terrain_risk_values_batch(self, states: np.ndarray) -> np.ndarray:
        flat = np.asarray(states, dtype=np.float32).reshape(-1, states.shape[-1])
        risks = np.zeros(flat.shape[0], dtype=np.float32)
        for idx, state in enumerate(flat):
            features = self.terrain.feature(float(state[0]), float(state[1]))
            risks[idx] = self.terrain.risk_cost(float(state[0]), float(state[1]), features=features)
        return risks.reshape(states.shape[0], states.shape[1])

    def _terrain_risk_terms(self, risks: np.ndarray) -> np.ndarray:
        mode = str(self.terrain_risk_mode).lower()
        if mode == "cumulative":
            values = np.maximum(risks, 0.0)
        elif mode == "excess":
            values = np.maximum(risks - self.terrain_risk_threshold, 0.0)
        elif mode == "none":
            return np.zeros_like(risks, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported terrain_risk_mode: {self.terrain_risk_mode}")
        return np.power(values, self.terrain_risk_power).astype(np.float32)

    def _shift_nominal_controls(self) -> None:
        self.nominal_u[:-1] = self.nominal_u[1:]
        self.nominal_u[-1] = 0.0

    def _apply_velocity_response(self, state: np.ndarray, command: np.ndarray) -> np.ndarray:
        prev_real = np.asarray(state, dtype=np.float32)[3:]
        command = np.clip(np.asarray(command, dtype=np.float32), self.min_control, self.max_control)
        lagged = self.velocity_lag_beta * prev_real + (1.0 - self.velocity_lag_beta) * command
        delta = np.clip(lagged - prev_real, -self.max_accel * self.dt, self.max_accel * self.dt)
        return np.clip(prev_real + delta, self.min_control, self.max_control)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _angle_diff_array(a: np.ndarray, b: float) -> np.ndarray:
        return (a - b + np.pi) % (2.0 * np.pi) - np.pi
