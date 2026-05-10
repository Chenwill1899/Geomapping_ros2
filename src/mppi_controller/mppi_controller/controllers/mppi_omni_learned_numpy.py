"""NumPy MPPI controller using learned residual FDM rollout dynamics."""

from __future__ import annotations

import numpy as np

from mppi_controller.controllers.mppi_omni_numpy import MppiOmniNumpy
from mppi_controller.core.learned_residual_dynamics import LearnedResidualDynamics
from mppi_controller.core.omni_b2 import OmniB2
from mppi_controller.core.terrain import TerrainField


class LearnedFdmMppiOmniNumpy(MppiOmniNumpy):
    def __init__(
        self,
        *args,
        learned_dynamics: LearnedResidualDynamics,
        residual_gain: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.learned_dynamics = learned_dynamics
        self.residual_gain = float(residual_gain)

    @classmethod
    def from_config(
        cls,
        config: dict,
        seed: int | None = None,
        *,
        learned_dynamics: LearnedResidualDynamics | None = None,
        **overrides,
    ) -> "LearnedFdmMppiOmniNumpy":
        sim = config["simulation"]
        mppi = config["mppi"]
        robot = config["robot"]
        sampling_rate = float(sim["sampling_rate"])
        dt = 1.0 / sampling_rate
        horizon_steps = int(float(sim["time_horizon"]) * sampling_rate)
        terrain = TerrainField.from_config(config.get("terrain"))
        if learned_dynamics is None:
            dynamics_robot = OmniB2(
                dt,
                float(robot["max_vx"]),
                float(robot["max_vy"]),
                float(robot["max_wz"]),
            )
            fdm = config.get("fdm", {})
            learned_dynamics = LearnedResidualDynamics.from_artifacts(
                fdm["model_dir"],
                robot=dynamics_robot,
                terrain=terrain,
                checkpoint=fdm.get("checkpoint", "best_model.pt"),
                normalization=fdm.get("normalization", "normalization.npz"),
                device=fdm.get("device", "cpu"),
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
            residual_gain=float(overrides.get("residual_gain", config.get("fdm", {}).get("residual_gain", 1.0))),
        )

    def _rollout_batch(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        return_controls: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        controls = np.clip(np.asarray(controls, dtype=np.float32), -self.max_control, self.max_control)
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
            response_command = np.clip(prev_real + delta, -self.max_control, self.max_control)
            residual = self.learned_dynamics.predict_residual_batch(prev, response_command)
            control = np.clip(response_command + self.residual_gain * residual, -self.max_control, self.max_control)
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
