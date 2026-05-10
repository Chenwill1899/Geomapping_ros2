"""Internal MPPI simulation runner used by the ROS 2 wrapper and tests."""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from mppi_controller.core.jackal import Jackal
from mppi_controller.core.obstacle import Obstacle
from mppi_controller.simulation import result_io
from mppi_controller.simulation.results_path import create_results_path


ControllerFactory = Callable[..., object]


@dataclass(frozen=True)
class SimulationSummary:
    steps: int
    reached_goal: bool
    failed: bool
    results_path: Path
    run_time: float


def goal_reached(state: np.ndarray, target: np.ndarray, minimum_distance: float) -> bool:
    distance = np.linalg.norm(target[:2] - state[:2])
    return bool(distance < minimum_distance - 0.1)


class MppiSimulationRunner:
    def __init__(
        self,
        config: dict,
        controller_factory: ControllerFactory | None = None,
        logger=None,
    ) -> None:
        self.config = config
        self.logger = logger

        sim = config["simulation"]
        mppi = config["mppi"]
        robot_cfg = config["robot"]
        cbf = config["cbf"]
        obstacles = config["obstacles"]

        self.hz = float(sim["sampling_rate"])
        self.dt = 1.0 / self.hz
        self.time_horizon = float(sim["time_horizon"])
        self.num_timesteps = int(self.time_horizon * self.hz)
        self.state_dim = int(mppi["state_dim"])
        self.control_dim = int(mppi["control_dim"])
        self.draw_num_traj = int(mppi["draw_num_traj"])
        self.max_steps = int(sim["max_steps"])
        self.minimum_distance = float(sim["minimum_distance"])
        self.without_heading = bool(sim["without_heading"])
        self.print_out = bool(sim["print_out"])
        self.soft_cbf = bool(mppi["soft_cbf"])
        self.max_linear_velocity = float(robot_cfg["max_linear_velocity"])
        self.max_angular_velocity = float(robot_cfg["max_angular_velocity"])
        self.robot_radius = float(robot_cfg["radius"])
        self.safety_dist = float(robot_cfg["safety_dist"])
        self.check_dist = float(cbf["check_dist"])
        self.ob_num_max = int(obstacles["num_max"])

        self.state = np.asarray(sim["initial_state"], dtype=np.float32)
        self.init_pose = np.copy(self.state)
        self.targets = np.asarray(sim["goal"], dtype=np.float32)
        self.desired_pose = np.asarray(sim["goal"], dtype=np.float32)
        self.weights = np.asarray(mppi["weights"], dtype=np.float32)
        self.std_n = np.asarray(mppi["std_normal"], dtype=np.float32)
        self.lambda_ = float(mppi["lambda"])
        self.std_n_sla = float(mppi["std_slack"])
        self.r_sla = self.lambda_ / self.std_n_sla
        self.R = np.divide(
            self.lambda_,
            self.std_n,
            out=np.full_like(self.std_n, np.inf, dtype=np.float32),
            where=self.std_n != 0,
        )
        self.logn_info = self._logn_info(int(mppi["dist_type"]), self.std_n)

        self.obstacle = Obstacle(
            self.dt,
            self.time_horizon,
            self.hz,
            float(cbf["slack_weight"]),
            float(cbf["max_slack_vari"]),
            num_max=self.ob_num_max,
            soft_cbf=self.soft_cbf,
            virtual_obstacles=obstacles["virtual"],
        )
        self.robot = Jackal(
            self.state_dim,
            self.dt,
            self.max_linear_velocity,
            self.max_angular_velocity,
            self.robot_radius,
            self.safety_dist,
            float(cbf["atau"]),
            int(robot_cfg["obstacle_state_num"]),
            cbf_type=int(cbf["type"]),
            dcbf_alpha=float(cbf["dcbf_alpha"]),
            dcbf_weight=float(cbf["dcbf_weight"]),
        )

        self.controller = (controller_factory or self._default_controller_factory)(
            config=config,
            robot=self.robot,
            obstacle=self.obstacle,
            runner=self,
        )

        self.results_path = create_results_path(config["results"])
        self.state_history: list[np.ndarray] = []
        self.desired_state_history: list[np.ndarray] = []
        self.control_history: list[np.ndarray] = []
        self.optimal_u_history: list[np.ndarray] = []
        self.sample_u_history: list[np.ndarray] = []
        self.state_cost_history: list[float] = []
        self.control_cost_history: list[float] = []
        self.min_cost_history: list[float] = []
        self.ob_state_history: list[list[np.ndarray]] = []
        self.ob_slack_history: list[np.ndarray] = []
        self.mppi_time_history: list[float] = []
        self.failed = False

    def run(self) -> SimulationSummary:
        start_time = time.time()
        steps = 0
        while steps < self.max_steps and not goal_reached(
            self.state, self.desired_pose, self.minimum_distance
        ):
            self.step()
            steps += 1
            if self.failed:
                break

        self._save_results()
        if self.config["results"].get("enable_plots", True):
            self._plot_results()

        return SimulationSummary(
            steps=steps,
            reached_goal=goal_reached(self.state, self.desired_pose, self.minimum_distance),
            failed=self.failed,
            results_path=self.results_path,
            run_time=steps * self.dt,
        )

    def step(self) -> np.ndarray:
        self.targets = np.copy(self.desired_pose)
        self._update_obstacles()
        distance_to_goal = np.linalg.norm(self.targets[:2] - self.state[:2])
        if self.without_heading and distance_to_goal > self.minimum_distance:
            self.targets[2] = self.state[2]
        elif not self.without_heading:
            self.targets[2] = self._heading_angle(self.state, self.targets)

        start = time.time()
        if self.soft_cbf:
            u, optimal_u, sample_u, ob_u, _normalizer, min_cost = self.controller.compute_control(
                self.state,
                [self.std_n, self.R, self.weights, self.targets],
                [
                    self.obstacle.data,
                    self.obstacle.num,
                    self.obstacle.id,
                    self.r_sla,
                    self.std_n_sla,
                    self.obstacle.U,
                    self.obstacle.state_sla,
                ],
            )
            self.obstacle.state_sla = self.obstacle.update_state(self.obstacle.state_sla, ob_u)
            self.ob_slack_history.append(np.copy(self.obstacle.state_sla))
        else:
            u, optimal_u, sample_u, _normalizer, min_cost = self.controller.compute_control(
                self.state,
                [self.std_n, self.R, self.weights, self.targets, self.obstacle.data, self.obstacle.num],
            )

        u = np.asarray(u, dtype=np.float32)
        u[0] = np.clip(u[0], -self.max_linear_velocity, self.max_linear_velocity)
        u[1] = np.clip(u[1], -self.max_angular_velocity, self.max_angular_velocity)
        t_mppi = (time.time() - start) * 1000.0

        state_cost, control_cost = self.robot.cost(self.state, u, [self.weights, self.targets, self.R])
        self.control_history.append(np.copy(u))
        self.state_history.append(np.copy(self.state))
        self.desired_state_history.append(np.copy(self.targets))
        self.state_cost_history.append(float(state_cost))
        self.control_cost_history.append(float(control_cost))
        self.min_cost_history.append(float(min_cost))
        self.optimal_u_history.append(np.copy(optimal_u))
        self.sample_u_history.append(np.copy(sample_u).reshape(self.draw_num_traj, self.num_timesteps, self.control_dim))
        self.ob_state_history.append([np.copy(ob) for ob in self.obstacle.all_cur_state])
        self.mppi_time_history.append(0.0 if t_mppi > 10.0 else t_mppi)

        if np.isnan(np.sum(self.state)):
            self.failed = True
            return u

        self.state = self.robot.update_state(self.state, u)
        if not self.config["obstacles"].get("static_enabled", False):
            self.obstacle.update_cur_state_virtual(self.obstacle.virtual_ob_state)
        return u

    def _update_obstacles(self) -> None:
        all_ob_cur = []
        active_obstacles = []
        active_cur_states = []
        active_ids = []
        data = np.copy(self.obstacle.update_predict_state_virtual(self.obstacle.virtual_ob_state))
        num_obs = int(len(data) / (7 * self.num_timesteps))
        for idx in range(num_obs):
            current = np.array(data[7 * (idx * self.num_timesteps) : 7 * (idx * self.num_timesteps + 1)])
            all_ob_cur.append(current)
            if self._obstacle_in_range(data[7 * idx * self.num_timesteps : 7 * idx * self.num_timesteps + 7]):
                active_ids.append(idx)
                active_cur_states.append(current)
                for step in range(self.num_timesteps):
                    active_obstacles.append(
                        np.array(data[7 * (idx * self.num_timesteps + step) : 7 * (idx * self.num_timesteps + step + 1)])
                    )

        self.obstacle.all_cur_state = all_ob_cur
        self.obstacle.data = active_obstacles
        self.obstacle.cur_state = active_cur_states
        self.obstacle.id_last = self.obstacle.id
        self.obstacle.id = active_ids
        self.obstacle.num = len(active_ids)

    def _obstacle_in_range(self, obstacle: np.ndarray) -> bool:
        distance = np.linalg.norm(self.state[:2] - obstacle[:2]) - obstacle[2] - self.robot_radius
        return max(distance, 1e-6) <= self.check_dist

    def _heading_angle(self, current_state: np.ndarray, desired_state: np.ndarray) -> float:
        return math.atan2(desired_state[1] - current_state[1], desired_state[0] - current_state[0])

    def _path_length(self) -> float:
        if not self.state_history:
            return 0.0
        positions = [state[:2] for state in self.state_history]
        positions.append(self.state[:2])
        deltas = np.diff(np.asarray(positions, dtype=np.float32), axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def _summary_metrics(self) -> dict:
        final_distance = float(np.linalg.norm(self.desired_pose[:2] - self.state[:2]))
        mean_mppi_time = float(np.mean(self.mppi_time_history)) if self.mppi_time_history else 0.0
        max_mppi_time = float(np.max(self.mppi_time_history)) if self.mppi_time_history else 0.0
        success = goal_reached(self.state, self.desired_pose, self.minimum_distance)
        return {
            "init_pose": self.init_pose.tolist(),
            "goal": self.desired_pose.tolist(),
            "steps": len(self.state_history),
            "success": success,
            "reached_goal": success,
            "failed": self.failed,
            "final_distance": final_distance,
            "path_length": self._path_length(),
            "arrival_time": len(self.state_history) * self.dt if success else None,
            "run_time": len(self.state_history) * self.dt,
            "mean_mppi_time_ms": mean_mppi_time,
            "max_mppi_time_ms": max_mppi_time,
            "average_mppi_time_ms": mean_mppi_time,
        }

    def _save_results(self) -> None:
        result_io.save_results(
            self.state_history,
            self.desired_state_history,
            self.state_cost_history,
            self.control_cost_history,
            self.control_history,
            self.results_path,
        )
        result_io.save_obs_results(self.ob_num_max, self.results_path, self.ob_state_history)
        result_io.save_time_results(self.results_path, self.mppi_time_history)
        result_io.save_summary(self.results_path, self._summary_metrics())

    def _plot_results(self) -> None:
        from mppi_controller.visualization import utils

        obs_num = len(self.ob_state_history[0]) if self.ob_state_history else 0
        utils.statePlotting(self.state_history, str(self.results_path))
        utils.controlPlotting(self.control_history, str(self.results_path))
        utils.costPlotting(
            self.state_cost_history,
            self.control_cost_history,
            self.min_cost_history,
            self.mppi_time_history,
            str(self.results_path),
        )
        utils.pathPlotting(obs_num, self.robot_radius, self.targets, str(self.results_path))
        utils.plot_cbf(
            self.dt,
            obs_num,
            self.robot_radius,
            self.safety_dist,
            str(self.results_path),
            self.config["obstacles"].get("static_enabled", False),
        )
        if self.config["results"].get("enable_animation", True):
            try:
                utils.animate_simulation(
                    self.dt,
                    obs_num,
                    self.safety_dist,
                    self.robot_radius,
                    self.config["cbf"]["atau"],
                    self.targets,
                    str(self.results_path),
                    self.sample_u_history,
                    optimal_us=self.optimal_u_history,
                    cbf_type=self.config["cbf"]["type"],
                )
            except Exception as exc:
                warnings.warn(
                    f"Animation failed; continuing without animation: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _default_controller_factory(self, *, config: dict, robot: Jackal, obstacle: Obstacle, runner: "MppiSimulationRunner") -> object:
        from mppi_controller.controllers.mppi_cbf import MPPI_Controller
        from mppi_controller.controllers.mppi_cbf_soft import MPPI_Controller_Soft

        mppi = config["mppi"]
        cbf = config["cbf"]
        if not self.soft_cbf:
            return MPPI_Controller(
                self.state_dim,
                self.control_dim,
                int(mppi["num_trajectories"]),
                self.draw_num_traj,
                self.time_horizon,
                self.hz,
                float(mppi["exploration_variance"]),
                robot.cuda_kinematics(),
                robot.cuda_state_cost(),
                int(mppi["sg_window"]),
                int(mppi["sg_poly_order"]),
                self.logn_info,
                int(cbf["type"]),
                check_dist=self.check_dist,
                beta_1=float(mppi["beta_1"]),
                beta_2=float(mppi["beta_2"]),
                beta_3=float(mppi["beta_3"]),
                lambda_=self.lambda_,
                atau=float(cbf["atau"]),
            )
        return MPPI_Controller_Soft(
            self.state_dim,
            self.control_dim,
            int(mppi["num_trajectories"]),
            self.draw_num_traj,
            self.time_horizon,
            self.hz,
            float(mppi["exploration_variance"]),
            robot.cuda_kinematics(),
            robot.cuda_state_cost(),
            obstacle.cuda_kinematics(),
            obstacle.cuda_state_cost(),
            int(mppi["sg_window"]),
            int(mppi["sg_poly_order"]),
            self.logn_info,
            check_dist=self.check_dist,
            beta_1=float(mppi["beta_1"]),
            beta_2=float(mppi["beta_2"]),
            beta_3=float(mppi["beta_3"]),
            ob_num_max=self.ob_num_max,
            lambda_=self.lambda_,
        )

    def _logn_info(self, dist_type: int, std_n: np.ndarray) -> list[float]:
        std_mean = float(np.mean(std_n))
        return [dist_type, 0.0, std_mean]
