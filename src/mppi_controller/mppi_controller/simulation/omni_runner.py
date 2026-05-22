"""Simulation runner for the B2 omnidirectional MPPI controller."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from mppi_controller.controllers.mppi_omni_numpy import MppiOmniNumpy
from mppi_controller.controllers.mppi_omni_high_level_fdm_torch import MppiOmniHighLevelFdmTorch
from mppi_controller.controllers.mppi_omni_sequence_fdm_torch import MppiOmniSequenceFdmTorch
from mppi_controller.controllers.mppi_omni_torch import MppiOmniTorch
from mppi_controller.core.omni_b2 import OmniB2
from mppi_controller.core.residual_world import ResidualWorld
from mppi_controller.core.terrain import TerrainField
from mppi_controller.simulation.random_obstacles import generate_random_obstacles
from mppi_controller.simulation.random_scenario import sample_start_goal
from mppi_controller.simulation.results_path import create_results_path
from mppi_controller.visualization.utils import map_axis_limits_from_config

try:
    from mppi_controller.controllers.mppi_omni_cuda import MppiOmniCuda
except Exception:  # pragma: no cover - exercised on machines without CUDA/PyCUDA.
    MppiOmniCuda = None


ControllerFactory = Callable[..., object]


def create_omni_controller(config: dict, seed: int = 123) -> object:
    backend = str(config["mppi"].get("backend", "numpy")).lower()
    fdm_cfg = config.get("fdm", {})
    fdm_mode = str(fdm_cfg.get("mode", "sequence")).lower()
    if bool(fdm_cfg.get("enabled", False)):
        if backend == "numpy":
            raise ValueError("sequence FDM requires torch or cuda backend; numpy learned FDM was removed")
        if backend in {"cuda", "torch"}:
            if fdm_mode in {"sequence", "sequence_fdm"}:
                return MppiOmniSequenceFdmTorch.from_config(config, seed=seed)
            if fdm_mode in {"high_level_fdm", "hfdm"}:
                return MppiOmniHighLevelFdmTorch.from_config(config, seed=seed)
            raise ValueError(f"Only sequence FDM/high_level_fdm modes are supported; got fdm.mode={fdm_mode!r}")
        raise ValueError(f"Unsupported sequence FDM/high_level_fdm MPPI backend: {backend}")
    if backend == "torch":
        return MppiOmniTorch.from_config(config, seed=seed)
    if backend == "cuda":
        if MppiOmniCuda is None:
            raise RuntimeError("mppi.backend is 'cuda' but PyCUDA controller is unavailable")
        return MppiOmniCuda.from_config(config, seed=seed)
    if backend == "numpy":
        return MppiOmniNumpy.from_config(config, seed=seed)
    raise ValueError(f"Unsupported omni MPPI backend: {backend}")


@dataclass(frozen=True)
class OmniSimulationSummary:
    steps: int
    reached_goal: bool
    failed: bool
    results_path: Path
    run_time: float


def goal_reached_xy(state: np.ndarray, target: np.ndarray, minimum_distance: float) -> bool:
    return bool(np.linalg.norm(target[:2] - state[:2]) < minimum_distance - 0.1)


class OmniMppiSimulationRunner:
    def __init__(
        self,
        config: dict,
        controller_factory: ControllerFactory | None = None,
        logger=None,
    ) -> None:
        self.config = config
        self.logger = logger
        sim = config["simulation"]
        robot_cfg = config["robot"]

        self.hz = float(sim["sampling_rate"])
        self.dt = 1.0 / self.hz
        self.max_steps = int(sim["max_steps"])
        self.minimum_distance = float(sim["minimum_distance"])
        self.disable_goal_termination = bool(sim.get("disable_goal_termination", False))
        self.scenario_mode = "fixed"
        self.scenario_random_seed = None
        self._apply_random_start_goal()
        self.state = np.asarray(sim["initial_state"], dtype=np.float32)
        self.init_pose = np.copy(self.state)
        self.goal = np.asarray(sim["goal"], dtype=np.float32)
        self.start_goal_distance = float(np.linalg.norm(self.goal[:2] - self.init_pose[:2]))
        self.world_mode = str(sim.get("world_mode", "nominal")).lower()
        if self.world_mode not in {"nominal", "oracle"}:
            raise ValueError(f"Unsupported simulation.world_mode: {self.world_mode}")
        self.obstacles, self.obstacle_mode, self.obstacle_random_seed = self._build_obstacles()
        execution_cfg = config.get("execution", {})
        self.filter_enabled = bool(execution_cfg.get("filter_enabled", False))
        self.filter_alpha = float(execution_cfg.get("filter_alpha", 0.0))
        self.filter_alpha = float(np.clip(self.filter_alpha, 0.0, 1.0))
        self.rollout_max_control = np.asarray(
            [robot_cfg["max_vx"], robot_cfg["max_vy"], robot_cfg["max_wz"]],
            dtype=np.float32,
        )
        self.rollout_max_accel = np.asarray(
            [
                robot_cfg.get("max_ax", 1000.0),
                robot_cfg.get("max_ay", 1000.0),
                robot_cfg.get("max_awz", 1000.0),
            ],
            dtype=np.float32,
        )
        self.rollout_velocity_lag_beta = float(np.clip(robot_cfg.get("velocity_lag_beta", 0.0), 0.0, 1.0))
        self.robot = OmniB2(
            self.dt,
            float(robot_cfg["max_vx"]),
            float(robot_cfg["max_vy"]),
            float(robot_cfg["max_wz"]),
        )
        self.terrain = TerrainField.from_config(config.get("terrain"))
        self.oracle_world = ResidualWorld.from_config(
            config.get("oracle_residual"),
            model=self.robot,
            terrain=self.terrain,
        )
        self.controller = (controller_factory or self._default_controller_factory)(
            config=config,
            runner=self,
        )
        self.results_path = create_results_path(config["results"])
        self.state_history: list[np.ndarray] = []
        self.control_history: list[np.ndarray] = []
        self.raw_control_history: list[np.ndarray] = []
        self.cmd_control_history: list[np.ndarray] = []
        self.residual_history: list[np.ndarray] = []
        self.terrain_history: list[np.ndarray] = []
        self.terrain_risk_history: list[float] = []
        self.previous_exec_control = np.zeros(3, dtype=np.float32)
        self.mppi_time_history: list[float] = []
        self.min_cost_history: list[float] = []
        self.optimal_u_history: list[np.ndarray] = []
        self.sample_u_history: list[np.ndarray] = []
        self.failed = False

    def _apply_random_start_goal(self) -> None:
        scenario_cfg = self.config.get("scenario", {})
        if not bool(scenario_cfg.get("random_start_goal_enabled", False)):
            return
        scenario_cfg["random_seed"] = self._resolve_scenario_seed(scenario_cfg.get("random_seed", 123))
        fixed_obstacles = None
        obstacle_cfg = self.config.get("obstacles", {})
        if not bool(obstacle_cfg.get("random_enabled", False)):
            fixed_obstacles = np.asarray(obstacle_cfg.get("virtual", []), dtype=np.float32).reshape(-1, 7)
        start_state, goal_state = sample_start_goal(self.config, obstacles=fixed_obstacles)
        self.config["simulation"]["initial_state"] = [float(v) for v in start_state.tolist()]
        self.config["simulation"]["goal"] = [float(v) for v in goal_state.tolist()]
        terrain_cfg = self.config.setdefault("terrain", {})
        relief_cfg = terrain_cfg.get("goal_relief")
        if isinstance(relief_cfg, dict) and bool(relief_cfg.get("enabled", False)):
            relief_cfg["center"] = [float(goal_state[0]), float(goal_state[1])]
        self.scenario_mode = "random_start_goal"
        self.scenario_random_seed = int(scenario_cfg["random_seed"])

    def _resolve_scenario_seed(self, value) -> int:
        if value is None or (isinstance(value, str) and value.lower() == "auto"):
            return int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
        return int(value)

    def run(self) -> OmniSimulationSummary:
        if self.world_mode == "oracle":
            self.oracle_world.reset()
        steps = 0
        while steps < self.max_steps and (
            self.disable_goal_termination or not goal_reached_xy(self.state, self.goal, self.minimum_distance)
        ):
            self.step()
            steps += 1
            if self.failed:
                break
        self._save_results()
        if self.config["results"].get("enable_plots", True):
            self._plot_results()
        return OmniSimulationSummary(
            steps=steps,
            reached_goal=goal_reached_xy(self.state, self.goal, self.minimum_distance),
            failed=self.failed,
            results_path=self.results_path,
            run_time=steps * self.dt,
        )

    def step(self) -> np.ndarray:
        start = time.time()
        raw_u, optimal_u, sample_u, _normalizer, min_cost = self.controller.compute_control(
            self.state,
            [None, None, None, self.goal, self.obstacles, len(self.obstacles)],
        )
        elapsed_ms = (time.time() - start) * 1000.0
        raw_u = np.asarray(raw_u, dtype=np.float32)
        u_cmd = self._execution_control(raw_u)

        if self.world_mode == "oracle":
            next_state, u_real, delta_u, terrain_features = self.oracle_world.update_state(self.state, u_cmd)
            terrain_risk = self.terrain.risk_cost(
                float(self.state[0]),
                float(self.state[1]),
                features=terrain_features,
            )
        else:
            terrain_features = self.terrain.feature(float(self.state[0]), float(self.state[1]))
            terrain_risk = self.terrain.risk_cost(
                float(self.state[0]),
                float(self.state[1]),
                features=terrain_features,
            )
            u_real = u_cmd
            delta_u = np.zeros(3, dtype=np.float32)
            next_state = self.robot.update_state(self.state, u_real)

        self.state_history.append(np.copy(self.state))
        self.raw_control_history.append(np.copy(raw_u))
        self.cmd_control_history.append(np.copy(u_cmd))
        self.control_history.append(np.copy(u_real))
        self.residual_history.append(np.copy(delta_u))
        self.terrain_history.append(np.copy(terrain_features))
        self.terrain_risk_history.append(float(terrain_risk))
        self.mppi_time_history.append(float(elapsed_ms))
        self.min_cost_history.append(float(min_cost))
        self.optimal_u_history.append(np.copy(optimal_u))
        self.sample_u_history.append(np.copy(sample_u))

        if np.isnan(np.sum(self.state)) or np.isnan(np.sum(u_real)) or np.isnan(np.sum(next_state)):
            self.failed = True
            return u_real
        self.state = next_state
        return u_real

    def _execution_control(self, raw_u: np.ndarray) -> np.ndarray:
        if not self.filter_enabled:
            self.previous_exec_control = np.asarray(raw_u, dtype=np.float32).copy()
            return self.previous_exec_control.copy()
        filtered = self.filter_alpha * self.previous_exec_control + (1.0 - self.filter_alpha) * raw_u
        self.previous_exec_control = np.asarray(filtered, dtype=np.float32)
        return self.previous_exec_control.copy()

    def _build_obstacles(self) -> tuple[np.ndarray, str, int | None]:
        obstacle_cfg = self.config.get("obstacles", {})
        if bool(obstacle_cfg.get("random_enabled", False)):
            obstacles = generate_random_obstacles(obstacle_cfg, self.init_pose[:2], self.goal[:2])
            obstacle_cfg["virtual"] = obstacles.tolist()
            obstacle_cfg["num_max"] = int(len(obstacles))
            return obstacles, "random", int(obstacle_cfg.get("random_seed", 0))
        obstacles = np.asarray(obstacle_cfg.get("virtual", []), dtype=np.float32).reshape(-1, 7)
        return obstacles, "fixed", None

    def _summary_metrics(self) -> dict:
        final_distance = float(np.linalg.norm(self.goal[:2] - self.state[:2]))
        path_length = self._path_length()
        mean_time = float(np.mean(self.mppi_time_history)) if self.mppi_time_history else 0.0
        max_time = float(np.max(self.mppi_time_history)) if self.mppi_time_history else 0.0
        success = goal_reached_xy(self.state, self.goal, self.minimum_distance)
        residual_norms = (
            np.linalg.norm(np.asarray(self.residual_history, dtype=np.float32), axis=1)
            if self.residual_history
            else np.zeros(0, dtype=np.float32)
        )
        cmd_controls = np.asarray(self.cmd_control_history, dtype=np.float32)
        real_controls = np.asarray(self.control_history, dtype=np.float32)
        cmd_real_errors = (
            np.linalg.norm(real_controls - cmd_controls, axis=1)
            if len(cmd_controls) and len(real_controls)
            else np.zeros(0, dtype=np.float32)
        )
        mean_residual = float(np.mean(residual_norms)) if residual_norms.size else 0.0
        max_residual = float(np.max(residual_norms)) if residual_norms.size else 0.0
        mean_cmd_real_error = float(np.mean(cmd_real_errors)) if cmd_real_errors.size else 0.0
        max_cmd_real_error = float(np.max(cmd_real_errors)) if cmd_real_errors.size else 0.0
        terrain_metrics = self._terrain_risk_metrics()
        return {
            "world_mode": self.world_mode,
            "controller_type": type(self.controller).__name__,
            **self._fdm_metadata(),
            "success": success,
            "reached_goal": success,
            "failed": self.failed,
            "goal_termination_disabled": self.disable_goal_termination,
            "init_pose": self.init_pose.tolist(),
            "goal": self.goal.tolist(),
            "steps": len(self.state_history),
            "final_distance": final_distance,
            "path_length": path_length,
            "arrival_time": len(self.state_history) * self.dt if success else None,
            "run_time": len(self.state_history) * self.dt,
            "mean_mppi_time_ms": mean_time,
            "max_mppi_time_ms": max_time,
            "min_obstacle_clearance": self._min_obstacle_clearance(),
            "obstacle_mode": self.obstacle_mode,
            "num_obstacles": int(len(self.obstacles)),
            "obstacle_random_seed": self.obstacle_random_seed,
            "scenario_mode": self.scenario_mode,
            "scenario_random_seed": self.scenario_random_seed,
            "start_goal_distance": self.start_goal_distance,
            "controls_csv": "executed_controls",
            "raw_controls_csv": "raw_controls",
            "mean_residual_norm": mean_residual,
            "max_residual_norm": max_residual,
            "mean_cmd_real_error": mean_cmd_real_error,
            "max_cmd_real_error": max_cmd_real_error,
            **terrain_metrics,
            **self._control_metrics(),
            **self._sample_coverage_metrics(),
        }

    def _terrain_risk_metrics(self) -> dict:
        risks = np.asarray(self.terrain_risk_history, dtype=np.float32)
        if risks.size == 0:
            return {
                "mean_terrain_risk": 0.0,
                "max_terrain_risk": 0.0,
                "cumulative_terrain_risk": 0.0,
                "terrain_risk_excess": 0.0,
                "terrain_risk_excess_integral": 0.0,
                "terrain_risk_exposure_ratio": 0.0,
            }
        threshold = self._terrain_risk_threshold()
        excess = np.maximum(risks - threshold, 0.0)
        return {
            "mean_terrain_risk": float(np.mean(risks)),
            "max_terrain_risk": float(np.max(risks)),
            "cumulative_terrain_risk": float(np.sum(risks)),
            "terrain_risk_excess": float(np.sum(excess)),
            "terrain_risk_excess_integral": float(np.sum(excess * excess)),
            "terrain_risk_exposure_ratio": float(np.mean(risks > threshold)),
        }

    def _terrain_risk_threshold(self) -> float:
        return float(self.config.get("mppi", {}).get("terrain_risk_threshold", 0.0))

    def _fdm_metadata(self) -> dict:
        fdm_cfg = self.config.get("fdm", {})
        enabled = bool(fdm_cfg.get("enabled", False))
        metadata = {
            "fdm_enabled": enabled,
            "fdm_model_dir": fdm_cfg.get("model_dir") if enabled else None,
            "fdm_checkpoint": fdm_cfg.get("checkpoint", "best_model.pt") if enabled else None,
            "fdm_normalization": fdm_cfg.get("normalization", "normalization.npz") if enabled else None,
            "fdm_device": fdm_cfg.get("device", "cpu") if enabled else None,
            "fdm_mode": fdm_cfg.get("mode", "sequence") if enabled else None,
            "fdm_sequence_horizon": int(fdm_cfg.get("sequence_horizon", 0)) if enabled else None,
            "fdm_profile_enabled": bool(fdm_cfg.get("profile_enabled", False)) if enabled else None,
        }
        learned = getattr(self.controller, "learned_dynamics", None)
        if learned is not None:
            metadata["fdm_checkpoint_path"] = str(getattr(learned, "checkpoint_path", ""))
            metadata["fdm_normalization_path"] = str(getattr(learned, "normalization_path", ""))
        else:
            metadata["fdm_checkpoint_path"] = None
            metadata["fdm_normalization_path"] = None
        profile_summary = getattr(self.controller, "profile_summary", None)
        if enabled and callable(profile_summary):
            metadata["fdm_runtime_profile"] = profile_summary()
        return metadata

    def _save_results(self) -> None:
        self._save_config()
        self._save_trajectory()
        self._save_controls()
        self._save_raw_controls()
        self._save_residuals()
        self._save_terrain()
        self._save_obstacles()
        pd.DataFrame({"mppi_time_ms": self.mppi_time_history}).to_csv(
            self.results_path / "time_results.csv", index=False
        )
        pd.DataFrame({"min_cost": self.min_cost_history}).to_csv(
            self.results_path / "costs.csv", index=False
        )
        summary = self._summary_metrics()
        (self.results_path / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        with (self.results_path / "test_summary.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(summary, stream, sort_keys=False)

    def _save_config(self) -> None:
        with (self.results_path / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.config, stream, sort_keys=False)

    def _save_trajectory(self) -> None:
        rows = []
        for idx, state in enumerate(self.state_history):
            rows.append(self._trajectory_row(idx, state))
        if self.state_history:
            rows.append(self._trajectory_row(len(self.state_history), self.state))
        pd.DataFrame(rows).to_csv(self.results_path / "trajectory.csv", index=False)
        pd.DataFrame(rows).rename(columns={"vx": "dx", "vy": "dy"}).to_csv(
            self.results_path / "results.csv", index=False
        )

    def _trajectory_row(self, step: int, state: np.ndarray) -> dict:
        return {
            "step": step,
            "x": state[0],
            "y": state[1],
            "theta": state[2],
            "vx": state[3],
            "vy": state[4],
            "wz": state[5],
            "x_des": self.goal[0],
            "y_des": self.goal[1],
            "theta_des": self.goal[2],
        }

    def _save_controls(self) -> None:
        rows = []
        for idx, control in enumerate(self.control_history):
            rows.append({"step": idx, "vx_cmd": control[0], "vy_cmd": control[1], "wz_cmd": control[2]})
        pd.DataFrame(rows).to_csv(self.results_path / "controls.csv", index=False)

    def _save_raw_controls(self) -> None:
        rows = []
        for idx, control in enumerate(self.raw_control_history):
            rows.append({"step": idx, "vx_cmd": control[0], "vy_cmd": control[1], "wz_cmd": control[2]})
        pd.DataFrame(rows).to_csv(self.results_path / "raw_controls.csv", index=False)

    def _save_residuals(self) -> None:
        rows = []
        for idx, (cmd, real, delta) in enumerate(
            zip(self.cmd_control_history, self.control_history, self.residual_history)
        ):
            exec_delta = np.asarray(real, dtype=np.float32) - np.asarray(cmd, dtype=np.float32)
            oracle_du_norm = float(np.linalg.norm(delta))
            exec_du_norm = float(np.linalg.norm(exec_delta))
            rows.append(
                {
                    "step": idx,
                    "cmd_vx": cmd[0],
                    "cmd_vy": cmd[1],
                    "cmd_wz": cmd[2],
                    "real_vx": real[0],
                    "real_vy": real[1],
                    "real_wz": real[2],
                    "oracle_du_vx": delta[0],
                    "oracle_du_vy": delta[1],
                    "oracle_du_wz": delta[2],
                    "oracle_du_norm": oracle_du_norm,
                    "exec_du_vx": exec_delta[0],
                    "exec_du_vy": exec_delta[1],
                    "exec_du_wz": exec_delta[2],
                    "exec_du_norm": exec_du_norm,
                    "du_vx": delta[0],
                    "du_vy": delta[1],
                    "du_wz": delta[2],
                    "du_norm": oracle_du_norm,
                }
            )
        pd.DataFrame(rows).to_csv(self.results_path / "residuals.csv", index=False)

    def _save_terrain(self) -> None:
        rows = []
        threshold = self._terrain_risk_threshold()
        for idx, (state, features, risk) in enumerate(
            zip(self.state_history, self.terrain_history, self.terrain_risk_history)
        ):
            risk_excess = max(float(risk) - threshold, 0.0)
            rows.append(
                {
                    "step": idx,
                    "x": state[0],
                    "y": state[1],
                    "slope_f": features[0],
                    "slope_l": features[1],
                    "roughness": features[2],
                    "friction": features[3],
                    "risk_cost": risk,
                    "risk_excess": risk_excess,
                    "risk_exposed": bool(float(risk) > threshold),
                }
            )
        pd.DataFrame(rows).to_csv(self.results_path / "terrain.csv", index=False)

    def _save_obstacles(self) -> None:
        rows = []
        for _ in self.state_history:
            row = {}
            for idx, obstacle in enumerate(self.obstacles):
                row.update(
                    {
                        f"x{idx}": obstacle[0],
                        f"y{idx}": obstacle[1],
                        f"r{idx}": obstacle[2],
                        f"theta{idx}": obstacle[4],
                        f"dx{idx}": obstacle[5],
                        f"dy{idx}": obstacle[6],
                    }
                )
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.results_path / "obs_results.csv", index=False)

    def _plot_results(self) -> None:
        self._plot_trajectory()
        if self.world_mode == "oracle":
            from mppi_controller.visualization.oracle_viewer import plot_oracle_diagnostics

            plot_oracle_diagnostics(self.results_path, self.config)
        if self.config["results"].get("enable_animation", True):
            self._save_animation()

    def _plot_trajectory(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        states = np.asarray(self.state_history)
        if len(states):
            ax.plot(states[:, 0], states[:, 1], color="tab:blue", label="trajectory")
        self._draw_obstacles(ax)
        ax.scatter([self.init_pose[0]], [self.init_pose[1]], color="green", label="start")
        ax.scatter([self.goal[0]], [self.goal[1]], color="purple", label="goal")
        self._draw_goal_tolerance(ax)
        xlim, ylim = map_axis_limits_from_config(self.config)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.savefig(self.results_path / "trajectory.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_animation(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        states = np.asarray(self.state_history)
        residual_norms = (
            np.linalg.norm(np.asarray(self.residual_history, dtype=np.float32), axis=1)
            if self.residual_history
            else np.zeros(0, dtype=np.float32)
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        xlim, ylim = map_axis_limits_from_config(self.config)
        risk_grid = self._terrain_risk_grid(xlim, ylim) if self.world_mode == "oracle" else None
        max_residual = float(np.max(residual_norms)) if residual_norms.size else 1.0

        def update(frame):
            ax.clear()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            if risk_grid is not None:
                self._draw_oracle_animation_background(ax, risk_grid, xlim, ylim)
            self._draw_obstacles(ax)
            ax.scatter([self.init_pose[0]], [self.init_pose[1]], color="green", label="start", zorder=5)
            ax.scatter([self.goal[0]], [self.goal[1]], color="purple", marker="*", s=110, label="goal", zorder=5)
            self._draw_goal_tolerance(ax)
            if frame >= 0 and len(states):
                self._draw_predicted_rollouts(ax, states[frame], frame)
                if self.world_mode == "oracle" and residual_norms.size:
                    upto = min(frame + 1, len(states), len(residual_norms))
                    ax.scatter(
                        states[:upto, 0],
                        states[:upto, 1],
                        c=residual_norms[:upto],
                        cmap="viridis",
                        vmin=0.0,
                        vmax=max_residual,
                        s=18,
                        label="residual path",
                        zorder=4,
                    )
                    self._draw_oracle_velocity_arrows(ax, states, frame)
                else:
                    ax.plot(states[: frame + 1, 0], states[: frame + 1, 1], color="tab:blue")
                self._draw_robot_heading(ax, states[frame])
                ax.scatter([states[frame, 0]], [states[frame, 1]], color="red", zorder=6)
            ax.legend(handles=self._animation_legend_handles(self.world_mode == "oracle"), loc="upper right", fontsize=7)

        frames = max(1, len(states))
        animation = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
        animation.save(self.results_path / "animation.gif", writer=PillowWriter(fps=5))
        plt.close(fig)

    def _animation_legend_handles(self, is_oracle: bool):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="green", markersize=6, label="start"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="purple", markersize=10, label="goal"),
            Line2D([0], [0], color="#f59e0b", linestyle="--", linewidth=1.2, label="goal tolerance"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="red", markersize=6, label="current state"),
            Line2D([0], [0], color="tab:cyan", linewidth=1.6, label="actual heading"),
            Line2D([0], [0], color="black", alpha=0.25, linewidth=1.0, label="nominal sampled rollouts"),
            Line2D([0], [0], color="orange", linewidth=1.4, label="nominal optimal rollout"),
        ]
        if is_oracle:
            handles.extend(
                [
                    Patch(facecolor="tab:red", alpha=0.25, label="terrain risk"),
                    Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:green", markersize=5, label="residual path"),
                ]
            )
        else:
            handles.append(Line2D([0], [0], color="tab:blue", linewidth=1.6, label="executed path"))
        return handles

    def _draw_goal_tolerance(self, ax) -> None:
        from matplotlib.patches import Circle

        if self.minimum_distance <= 0.0:
            return
        ax.add_patch(
            Circle(
                (float(self.goal[0]), float(self.goal[1])),
                self.minimum_distance,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                edgecolor="#f59e0b",
                alpha=0.95,
                zorder=4,
            )
        )

    def _terrain_risk_grid(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> np.ndarray:
        resolution = int(self.config.get("visualization", {}).get("terrain_grid_resolution", 100))
        grid_x = np.linspace(xlim[0], xlim[1], resolution)
        grid_y = np.linspace(ylim[0], ylim[1], resolution)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        risk_grid = np.zeros_like(mesh_x, dtype=np.float32)
        for i in range(mesh_x.shape[0]):
            for j in range(mesh_x.shape[1]):
                risk_grid[i, j] = self.terrain.risk_cost(float(mesh_x[i, j]), float(mesh_y[i, j]))
        return risk_grid

    def _draw_oracle_animation_background(self, ax, risk_grid: np.ndarray, xlim, ylim) -> None:
        ax.imshow(
            risk_grid,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            cmap="plasma",
            alpha=0.35,
            aspect="auto",
            zorder=0,
        )

    def _draw_oracle_velocity_arrows(self, ax, states: np.ndarray, frame: int) -> None:
        if frame >= len(self.cmd_control_history) or frame >= len(self.control_history):
            return
        x, y, theta = states[frame, :3]
        cos_theta = float(np.cos(theta))
        sin_theta = float(np.sin(theta))
        cmd = self.cmd_control_history[frame]
        real = self.control_history[frame]
        cmd_dx = float(cmd[0] * cos_theta - cmd[1] * sin_theta)
        cmd_dy = float(cmd[0] * sin_theta + cmd[1] * cos_theta)
        real_dx = float(real[0] * cos_theta - real[1] * sin_theta)
        real_dy = float(real[0] * sin_theta + real[1] * cos_theta)
        ax.arrow(
            x,
            y,
            cmd_dx,
            cmd_dy,
            color="white",
            linestyle="--",
            width=0.012,
            length_includes_head=True,
            alpha=0.8,
            zorder=7,
        )
        ax.arrow(
            x,
            y,
            real_dx,
            real_dy,
            color="black",
            width=0.012,
            length_includes_head=True,
            alpha=0.85,
            zorder=8,
        )

    def _draw_robot_heading(self, ax, state: np.ndarray) -> None:
        x, y, theta = state[:3]
        length = 0.65
        ax.arrow(
            float(x),
            float(y),
            length * float(np.cos(theta)),
            length * float(np.sin(theta)),
            color="tab:cyan",
            width=0.01,
            length_includes_head=True,
            alpha=0.95,
            zorder=9,
        )

    def _draw_predicted_rollouts(self, ax, state: np.ndarray, frame: int) -> None:
        if frame < len(self.sample_u_history):
            sampled_controls = self.sample_u_history[frame]
            max_draw = min(50, len(sampled_controls))
            for control_sequence in sampled_controls[:max_draw]:
                predicted = self._predict_trajectory(state, control_sequence)
                ax.plot(predicted[:, 0], predicted[:, 1], color="black", alpha=0.08, linewidth=0.8)
        if frame < len(self.optimal_u_history):
            predicted = self._predict_trajectory(state, self.optimal_u_history[frame])
            ax.plot(
                predicted[:, 0],
                predicted[:, 1],
                color="orange",
                alpha=0.75,
                linewidth=1.4,
                label="nominal optimal rollout",
            )

    def _predict_trajectory(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        predicted = [np.asarray(state, dtype=np.float32).copy()]
        rollout_state = predicted[0].copy()
        prev_real = rollout_state[3:].copy()
        max_delta = self.rollout_max_accel * self.dt
        for command in controls:
            command = np.clip(np.asarray(command, dtype=np.float32), -self.rollout_max_control, self.rollout_max_control)
            lagged = self.rollout_velocity_lag_beta * prev_real + (1.0 - self.rollout_velocity_lag_beta) * command
            delta = np.clip(lagged - prev_real, -max_delta, max_delta)
            control = np.clip(prev_real + delta, -self.rollout_max_control, self.rollout_max_control)
            rollout_state = self.robot.update_state(rollout_state, control)
            predicted.append(rollout_state.copy())
            prev_real = control
        return np.asarray(predicted, dtype=np.float32)

    def _draw_obstacles(self, ax) -> None:
        import matplotlib.patches as patches

        robot_radius = float(self.config["robot"]["radius"])
        safety_dist = float(self.config["robot"]["safety_dist"])
        for obstacle in self.obstacles:
            x, y, radius = obstacle[:3]
            ax.add_patch(patches.Circle((x, y), radius, color="gray", alpha=0.4))
            ax.add_patch(
                patches.Circle(
                    (x, y),
                    radius + robot_radius + safety_dist,
                    fill=False,
                    edgecolor="red",
                    linestyle="--",
                    linewidth=1.0,
                )
            )

    def _path_length(self) -> float:
        if not self.state_history:
            return 0.0
        positions = [state[:2] for state in self.state_history]
        positions.append(self.state[:2])
        deltas = np.diff(np.asarray(positions, dtype=np.float32), axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def _min_obstacle_clearance(self) -> float | None:
        if len(self.obstacles) == 0 or not self.state_history:
            return None
        states = np.asarray(self.state_history, dtype=np.float32)
        min_clearance = np.inf
        robot_radius = float(self.config["robot"]["radius"])
        for obstacle in self.obstacles:
            clearance = (
                np.linalg.norm(states[:, :2] - obstacle[:2], axis=1)
                - float(obstacle[2])
                - robot_radius
            )
            min_clearance = min(min_clearance, float(np.min(clearance)))
        return min_clearance

    def _control_metrics(self) -> dict:
        if not self.control_history:
            return {
                "control_smoothness": 0.0,
                "smooth_vx": 0.0,
                "smooth_vy": 0.0,
                "smooth_wz": 0.0,
                "control_jerk": 0.0,
                "jerk_vx": 0.0,
                "jerk_vy": 0.0,
                "jerk_wz": 0.0,
                "acceleration_cost": 0.0,
                "lateral_usage": 0.0,
                "yaw_rate_usage": 0.0,
                "vx_variance": 0.0,
                "vy_variance": 0.0,
                "wz_variance": 0.0,
            }
        controls = np.asarray(self.control_history, dtype=np.float64)
        variances = np.var(controls, axis=0)
        if len(controls) >= 2:
            deltas = np.diff(controls, axis=0)
            smooth_by_channel = np.mean(deltas * deltas, axis=0)
            smoothness = float(np.mean(np.sum(deltas * deltas, axis=1)))
            accelerations = deltas / self.dt
            acceleration_cost = float(np.sum(accelerations * accelerations))
        else:
            smooth_by_channel = np.zeros(3, dtype=np.float64)
            smoothness = 0.0
            acceleration_cost = 0.0
        if len(controls) >= 3:
            jerks = controls[2:] - 2.0 * controls[1:-1] + controls[:-2]
            jerk_by_channel = np.mean(jerks * jerks, axis=0)
            jerk = float(np.mean(np.sum(jerks * jerks, axis=1)))
        else:
            jerk_by_channel = np.zeros(3, dtype=np.float64)
            jerk = 0.0
        return {
            "control_smoothness": smoothness,
            "smooth_vx": float(smooth_by_channel[0]),
            "smooth_vy": float(smooth_by_channel[1]),
            "smooth_wz": float(smooth_by_channel[2]),
            "control_jerk": jerk,
            "jerk_vx": float(jerk_by_channel[0]),
            "jerk_vy": float(jerk_by_channel[1]),
            "jerk_wz": float(jerk_by_channel[2]),
            "acceleration_cost": acceleration_cost,
            "lateral_usage": float(np.mean(controls[:, 1] * controls[:, 1])),
            "yaw_rate_usage": float(np.mean(controls[:, 2] * controls[:, 2])),
            "vx_variance": float(variances[0]),
            "vy_variance": float(variances[1]),
            "wz_variance": float(variances[2]),
        }

    def _sample_coverage_metrics(self) -> dict:
        empty = {
            "sample_terminal_y_std_mean": 0.0,
            "sample_terminal_y_range_mean": 0.0,
            "sample_terminal_spread_mean": 0.0,
            "sample_terminal_x_range_mean": 0.0,
        }
        if not self.sample_u_history or not self.state_history:
            return empty
        y_stds = []
        y_ranges = []
        spreads = []
        x_ranges = []
        for frame, sampled_controls in enumerate(self.sample_u_history):
            if frame >= len(self.state_history) or len(sampled_controls) == 0:
                continue
            terminals = []
            for control_sequence in sampled_controls:
                predicted = self._predict_trajectory(self.state_history[frame], control_sequence)
                terminals.append(predicted[-1, :2])
            terminal_xy = np.asarray(terminals, dtype=np.float64)
            if terminal_xy.size == 0:
                continue
            y_values = terminal_xy[:, 1]
            x_values = terminal_xy[:, 0]
            center = np.mean(terminal_xy, axis=0)
            y_stds.append(float(np.std(y_values)))
            y_ranges.append(float(np.max(y_values) - np.min(y_values)))
            x_ranges.append(float(np.max(x_values) - np.min(x_values)))
            spreads.append(float(np.mean(np.linalg.norm(terminal_xy - center[None, :], axis=1))))
        if not y_stds:
            return empty
        return {
            "sample_terminal_y_std_mean": float(np.mean(y_stds)),
            "sample_terminal_y_range_mean": float(np.mean(y_ranges)),
            "sample_terminal_spread_mean": float(np.mean(spreads)),
            "sample_terminal_x_range_mean": float(np.mean(x_ranges)),
        }

    def _default_controller_factory(self, *, config: dict, runner: "OmniMppiSimulationRunner") -> object:
        return create_omni_controller(config, seed=123)
