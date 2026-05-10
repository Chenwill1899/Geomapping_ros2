#!/usr/bin/env python3
"""Evaluate a trained residual FDM with open-loop rollout replay."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import yaml

from mppi_controller.config import load_config
from mppi_controller.core.omni_b2 import OmniB2
from mppi_controller.core.residual_fdm_model import FEATURE_NAMES, ResidualFdmMlp, build_feature_vector
from mppi_controller.core.terrain import TerrainField
from mppi_controller.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller
from mppi_controller.visualization.utils import map_axis_limits_from_config


ResidualPredictor = Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]


@dataclass
class ResidualFdmPredictor:
    model: ResidualFdmMlp
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    device: torch.device
    checkpoint_path: Path
    normalization_path: Path

    def __call__(
        self,
        state: np.ndarray,
        command: np.ndarray,
        terrain_features: np.ndarray,
        terrain_risk: float,
    ) -> np.ndarray:
        feature = build_feature_vector(state, command, terrain_features, terrain_risk)
        standardized = ((feature - self.feature_mean) / self.feature_std).astype(np.float32, copy=False)
        tensor = torch.as_tensor(standardized.reshape(1, -1), dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(tensor).detach().cpu().numpy()[0]
        return (pred * self.target_std + self.target_mean).astype(np.float32, copy=False)


def load_residual_fdm_predictor(
    model_dir: str | Path,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    normalization_path: str | Path | None = None,
) -> ResidualFdmPredictor:
    model_dir = Path(model_dir)
    checkpoint_file = resolve_model_artifact_path(model_dir, checkpoint_path or "model.pt")
    normalization_file = resolve_model_artifact_path(model_dir, normalization_path or "normalization.npz")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    normalizer = np.load(normalization_file)
    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    if input_dim != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} input features, checkpoint has {input_dim}")
    torch_device = torch.device(device)
    model = ResidualFdmMlp(input_dim=input_dim, hidden_dim=hidden_dim).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return ResidualFdmPredictor(
        model=model,
        feature_mean=np.asarray(normalizer["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(normalizer["feature_std"], dtype=np.float32),
        target_mean=np.asarray(normalizer["target_mean"], dtype=np.float32),
        target_std=np.asarray(normalizer["target_std"], dtype=np.float32),
        device=torch_device,
        checkpoint_path=checkpoint_file,
        normalization_path=normalization_file,
    )


def resolve_model_artifact_path(model_dir: str | Path, artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return Path(model_dir) / path


def replay_controls(
    *,
    initial_state: np.ndarray,
    cmd_controls: np.ndarray,
    robot,
    terrain,
    residual_predictor: ResidualPredictor | None = None,
) -> dict[str, np.ndarray]:
    cmd_controls = np.asarray(cmd_controls, dtype=np.float32)
    states = np.zeros((len(cmd_controls) + 1, 6), dtype=np.float32)
    states[0] = np.asarray(initial_state, dtype=np.float32).reshape(6)
    predicted_residuals = np.zeros((len(cmd_controls), 3), dtype=np.float32)
    real_controls = np.zeros((len(cmd_controls), 3), dtype=np.float32)
    terrain_features_history = np.zeros((len(cmd_controls), 4), dtype=np.float32)
    terrain_risk_history = np.zeros(len(cmd_controls), dtype=np.float32)

    for idx, command in enumerate(cmd_controls):
        state = states[idx]
        features = terrain.feature(float(state[0]), float(state[1]))
        risk = float(terrain.risk_cost(float(state[0]), float(state[1]), features=features))
        if residual_predictor is None:
            residual = np.zeros(3, dtype=np.float32)
        else:
            residual = np.asarray(residual_predictor(state, command, features, risk), dtype=np.float32).reshape(3)
        real_control = robot.clip_control(np.asarray(command, dtype=np.float32) + residual)
        states[idx + 1] = robot.update_state(state, real_control)
        predicted_residuals[idx] = residual
        real_controls[idx] = real_control
        terrain_features_history[idx] = features
        terrain_risk_history[idx] = risk

    return {
        "states": states,
        "predicted_residuals": predicted_residuals,
        "real_controls": real_controls,
        "terrain_features": terrain_features_history,
        "terrain_risk": terrain_risk_history,
    }


def compute_rollout_metrics(
    *,
    oracle_states: np.ndarray,
    nominal_states: np.ndarray,
    learned_states: np.ndarray,
    oracle_residuals: np.ndarray,
    learned_residuals: np.ndarray,
    dt: float | None = None,
    horizons_s: Sequence[float] = (1.0, 2.0, 4.0),
) -> dict:
    oracle_states = np.asarray(oracle_states, dtype=np.float32)
    nominal_states = np.asarray(nominal_states, dtype=np.float32)
    learned_states = np.asarray(learned_states, dtype=np.float32)
    oracle_residuals = np.asarray(oracle_residuals, dtype=np.float32)
    learned_residuals = np.asarray(learned_residuals, dtype=np.float32)

    state_count = min(len(oracle_states), len(nominal_states), len(learned_states))
    residual_count = min(len(oracle_residuals), len(learned_residuals))
    oracle_s = oracle_states[:state_count]
    nominal_s = nominal_states[:state_count]
    learned_s = learned_states[:state_count]
    nominal_xy_error = np.linalg.norm(nominal_s[:, :2] - oracle_s[:, :2], axis=1)
    learned_xy_error = np.linalg.norm(learned_s[:, :2] - oracle_s[:, :2], axis=1)
    state_error = learned_s - oracle_s

    if residual_count:
        residual_error = learned_residuals[:residual_count] - oracle_residuals[:residual_count]
        residual_mse_axis = np.mean(residual_error * residual_error, axis=0)
        zero_residual_mse_axis = np.mean(oracle_residuals[:residual_count] ** 2, axis=0)
    else:
        residual_mse_axis = np.zeros(3, dtype=np.float32)
        zero_residual_mse_axis = np.zeros(3, dtype=np.float32)

    nominal_ade = float(np.mean(nominal_xy_error)) if state_count else 0.0
    learned_ade = float(np.mean(learned_xy_error)) if state_count else 0.0
    nominal_fde = float(nominal_xy_error[-1]) if state_count else 0.0
    learned_fde = float(learned_xy_error[-1]) if state_count else 0.0
    residual_mse = float(np.mean(residual_mse_axis))
    zero_residual_mse = float(np.mean(zero_residual_mse_axis))
    metrics = {
        "state_count": int(state_count),
        "residual_count": int(residual_count),
        "nominal_ade_xy": nominal_ade,
        "learned_ade_xy": learned_ade,
        "nominal_fde_xy": nominal_fde,
        "learned_fde_xy": learned_fde,
        "learned_vs_nominal_ade_improvement_pct": _improvement_pct(nominal_ade, learned_ade),
        "learned_vs_nominal_fde_improvement_pct": _improvement_pct(nominal_fde, learned_fde),
        "learned_state_rmse": float(np.sqrt(np.mean(state_error * state_error))) if state_count else 0.0,
        "residual_mse": residual_mse,
        "zero_residual_mse": zero_residual_mse,
        "residual_mse_axis": [float(v) for v in residual_mse_axis.tolist()],
        "zero_residual_mse_axis": [float(v) for v in zero_residual_mse_axis.tolist()],
        "residual_mse_improvement_pct": _improvement_pct(zero_residual_mse, residual_mse),
    }
    if dt is not None and float(dt) > 0.0:
        metrics.update(_horizon_error_metrics(nominal_xy_error, learned_xy_error, dt=float(dt), horizons_s=horizons_s))
    return metrics


def _horizon_error_metrics(
    nominal_xy_error: np.ndarray,
    learned_xy_error: np.ndarray,
    *,
    dt: float,
    horizons_s: Sequence[float],
) -> dict:
    metrics: dict = {"horizon_metrics": {}}
    state_count = min(len(nominal_xy_error), len(learned_xy_error))
    if state_count == 0:
        return metrics
    for horizon_s in horizons_s:
        key = _horizon_key(float(horizon_s))
        state_index = min(max(0, int(round(float(horizon_s) / dt))), state_count - 1)
        nominal_window = nominal_xy_error[: state_index + 1]
        learned_window = learned_xy_error[: state_index + 1]
        horizon_metrics = {
            "seconds": float(horizon_s),
            "state_index": int(state_index),
            "nominal_ade_xy": float(np.mean(nominal_window)),
            "learned_ade_xy": float(np.mean(learned_window)),
            "nominal_fde_xy": float(nominal_xy_error[state_index]),
            "learned_fde_xy": float(learned_xy_error[state_index]),
        }
        horizon_metrics["learned_vs_nominal_ade_improvement_pct"] = _improvement_pct(
            horizon_metrics["nominal_ade_xy"],
            horizon_metrics["learned_ade_xy"],
        )
        horizon_metrics["learned_vs_nominal_fde_improvement_pct"] = _improvement_pct(
            horizon_metrics["nominal_fde_xy"],
            horizon_metrics["learned_fde_xy"],
        )
        metrics["horizon_metrics"][key] = horizon_metrics
        for name, value in horizon_metrics.items():
            if name in {"seconds", "state_index"}:
                continue
            metrics[f"{name}_at_{key}"] = value
    return metrics


def _horizon_key(seconds: float) -> str:
    if abs(seconds - round(seconds)) <= 1e-9:
        return f"{int(round(seconds))}s"
    return f"{seconds:g}s"


def evaluate_residual_fdm_rollout(
    *,
    config_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    seed: int = 123,
    backend: str | None = None,
    device: str = "cpu",
    checkpoint: str | Path = "model.pt",
    normalization: str | Path = "normalization.npz",
    generate_gif: bool = True,
    gif_fps: int = 8,
    gif_max_frames: int = 180,
    command: str | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    config.setdefault("scenario", {})["random_seed"] = int(seed)
    config.setdefault("oracle_residual", {})["seed"] = int(seed)
    if backend is not None:
        config.setdefault("mppi", {})["backend"] = str(backend).lower()
    config["results"] = {
        **config.get("results", {}),
        "root": str(output_dir),
        "run_name": "oracle_run",
        "timestamp_suffix": False,
        "overwrite": True,
        "enable_plots": False,
        "enable_animation": False,
    }

    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *, config, runner: create_omni_controller(config, seed=seed),
    )
    summary = runner.run()
    oracle_states = _states_with_final(runner)
    cmd_controls = np.asarray(runner.cmd_control_history, dtype=np.float32)
    oracle_residuals = (
        np.asarray(runner.control_history, dtype=np.float32)
        - np.asarray(runner.cmd_control_history, dtype=np.float32)
    )

    replay_robot = _robot_from_config(config)
    terrain = TerrainField.from_config(config.get("terrain"))
    initial_state = oracle_states[0]
    nominal = replay_controls(
        initial_state=initial_state,
        cmd_controls=cmd_controls,
        robot=replay_robot,
        terrain=terrain,
        residual_predictor=None,
    )
    predictor = load_residual_fdm_predictor(
        model_dir,
        device=device,
        checkpoint_path=checkpoint,
        normalization_path=normalization,
    )
    learned = replay_controls(
        initial_state=initial_state,
        cmd_controls=cmd_controls,
        robot=_robot_from_config(config),
        terrain=terrain,
        residual_predictor=predictor,
    )
    metrics = compute_rollout_metrics(
        oracle_states=oracle_states,
        nominal_states=nominal["states"],
        learned_states=learned["states"],
        oracle_residuals=oracle_residuals,
        learned_residuals=learned["predicted_residuals"],
        dt=1.0 / float(config["simulation"]["sampling_rate"]),
    )
    metrics.update(
        {
            "config_path": str(config_path),
            "model_dir": str(model_dir),
            "output_dir": str(output_dir),
            "seed": int(seed),
            "backend": str(config["mppi"].get("backend", "numpy")).lower(),
            "parameter_snapshot": parameter_snapshot(config),
            **build_run_metadata(
                command=command,
                device=device,
                checkpoint_path=predictor.checkpoint_path,
                normalization_path=predictor.normalization_path,
                generate_gif=generate_gif,
                gif_fps=gif_fps,
                gif_max_frames=gif_max_frames,
            ),
            "oracle_results_path": str(summary.results_path),
            "oracle_reached_goal": bool(summary.reached_goal),
            "oracle_failed": bool(summary.failed),
            "oracle_steps": int(summary.steps),
        }
    )

    _save_rollout_arrays(output_dir, oracle_states, nominal, learned, oracle_residuals, cmd_controls)
    _plot_trajectory_comparison(output_dir, config, oracle_states, nominal["states"], learned["states"])
    _plot_residual_comparison(output_dir, oracle_residuals, learned["predicted_residuals"])
    if generate_gif:
        _save_rollout_comparison_gif(
            output_dir=output_dir,
            config=config,
            oracle_states=oracle_states,
            nominal_states=nominal["states"],
            learned_states=learned["states"],
            fps=gif_fps,
            max_frames=gif_max_frames,
        )
    metrics.update(rollout_gif_metrics(output_dir, generated_this_run=generate_gif))
    (output_dir / "rollout_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "rollout_metrics.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metrics, stream, sort_keys=False)
    return metrics


def build_run_metadata(
    *,
    command: str | None,
    device: str,
    checkpoint_path: str | Path,
    normalization_path: str | Path,
    generate_gif: bool,
    gif_fps: int,
    gif_max_frames: int,
    git_metadata: dict | None = None,
) -> dict:
    git = current_git_metadata() if git_metadata is None else git_metadata
    return {
        "command": command,
        "git_sha": git.get("sha"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "normalization_path": str(normalization_path),
        "gif_parameters": {
            "enabled": bool(generate_gif),
            "fps": int(gif_fps),
            "max_frames": int(gif_max_frames),
        },
    }


def current_git_metadata() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "sha": _git_output(repo_root, "rev-parse", "HEAD"),
        "branch": _git_output(repo_root, "branch", "--show-current"),
        "dirty": _git_dirty(repo_root),
    }


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = result.stdout.strip()
    return output or None


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def rollout_gif_metrics(output_dir: str | Path, *, generated_this_run: bool) -> dict:
    gif_path = Path(output_dir) / "rollout_compare.gif"
    generated = bool(generated_this_run)
    return {
        "rollout_compare_gif": str(gif_path) if generated and gif_path.exists() else None,
        "rollout_compare_gif_generated_this_run": generated and gif_path.exists(),
    }


def _states_with_final(runner: OmniMppiSimulationRunner) -> np.ndarray:
    states = [np.asarray(state, dtype=np.float32).copy() for state in runner.state_history]
    states.append(np.asarray(runner.state, dtype=np.float32).copy())
    return np.asarray(states, dtype=np.float32)


def parameter_snapshot(config: dict) -> dict:
    robot_cfg = config.get("robot", {})
    robot_radius = float(robot_cfg.get("radius", 0.0))
    safety_dist = float(robot_cfg.get("safety_dist", 0.0))
    obstacles = np.asarray(config.get("obstacles", {}).get("virtual", []), dtype=np.float32).reshape(-1, 7)
    obstacle_radii = [_rounded_float(v) for v in obstacles[:, 2].tolist()] if len(obstacles) else []
    boundary_radii = [_rounded_float(radius + robot_radius + safety_dist) for radius in obstacle_radii]
    return {
        "robot_radius": robot_radius,
        "robot_safety_dist": safety_dist,
        "obstacle_count": int(len(obstacles)),
        "obstacle_radii": obstacle_radii,
        "visualized_safety_boundary_formula": "obstacle_radius + robot.radius + robot.safety_dist",
        "visualized_safety_boundary_radii": boundary_radii,
    }


def _rounded_float(value: float) -> float:
    return round(float(value), 6)


def _robot_from_config(config: dict) -> OmniB2:
    sim = config["simulation"]
    robot_cfg = config["robot"]
    return OmniB2(
        dt=1.0 / float(sim["sampling_rate"]),
        max_vx=float(robot_cfg["max_vx"]),
        max_vy=float(robot_cfg["max_vy"]),
        max_wz=float(robot_cfg["max_wz"]),
    )


def _save_rollout_arrays(
    output_dir: Path,
    oracle_states: np.ndarray,
    nominal: dict[str, np.ndarray],
    learned: dict[str, np.ndarray],
    oracle_residuals: np.ndarray,
    cmd_controls: np.ndarray,
) -> None:
    np.savez(
        output_dir / "rollout_replay.npz",
        oracle_states=oracle_states,
        nominal_states=nominal["states"],
        learned_states=learned["states"],
        cmd_controls=cmd_controls,
        oracle_residuals=oracle_residuals,
        learned_residuals=learned["predicted_residuals"],
        learned_real_controls=learned["real_controls"],
        nominal_real_controls=nominal["real_controls"],
    )


def _plot_trajectory_comparison(
    output_dir: Path,
    config: dict,
    oracle_states: np.ndarray,
    nominal_states: np.ndarray,
    learned_states: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(oracle_states[:, 0], oracle_states[:, 1], label="oracle", color="black", linewidth=2.0)
    ax.plot(nominal_states[:, 0], nominal_states[:, 1], label="nominal replay", color="tab:blue", alpha=0.8)
    ax.plot(learned_states[:, 0], learned_states[:, 1], label="learned FDM replay", color="tab:orange", alpha=0.9)
    ax.scatter([oracle_states[0, 0]], [oracle_states[0, 1]], color="green", label="start", zorder=5)
    ax.scatter([oracle_states[-1, 0]], [oracle_states[-1, 1]], color="black", marker="x", label="oracle final", zorder=5)
    goal = np.asarray(config["simulation"]["goal"], dtype=np.float32)
    ax.scatter([goal[0]], [goal[1]], color="purple", marker="*", s=120, label="goal", zorder=5)
    for obstacle in np.asarray(config.get("obstacles", {}).get("virtual", []), dtype=np.float32).reshape(-1, 7):
        circle = plt.Circle((float(obstacle[0]), float(obstacle[1])), float(obstacle[2]), color="gray", alpha=0.25)
        ax.add_patch(circle)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Open-loop rollout replay vs oracle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(output_dir / "trajectory_compare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_comparison(
    output_dir: Path,
    oracle_residuals: np.ndarray,
    learned_residuals: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    names = ["vx", "vy", "wz"]
    count = min(len(oracle_residuals), len(learned_residuals))
    steps = np.arange(count)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for idx, name in enumerate(names):
        axes[idx].plot(steps, oracle_residuals[:count, idx], label="oracle exec residual", color="black")
        axes[idx].plot(steps, learned_residuals[:count, idx], label="learned residual", color="tab:orange", alpha=0.85)
        axes[idx].set_ylabel(f"du_{name}")
        axes[idx].grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_compare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_rollout_comparison_gif(
    *,
    output_dir: Path,
    config: dict,
    oracle_states: np.ndarray,
    nominal_states: np.ndarray,
    learned_states: np.ndarray,
    fps: int = 8,
    max_frames: int = 180,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    oracle_states = np.asarray(oracle_states, dtype=np.float32)
    nominal_states = np.asarray(nominal_states, dtype=np.float32)
    learned_states = np.asarray(learned_states, dtype=np.float32)
    state_count = min(len(oracle_states), len(nominal_states), len(learned_states))
    if state_count == 0:
        return

    frame_count = min(max(1, int(max_frames)), state_count)
    frame_indices = np.unique(np.linspace(0, state_count - 1, frame_count).astype(int))
    terrain = TerrainField.from_config(config.get("terrain"))
    xlim, ylim = map_axis_limits_from_config(config)
    risk_grid = _terrain_risk_grid(terrain, config, xlim, ylim)
    max_risk = float(np.max(risk_grid)) if risk_grid.size else 1.0
    if max_risk <= 1e-9:
        max_risk = 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    goal = np.asarray(config["simulation"]["goal"], dtype=np.float32)

    legend_handles = [
        Patch(facecolor="tab:red", alpha=0.25, label="terrain risk"),
        Line2D([0], [0], color="black", linewidth=2.0, label="oracle"),
        Line2D([0], [0], color="tab:blue", linewidth=1.7, label="nominal replay"),
        Line2D([0], [0], color="tab:orange", linewidth=1.9, label="learned FDM replay"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="green", markersize=6, label="start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="purple", markersize=10, label="goal"),
    ]

    def update(frame_idx: int):
        frame = int(frame_indices[frame_idx])
        ax.clear()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.imshow(
            risk_grid,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            cmap="plasma",
            alpha=0.35,
            aspect="auto",
            vmin=0.0,
            vmax=max_risk,
            zorder=0,
        )
        _draw_eval_obstacles(ax, config)
        ax.plot(oracle_states[: frame + 1, 0], oracle_states[: frame + 1, 1], color="black", linewidth=2.0, zorder=4)
        ax.plot(
            nominal_states[: frame + 1, 0],
            nominal_states[: frame + 1, 1],
            color="tab:blue",
            linewidth=1.7,
            alpha=0.85,
            zorder=3,
        )
        ax.plot(
            learned_states[: frame + 1, 0],
            learned_states[: frame + 1, 1],
            color="tab:orange",
            linewidth=1.9,
            alpha=0.95,
            zorder=5,
        )
        ax.scatter([oracle_states[0, 0]], [oracle_states[0, 1]], color="green", zorder=7)
        ax.scatter([goal[0]], [goal[1]], color="purple", marker="*", s=120, zorder=7)
        ax.scatter([oracle_states[frame, 0]], [oracle_states[frame, 1]], color="black", s=28, zorder=8)
        ax.scatter([nominal_states[frame, 0]], [nominal_states[frame, 1]], color="tab:blue", s=28, zorder=8)
        ax.scatter([learned_states[frame, 0]], [learned_states[frame, 1]], color="tab:orange", s=32, zorder=8)

        nominal_err = float(np.linalg.norm(nominal_states[frame, :2] - oracle_states[frame, :2]))
        learned_err = float(np.linalg.norm(learned_states[frame, :2] - oracle_states[frame, :2]))
        ax.set_title(
            f"Open-loop replay vs oracle | step {frame}/{state_count - 1}\n"
            f"XY error: nominal {nominal_err:.3f} m, learned {learned_err:.3f} m"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000.0 / max(1, int(fps)), blit=False)
    animation.save(output_dir / "rollout_compare.gif", writer=PillowWriter(fps=max(1, int(fps))))
    plt.close(fig)


def _terrain_risk_grid(
    terrain: TerrainField,
    config: dict,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> np.ndarray:
    resolution = int(config.get("visualization", {}).get("terrain_grid_resolution", 80))
    resolution = max(8, min(resolution, 160))
    grid_x = np.linspace(xlim[0], xlim[1], resolution)
    grid_y = np.linspace(ylim[0], ylim[1], resolution)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    risk_grid = np.zeros_like(mesh_x, dtype=np.float32)
    for row in range(mesh_x.shape[0]):
        for col in range(mesh_x.shape[1]):
            risk_grid[row, col] = terrain.risk_cost(float(mesh_x[row, col]), float(mesh_y[row, col]))
    return risk_grid


def _draw_eval_obstacles(ax, config: dict) -> None:
    import matplotlib.patches as patches

    robot_cfg = config.get("robot", {})
    robot_radius = float(robot_cfg.get("radius", 0.0))
    safety_dist = float(robot_cfg.get("safety_dist", 0.0))
    obstacles = np.asarray(config.get("obstacles", {}).get("virtual", []), dtype=np.float32).reshape(-1, 7)
    for obstacle in obstacles:
        x, y, radius = obstacle[:3]
        ax.add_patch(patches.Circle((float(x), float(y)), float(radius), color="gray", alpha=0.35, zorder=1))
        ax.add_patch(
            patches.Circle(
                (float(x), float(y)),
                float(radius) + robot_radius + safety_dist,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                zorder=2,
            )
        )


def _improvement_pct(baseline: float, candidate: float) -> float:
    if baseline <= 1e-12:
        return 0.0
    return float((1.0 - candidate / baseline) * 100.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--backend", choices=["cuda", "numpy"], default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="model.pt", help="Checkpoint filename under --model-dir or absolute path.")
    parser.add_argument("--normalization", default="normalization.npz", help="Normalizer filename under --model-dir or absolute path.")
    parser.add_argument("--no-gif", action="store_true", help="Skip rollout_compare.gif generation.")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--gif-max-frames", type=int, default=180)
    args = parser.parse_args(argv)

    metrics = evaluate_residual_fdm_rollout(
        config_path=args.config,
        model_dir=args.model_dir,
        output_dir=args.output,
        seed=args.seed,
        backend=args.backend,
        device=args.device,
        checkpoint=args.checkpoint,
        normalization=args.normalization,
        generate_gif=not args.no_gif,
        gif_fps=args.gif_fps,
        gif_max_frames=args.gif_max_frames,
        command=shell_join([sys.executable, *sys.argv]),
    )
    print(json.dumps(metrics, indent=2))


def shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in argv)


if __name__ == "__main__":
    main()
