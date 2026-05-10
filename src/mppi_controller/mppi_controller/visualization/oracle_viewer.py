"""Oracle residual-world diagnostics visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mppi_controller.core.terrain import TerrainField
from mppi_controller.visualization.utils import map_axis_limits_from_config


def _resolve_goal_xy(trajectory: pd.DataFrame, config: dict) -> tuple[float, float]:
    goal = config.get("simulation", {}).get("goal")
    if goal is not None and len(goal) >= 2:
        return float(goal[0]), float(goal[1])
    if {"x_des", "y_des"}.issubset(trajectory.columns) and not trajectory.empty:
        return float(trajectory["x_des"].iloc[-1]), float(trajectory["y_des"].iloc[-1])
    if not trajectory.empty:
        return float(trajectory["x"].iloc[-1]), float(trajectory["y"].iloc[-1])
    return 0.0, 0.0


def plot_oracle_diagnostics(results_path: Path, config: dict) -> None:
    results_path = Path(results_path)
    trajectory = pd.read_csv(results_path / "trajectory.csv")
    controls = pd.read_csv(results_path / "controls.csv")
    raw_controls = pd.read_csv(results_path / "raw_controls.csv")
    residuals = pd.read_csv(results_path / "residuals.csv")
    terrain = pd.read_csv(results_path / "terrain.csv")

    config_path = results_path / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or config

    xlim, ylim = map_axis_limits_from_config(config)
    terrain_field = TerrainField.from_config(config.get("terrain"))
    grid_resolution = int(config.get("visualization", {}).get("terrain_grid_resolution", 100))
    grid_x = np.linspace(xlim[0], xlim[1], grid_resolution)
    grid_y = np.linspace(ylim[0], ylim[1], grid_resolution)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    risk_grid = np.zeros_like(mesh_x, dtype=np.float32)
    for i in range(mesh_x.shape[0]):
        for j in range(mesh_x.shape[1]):
            risk_grid[i, j] = terrain_field.risk_cost(float(mesh_x[i, j]), float(mesh_y[i, j]))

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(10, 12))
    grid = GridSpec(6, 1, height_ratios=[3.0, 1.0, 1.0, 1.0, 0.8, 0.8], hspace=0.35)
    ax_map = fig.add_subplot(grid[0, 0])

    heatmap = ax_map.imshow(
        risk_grid,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        cmap="plasma",
        alpha=0.55,
        aspect="auto",
    )
    fig.colorbar(heatmap, ax=ax_map, fraction=0.046, pad=0.04, label="Terrain risk")

    du_norm = residuals["du_norm"].to_numpy(dtype=np.float32)
    action_count = min(len(trajectory), len(residuals), len(terrain))
    action_trajectory = trajectory.iloc[:action_count]
    scatter = ax_map.scatter(
        action_trajectory["x"],
        action_trajectory["y"],
        c=du_norm[:action_count],
        cmap="viridis",
        s=20,
        label="trajectory",
    )
    fig.colorbar(scatter, ax=ax_map, fraction=0.046, pad=0.08, label="Residual norm ||Δu||")

    obstacles = np.asarray(config.get("obstacles", {}).get("virtual", []), dtype=np.float32).reshape(-1, 7)
    robot_radius = float(config.get("robot", {}).get("radius", 0.6))
    safety_dist = float(config.get("robot", {}).get("safety_dist", 0.5))
    for obstacle in obstacles:
        ox, oy, radius = obstacle[:3]
        ax_map.add_patch(patches.Circle((ox, oy), radius, color="gray", alpha=0.4))
        ax_map.add_patch(
            patches.Circle(
                (ox, oy),
                radius + robot_radius + safety_dist,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=1.0,
            )
        )

    goal_x, goal_y = _resolve_goal_xy(trajectory, config)
    if not trajectory.empty:
        ax_map.scatter([trajectory["x"].iloc[0]], [trajectory["y"].iloc[0]], color="green", label="start")
        ax_map.scatter([goal_x], [goal_y], color="purple", marker="*", s=120, label="goal")
        ax_map.scatter(
            [trajectory["x"].iloc[-1]],
            [trajectory["y"].iloc[-1]],
            color="tab:orange",
            marker="x",
            s=60,
            label="end",
        )

    states = action_trajectory[["x", "y", "theta"]].to_numpy(dtype=np.float32)
    cmd_controls = residuals.iloc[:action_count][["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(dtype=np.float32)
    real_controls = residuals.iloc[:action_count][["real_vx", "real_vy", "real_wz"]].to_numpy(dtype=np.float32)
    stride = max(10, len(states) // 15 if len(states) else 10)
    for idx in range(0, len(states), stride):
        x, y, theta = states[idx]
        cos_theta = float(np.cos(theta))
        sin_theta = float(np.sin(theta))
        cmd = cmd_controls[idx]
        real = real_controls[idx]
        cmd_dx = cmd[0] * cos_theta - cmd[1] * sin_theta
        cmd_dy = cmd[0] * sin_theta + cmd[1] * cos_theta
        real_dx = real[0] * cos_theta - real[1] * sin_theta
        real_dy = real[0] * sin_theta + real[1] * cos_theta
        ax_map.arrow(
            x,
            y,
            cmd_dx,
            cmd_dy,
            color="white",
            linestyle="--",
            width=0.01,
            length_includes_head=True,
            alpha=0.7,
        )
        ax_map.arrow(
            x,
            y,
            real_dx,
            real_dy,
            color="black",
            width=0.01,
            length_includes_head=True,
            alpha=0.8,
        )

    ax_map.set_xlim(*xlim)
    ax_map.set_ylim(*ylim)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_title("Oracle residual diagnostics")
    ax_map.grid(True, alpha=0.25)
    ax_map.legend(loc="upper right")

    steps = np.arange(len(residuals))
    ax_vx = fig.add_subplot(grid[1, 0])
    ax_vy = fig.add_subplot(grid[2, 0], sharex=ax_vx)
    ax_wz = fig.add_subplot(grid[3, 0], sharex=ax_vx)
    ax_du = fig.add_subplot(grid[4, 0], sharex=ax_vx)
    ax_terrain = fig.add_subplot(grid[5, 0], sharex=ax_vx)

    ax_vx.plot(steps, residuals["cmd_vx"], label="vx_cmd", linestyle="--")
    ax_vx.plot(steps, residuals["real_vx"], label="vx_real")
    ax_vx.set_ylabel("vx")
    ax_vx.legend(loc="upper right")

    ax_vy.plot(steps, residuals["cmd_vy"], label="vy_cmd", linestyle="--")
    ax_vy.plot(steps, residuals["real_vy"], label="vy_real")
    ax_vy.set_ylabel("vy")
    ax_vy.legend(loc="upper right")

    ax_wz.plot(steps, residuals["cmd_wz"], label="wz_cmd", linestyle="--")
    ax_wz.plot(steps, residuals["real_wz"], label="wz_real")
    ax_wz.set_ylabel("wz")
    ax_wz.legend(loc="upper right")

    ax_du.plot(steps, du_norm, label="||delta_u||", color="tab:purple")
    ax_du.set_ylabel("residual")
    ax_du.legend(loc="upper right")

    terrain_steps = np.arange(len(terrain))
    ax_terrain.plot(terrain_steps, terrain["risk_cost"], label="risk", color="tab:red")
    ax_terrain.plot(terrain_steps, terrain["roughness"], label="roughness", color="tab:green")
    ax_terrain.plot(terrain_steps, terrain["friction"], label="friction", color="tab:blue")
    ax_terrain.set_ylabel("terrain")
    ax_terrain.set_xlabel("step")
    ax_terrain.legend(loc="upper right")

    fig.savefig(results_path / "oracle_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
