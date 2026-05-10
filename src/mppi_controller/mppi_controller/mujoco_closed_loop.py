"""ROS2 closed-loop adapter for controlling an ausim2 MuJoCo Scout."""

from __future__ import annotations

import json
import math
import time
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml

from mppi_controller.experiment import (
    build_experiment_config,
    collect_run_artifacts,
    validate_learned_fdm_artifacts,
)
from mppi_controller.simulation.omni_runner import create_omni_controller, goal_reached_xy
from mppi_controller.simulation.results_path import create_results_path
from mppi_controller.core.terrain import TerrainField


def main(argv: list[str] | None = None) -> int:
    from mppi_controller.cli import main as cli_main

    args = ["mujoco-closed-loop"]
    if argv is not None:
        args.extend(argv)
    return cli_main(args)


def body_yaw_from_quaternion(quaternion: Any) -> float:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def odom_message_to_state(message: Any) -> np.ndarray:
    pose = message.pose.pose
    twist = message.twist.twist
    return np.asarray(
        [
            float(pose.position.x),
            float(pose.position.y),
            body_yaw_from_quaternion(pose.orientation),
            float(twist.linear.x),
            float(twist.linear.y),
            float(twist.angular.z),
        ],
        dtype=np.float32,
    )


def project_scout_state(state: np.ndarray) -> np.ndarray:
    """Project an odometry state onto the differential-drive Scout model."""
    projected = np.asarray(state, dtype=np.float32).reshape(6).copy()
    projected[3:5] = _world_xy_velocity_to_body(projected)
    projected[4] = 0.0
    return projected


def project_mujoco_state(state: np.ndarray, drive_mode: str = "differential") -> np.ndarray:
    if str(drive_mode).lower() in {"omni", "omni_freejoint", "holonomic"}:
        projected = np.asarray(state, dtype=np.float32).reshape(6).copy()
        projected[3:5] = _world_xy_velocity_to_body(projected)
        return projected
    return project_scout_state(state)


def _world_xy_velocity_to_body(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(6)
    yaw = float(state[2])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_vx = float(state[3])
    world_vy = float(state[4])
    return np.asarray(
        [
            cos_yaw * world_vx + sin_yaw * world_vy,
            -sin_yaw * world_vx + cos_yaw * world_vy,
        ],
        dtype=np.float32,
    )


def initial_odom_timeout_expired(start_time: float, now: float, startup_timeout: float) -> bool:
    timeout = float(startup_timeout)
    if timeout <= 0.0:
        return False
    return float(now) - float(start_time) > timeout


def scout_twist_command(control: np.ndarray, twist_factory: Any | None = None):
    constrained = np.asarray(control, dtype=np.float32).reshape(3).copy()
    constrained[1] = 0.0
    twist = twist_factory() if twist_factory is not None else _simple_twist()
    twist.linear.x = float(constrained[0])
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = float(constrained[2])
    return twist, constrained


def omni_twist_command(control: np.ndarray, twist_factory: Any | None = None):
    constrained = np.asarray(control, dtype=np.float32).reshape(3).copy()
    twist = twist_factory() if twist_factory is not None else _simple_twist()
    twist.linear.x = float(constrained[0])
    twist.linear.y = float(constrained[1])
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = float(constrained[2])
    return twist, constrained


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = float(yaw) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def mujoco_twist_command(control: np.ndarray, drive_mode: str = "differential", twist_factory: Any | None = None):
    if str(drive_mode).lower() in {"omni", "omni_freejoint", "holonomic"}:
        return omni_twist_command(control, twist_factory=twist_factory)
    return scout_twist_command(control, twist_factory=twist_factory)


def angle_diff(target: float, current: float) -> float:
    return float((target - current + math.pi) % (2.0 * math.pi) - math.pi)


def final_approach_control(state: np.ndarray, goal: np.ndarray, config: dict[str, Any]) -> np.ndarray | None:
    cfg = config.get("final_controller", {})
    if not bool(cfg.get("enabled", False)):
        return None
    state = np.asarray(state, dtype=np.float32).reshape(6)
    goal = np.asarray(goal, dtype=np.float32).reshape(6)
    delta = goal[:2] - state[:2]
    distance = float(np.linalg.norm(delta))
    trigger_distance = float(cfg.get("trigger_distance", 2.0))
    if distance > trigger_distance:
        return None
    drive_mode = str(config.get("mujoco", {}).get("drive_mode", "differential")).lower()
    if drive_mode in {"omni", "omni_freejoint", "holonomic"}:
        max_vx = float(config.get("robot", {}).get("max_vx", 1.0))
        max_vy = float(config.get("robot", {}).get("max_vy", max_vx))
        max_wz = float(config.get("robot", {}).get("max_wz", 1.0))
        xy_gain = float(cfg.get("xy_gain", cfg.get("vx_gain", 0.7)))
        lateral_gain = float(cfg.get("lateral_gain", xy_gain))
        final_max_vy = min(max_vy, float(cfg.get("max_vy", max_vy)))
        wz_gain = float(cfg.get("wz_gain", 1.6))
        heading_gain = float(cfg.get("heading_gain", 0.0))
        final_yaw_gain = float(cfg.get("final_yaw_gain", 1.0))
        if distance <= 1e-4:
            heading_error = 0.0
        else:
            target_yaw = math.atan2(float(delta[1]), float(delta[0]))
            heading_error = angle_diff(target_yaw, float(state[2]))
        yaw_error = angle_diff(float(goal[2]), float(state[2]))
        cos_yaw = math.cos(float(state[2]))
        sin_yaw = math.sin(float(state[2]))
        body_dx = cos_yaw * float(delta[0]) + sin_yaw * float(delta[1])
        body_dy = -sin_yaw * float(delta[0]) + cos_yaw * float(delta[1])
        scale = min(distance / 0.5, 1.0) if distance < 0.5 else 1.0
        return np.asarray(
            [
                np.clip(xy_gain * body_dx, -max_vx, max_vx),
                np.clip(lateral_gain * body_dy, -final_max_vy, final_max_vy),
                np.clip(
                    wz_gain * (heading_gain * heading_error * scale + final_yaw_gain * yaw_error * scale),
                    -max_wz,
                    max_wz,
                ),
            ],
            dtype=np.float32,
        )
    target_yaw = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = angle_diff(target_yaw, float(state[2]))
    rotate_threshold = float(cfg.get("rotate_threshold", 0.6))
    max_vx = float(config.get("robot", {}).get("max_vx", 1.0))
    max_wz = float(config.get("robot", {}).get("max_wz", 1.0))
    vx_gain = float(cfg.get("vx_gain", 0.7))
    wz_gain = float(cfg.get("wz_gain", 1.6))
    vx = 0.0 if abs(heading_error) > rotate_threshold else vx_gain * distance * max(math.cos(heading_error), 0.0)
    wz = wz_gain * heading_error
    return np.asarray([np.clip(vx, 0.0, max_vx), 0.0, np.clip(wz, -max_wz, max_wz)], dtype=np.float32)


@dataclass(frozen=True)
class CommandFilterConfig:
    enabled: bool = False
    alpha: float = 0.25
    max_ax: float = 0.65
    max_ay: float = 0.65
    max_awz: float = 1.2
    drive_mode: str = "differential"
    lateral_scale: float = 1.0
    lateral_deadband: float = 0.0
    yaw_deadband: float = 0.0
    min_turn_vx: float = 0.0
    turn_wz_threshold: float = 0.0
    min_turn_vx_goal_distance: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CommandFilterConfig":
        cfg = config.get("command_filter", {})
        robot = config.get("robot", {})
        final_cfg = config.get("final_controller", {})
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            alpha=float(np.clip(cfg.get("alpha", 0.25), 0.0, 0.95)),
            max_ax=float(cfg.get("max_ax", robot.get("max_ax", 0.65))),
            max_ay=float(cfg.get("max_ay", robot.get("max_ay", 0.65))),
            max_awz=float(cfg.get("max_awz", robot.get("max_awz", 1.2))),
            drive_mode=str(config.get("mujoco", {}).get("drive_mode", cfg.get("drive_mode", "differential"))).lower(),
            lateral_scale=float(np.clip(cfg.get("lateral_scale", 1.0), 0.0, 1.0)),
            lateral_deadband=max(float(cfg.get("lateral_deadband", 0.0)), 0.0),
            yaw_deadband=max(float(cfg.get("yaw_deadband", 0.0)), 0.0),
            min_turn_vx=max(float(cfg.get("min_turn_vx", 0.0)), 0.0),
            turn_wz_threshold=max(float(cfg.get("turn_wz_threshold", 0.0)), 0.0),
            min_turn_vx_goal_distance=max(
                float(cfg.get("min_turn_vx_goal_distance", final_cfg.get("trigger_distance", 0.0))),
                0.0,
            ),
        )


def filter_mujoco_command(
    target: np.ndarray,
    previous: np.ndarray,
    cfg: CommandFilterConfig,
    dt: float,
    *,
    distance_to_goal: float | None = None,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).reshape(3).copy()
    previous = np.asarray(previous, dtype=np.float32).reshape(3)
    if cfg.drive_mode not in {"omni", "omni_freejoint", "holonomic"}:
        target[1] = 0.0
    else:
        target[1] *= cfg.lateral_scale
        if abs(float(target[1])) < cfg.lateral_deadband:
            target[1] = 0.0
    if abs(float(target[2])) < cfg.yaw_deadband:
        target[2] = 0.0
    if (
        cfg.min_turn_vx > 0.0
        and target[0] > 0.0
        and target[0] < cfg.min_turn_vx
        and abs(float(target[2])) >= cfg.turn_wz_threshold
        and (
            distance_to_goal is None
            or float(distance_to_goal) > float(cfg.min_turn_vx_goal_distance)
        )
    ):
        target[0] = float(cfg.min_turn_vx)
    if not cfg.enabled:
        return target
    blended = cfg.alpha * previous + (1.0 - cfg.alpha) * target
    max_delta = np.asarray([cfg.max_ax * dt, cfg.max_ay * dt, cfg.max_awz * dt], dtype=np.float32)
    if cfg.drive_mode not in {"omni", "omni_freejoint", "holonomic"}:
        max_delta[1] = 0.0
    delta = np.clip(blended - previous, -max_delta, max_delta)
    filtered = previous + delta
    if cfg.drive_mode not in {"omni", "omni_freejoint", "holonomic"}:
        filtered[1] = 0.0
    elif abs(float(filtered[1])) < cfg.lateral_deadband:
        filtered[1] = 0.0
    if abs(float(filtered[2])) < cfg.yaw_deadband:
        filtered[2] = 0.0
    return filtered.astype(np.float32, copy=False)


def filter_diff_drive_command(
    target: np.ndarray,
    previous: np.ndarray,
    cfg: CommandFilterConfig,
    dt: float,
) -> np.ndarray:
    return filter_mujoco_command(target, previous, cfg, dt)


def predict_omni_rollout(initial_state: np.ndarray, controls: np.ndarray, dt: float, max_control: np.ndarray) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float32).reshape(6).copy()
    controls = np.asarray(controls, dtype=np.float32).reshape(-1, 3)
    max_control = np.asarray(max_control, dtype=np.float32).reshape(3)
    states = np.zeros((len(controls) + 1, 6), dtype=np.float32)
    states[0] = state
    for idx, control in enumerate(controls):
        u = np.clip(control, -max_control, max_control)
        theta = float(state[2])
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        state[0] += (float(u[0]) * cos_theta - float(u[1]) * sin_theta) * float(dt)
        state[1] += (float(u[0]) * sin_theta + float(u[1]) * cos_theta) * float(dt)
        state[2] += float(u[2]) * float(dt)
        state[3:6] = u
        states[idx + 1] = state
    return states


def append_history_state(history: np.ndarray, state: np.ndarray, *, max_points: int) -> np.ndarray:
    history = np.asarray(history, dtype=np.float32).reshape(-1, 6)
    state = np.asarray(state, dtype=np.float32).reshape(1, 6)
    updated = np.vstack([history, state]) if history.size else state.copy()
    max_points = int(max_points)
    if max_points > 0 and len(updated) > max_points:
        updated = updated[-max_points:]
    return updated.astype(np.float32, copy=False)


def compute_executed_residual(measured_state: np.ndarray, commanded_control: np.ndarray) -> np.ndarray:
    measured = np.asarray(measured_state, dtype=np.float32).reshape(6)[3:6]
    command = np.asarray(commanded_control, dtype=np.float32).reshape(3)
    return (measured - command).astype(np.float32, copy=False)


def decode_grid_map_layer(grid_map_msg: Any, layer: str) -> np.ndarray:
    layers = list(getattr(grid_map_msg, "layers", []))
    if layer not in layers:
        return np.empty((0, 0), dtype=np.float32)
    data_msg = grid_map_msg.data[layers.index(layer)]
    data = np.asarray(data_msg.data, dtype=np.float32)
    dims = list(getattr(data_msg.layout, "dim", []))
    if len(dims) >= 2 and getattr(dims[0], "label", "") == "column_index":
        cols = int(dims[0].size)
        rows = int(dims[1].size)
        return data.reshape((rows, cols), order="F")
    if len(dims) >= 2 and getattr(dims[0], "label", "") == "row_index":
        rows = int(dims[0].size)
        cols = int(dims[1].size)
        return data.reshape((rows, cols), order="C")
    side = int(math.sqrt(data.size))
    if side * side != data.size:
        return np.empty((0, 0), dtype=np.float32)
    return data.reshape((side, side), order="C")


def grid_map_cell_centers(grid_map_msg: Any, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    info = grid_map_msg.info
    resolution = float(info.resolution)
    center_x = float(info.pose.position.x)
    center_y = float(info.pose.position.y)
    row_idx = np.arange(rows, dtype=np.float32)
    col_idx = np.arange(cols, dtype=np.float32)
    # grid_map convention: increasing row points toward -X, increasing column toward -Y.
    xs = center_x + ((rows * 0.5) - row_idx - 0.5) * resolution
    ys = center_y + ((cols * 0.5) - col_idx - 0.5) * resolution
    return xs, ys


def grid_map_traversability_obstacles(
    grid_map_msg: Any,
    *,
    state_xy: np.ndarray,
    layer: str = "traversability",
    mode: str = "traversability",
    elevation_layer: str = "elevation",
    elevation_threshold: float = 0.25,
    threshold: float = 0.55,
    obstacle_radius: float = 0.25,
    max_obstacles: int = 32,
    min_distance: float = 0.35,
    max_distance: float = 6.0,
    stride: int = 2,
) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "elevation":
        values = decode_grid_map_layer(grid_map_msg, elevation_layer)
    else:
        values = decode_grid_map_layer(grid_map_msg, layer)
    if values.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    values = values[::stride, ::stride]
    rows, cols = values.shape
    xs, ys = grid_map_cell_centers(grid_map_msg, rows * stride, cols * stride)
    xs = xs[::stride]
    ys = ys[::stride]
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    valid = np.isfinite(values)
    if mode == "elevation":
        unsafe = valid & (values > float(elevation_threshold))
    else:
        unsafe = valid & (values < float(threshold))
    if not np.any(unsafe):
        return np.empty((0, 7), dtype=np.float32)
    state_xy = np.asarray(state_xy, dtype=np.float32).reshape(2)
    centers = np.stack([xx[unsafe], yy[unsafe]], axis=1).astype(np.float32)
    distances = np.linalg.norm(centers - state_xy[None, :], axis=1)
    mask = (distances >= float(min_distance)) & (distances <= float(max_distance))
    centers = centers[mask]
    distances = distances[mask]
    if centers.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    order = np.argsort(distances)
    selected: list[np.ndarray] = []
    spacing = max(float(obstacle_radius) * 1.5, float(stride) * float(grid_map_msg.info.resolution))
    for index in order:
        center = centers[index]
        if any(np.linalg.norm(center - prev) < spacing for prev in selected):
            continue
        selected.append(center)
        if len(selected) >= int(max_obstacles):
            break
    if not selected:
        return np.empty((0, 7), dtype=np.float32)
    obstacles = np.zeros((len(selected), 7), dtype=np.float32)
    obstacles[:, :2] = np.asarray(selected, dtype=np.float32)
    obstacles[:, 2] = float(obstacle_radius)
    obstacles[:, 3] = float(obstacle_radius)
    return obstacles


def merge_static_and_map_obstacles(
    static_obstacles: np.ndarray,
    map_obstacles: np.ndarray,
    *,
    dedupe_distance: float = 0.6,
) -> np.ndarray:
    static = np.asarray(static_obstacles, dtype=np.float32).reshape(-1, 7)
    dynamic = np.asarray(map_obstacles, dtype=np.float32).reshape(-1, 7)
    if static.size == 0:
        return dynamic
    if dynamic.size == 0:
        return static
    keep = []
    extra = max(float(dedupe_distance), 0.0)
    for obstacle in dynamic:
        distances = np.linalg.norm(static[:, :2] - obstacle[:2], axis=1)
        duplicate_radius = static[:, 2] + float(obstacle[2]) + extra
        if not bool(np.any(distances <= duplicate_radius)):
            keep.append(obstacle)
    if not keep:
        return static
    return np.vstack([static, np.asarray(keep, dtype=np.float32)]).astype(np.float32, copy=False)


def ros_path_message_to_waypoints(path_msg: Any) -> np.ndarray:
    poses = list(getattr(path_msg, "poses", []))
    points = []
    for pose_stamped in poses:
        position = pose_stamped.pose.position
        points.append([float(position.x), float(position.y)])
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def polynome_message_to_waypoints(path_msg: Any) -> np.ndarray:
    points = []
    for point in list(getattr(path_msg, "pos_pts", [])):
        points.append([float(point.x), float(point.y)])
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def external_path_message_to_waypoints(path_msg: Any) -> np.ndarray:
    if hasattr(path_msg, "poses"):
        return ros_path_message_to_waypoints(path_msg)
    if hasattr(path_msg, "pos_pts"):
        return polynome_message_to_waypoints(path_msg)
    return np.empty((0, 2), dtype=np.float32)


def runtime_goal_required(config: dict[str, Any]) -> bool:
    goal_topic = config.get("goal_topic", {})
    return bool(goal_topic.get("enabled", False)) and bool(goal_topic.get("required", False))


def _goal_relief_config(config: dict[str, Any]) -> dict[str, Any] | None:
    terrain_cfg = config.get("terrain")
    if not isinstance(terrain_cfg, dict):
        return None
    relief_cfg = terrain_cfg.get("goal_relief")
    if not isinstance(relief_cfg, dict) or not bool(relief_cfg.get("enabled", False)):
        return None
    return relief_cfg


def _is_auto_goal_relief_center(value: Any) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def runtime_goal_relief_center_follows_goal(config: dict[str, Any]) -> bool:
    relief_cfg = _goal_relief_config(config)
    return relief_cfg is not None and _is_auto_goal_relief_center(relief_cfg.get("center"))


def _set_goal_relief_center(owner: Any, center: list[float], seen: set[int]) -> None:
    if owner is None:
        return
    terrain = owner if isinstance(owner, TerrainField) else getattr(owner, "terrain", None)
    if terrain is not None and id(terrain) not in seen:
        seen.add(id(terrain))
        relief = getattr(terrain, "goal_relief", None)
        if isinstance(relief, dict) and bool(relief.get("enabled", False)):
            relief["center"] = list(center)
    learned_dynamics = getattr(owner, "learned_dynamics", None)
    if learned_dynamics is not None and learned_dynamics is not owner:
        _set_goal_relief_center(learned_dynamics, center, seen)


def sync_runtime_goal_relief_center(
    config: dict[str, Any],
    goal: np.ndarray,
    *owners: Any,
    follow_goal: bool | None = None,
) -> bool:
    relief_cfg = _goal_relief_config(config)
    if relief_cfg is None:
        return False
    if follow_goal is None:
        follow_goal = _is_auto_goal_relief_center(relief_cfg.get("center"))
    if not bool(follow_goal):
        return False
    goal_xy = np.asarray(goal, dtype=np.float32).reshape(-1)
    if goal_xy.size < 2:
        return False
    center = [float(goal_xy[0]), float(goal_xy[1])]
    relief_cfg["center"] = list(center)
    seen: set[int] = set()
    for owner in owners:
        _set_goal_relief_center(owner, center, seen)
    return True


def optional_goal_from_config(config: dict[str, Any]) -> np.ndarray | None:
    if runtime_goal_required(config):
        return None
    goal = config.get("simulation", {}).get("goal")
    if goal is None:
        return None
    return np.asarray(goal, dtype=np.float32).reshape(6)


def pose_stamped_message_to_goal(message: Any) -> np.ndarray:
    pose = message.pose
    goal = np.zeros(6, dtype=np.float32)
    goal[0] = float(pose.position.x)
    goal[1] = float(pose.position.y)
    goal[2] = body_yaw_from_quaternion(pose.orientation)
    return goal


def path_terminal_goal(path: np.ndarray, state: np.ndarray) -> np.ndarray:
    waypoints = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    if len(waypoints) == 0:
        return np.asarray(state, dtype=np.float32).reshape(6).copy()
    goal = np.zeros(6, dtype=np.float32)
    goal[:2] = waypoints[-1]
    if len(waypoints) >= 2:
        tangent = waypoints[-1] - waypoints[-2]
    else:
        tangent = waypoints[-1] - np.asarray(state, dtype=np.float32).reshape(6)[:2]
    if float(np.linalg.norm(tangent)) > 1e-6:
        goal[2] = math.atan2(float(tangent[1]), float(tangent[0]))
    else:
        goal[2] = float(np.asarray(state, dtype=np.float32).reshape(6)[2])
    return goal


@dataclass(frozen=True)
class ExternalPathConfig:
    enabled: bool = False
    path_topic: str = "/smooth_path"
    message_type: str = "nav_msgs/Path"
    frame_id: str = "map"
    stale_timeout: float = 1.0
    stop_on_stale: bool = True
    lookahead: float = 2.0
    max_points: int = 48
    yaw_mode: str = "line_of_sight"
    smoothing_iterations: int = 0
    smoothing_alpha: float = 0.20
    temporal_alpha: float = 0.0
    temporal_endpoint_tolerance: float = 1.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExternalPathConfig":
        cfg = config.get("external_path", {})
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            path_topic=str(cfg.get("path_topic", "/smooth_path")),
            message_type=str(cfg.get("message_type", cfg.get("type", "nav_msgs/Path"))),
            frame_id=str(cfg.get("frame_id", "map")),
            stale_timeout=max(float(cfg.get("stale_timeout", 1.0)), 0.0),
            stop_on_stale=bool(cfg.get("stop_on_stale", True)),
            lookahead=max(float(cfg.get("lookahead", 2.0)), 0.2),
            max_points=max(int(cfg.get("max_points", 48)), 2),
            yaw_mode=str(cfg.get("yaw_mode", "line_of_sight")).lower(),
            smoothing_iterations=max(int(cfg.get("smoothing_iterations", 0)), 0),
            smoothing_alpha=float(np.clip(cfg.get("smoothing_alpha", 0.20), 0.0, 0.9)),
            temporal_alpha=float(np.clip(cfg.get("temporal_alpha", 0.0), 0.0, 0.95)),
            temporal_endpoint_tolerance=max(float(cfg.get("temporal_endpoint_tolerance", 1.0)), 0.0),
        )


@dataclass
class ExternalPathAdapter:
    cfg: ExternalPathConfig
    waypoints: np.ndarray | None = None
    last_time: float = 0.0

    def update(self, path_msg: Any, now: float | None = None) -> None:
        waypoints = external_path_message_to_waypoints(path_msg)
        if len(waypoints) >= 2 and self.cfg.smoothing_iterations > 0:
            waypoints = _laplacian_smooth_path(
                waypoints,
                iterations=self.cfg.smoothing_iterations,
                alpha=self.cfg.smoothing_alpha,
            )
        if len(waypoints) >= 2:
            waypoints = _limit_path_points(waypoints, max_points=self.cfg.max_points)
        if (
            self.waypoints is not None
            and len(self.waypoints) >= 2
            and len(waypoints) >= 2
            and self.cfg.temporal_alpha > 0.0
            and self._same_path_endpoint(waypoints)
        ):
            waypoints = self._blend_with_previous(waypoints)
        self.waypoints = waypoints
        self.last_time = time.monotonic() if now is None else float(now)

    def has_fresh_path(self, now: float | None = None) -> bool:
        if not self.cfg.enabled or self.waypoints is None or len(self.waypoints) < 2:
            return False
        current = time.monotonic() if now is None else float(now)
        return current - self.last_time <= self.cfg.stale_timeout

    def path(self, now: float | None = None) -> np.ndarray:
        if not self.has_fresh_path(now=now):
            return np.empty((0, 2), dtype=np.float32)
        assert self.waypoints is not None
        return self.waypoints.astype(np.float32, copy=False)

    def clear(self) -> None:
        self.waypoints = None
        self.last_time = 0.0

    def _same_path_endpoint(self, waypoints: np.ndarray) -> bool:
        assert self.waypoints is not None
        tolerance = float(self.cfg.temporal_endpoint_tolerance)
        if tolerance <= 0.0:
            return True
        previous_end = self.waypoints[-1]
        current_end = np.asarray(waypoints, dtype=np.float32).reshape(-1, 2)[-1]
        return bool(np.linalg.norm(previous_end - current_end) <= tolerance)

    def _blend_with_previous(self, waypoints: np.ndarray) -> np.ndarray:
        assert self.waypoints is not None
        current = np.asarray(waypoints, dtype=np.float32).reshape(-1, 2)
        previous = np.asarray(self.waypoints, dtype=np.float32).reshape(-1, 2)
        count = min(len(current), len(previous), int(self.cfg.max_points))
        if count < 2:
            return current
        current_limited = _limit_path_points(current, max_points=count)
        previous_limited = _limit_path_points(previous, max_points=count)
        alpha = float(self.cfg.temporal_alpha)
        blended = alpha * previous_limited + (1.0 - alpha) * current_limited
        blended[0] = current_limited[0]
        blended[-1] = current_limited[-1]
        return blended.astype(np.float32, copy=False)


def select_external_path_goal(
    state: np.ndarray,
    global_goal: np.ndarray,
    path: np.ndarray,
    cfg: ExternalPathConfig,
) -> tuple[np.ndarray, np.ndarray]:
    waypoints = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    goal = select_path_lookahead_goal(
        state,
        global_goal,
        waypoints,
        cfg.lookahead,
        yaw_mode=cfg.yaw_mode,
    )
    return goal, _limit_path_points(waypoints, max_points=cfg.max_points)


@dataclass(frozen=True)
class MujocoClosedLoopSummary:
    steps: int
    reached_goal: bool
    failed: bool
    results_path: Path
    run_time: float


@dataclass
class ElevationMapObstacleAdapter:
    enabled: bool = False
    required: bool = False
    layer: str = "traversability"
    mode: str = "traversability"
    elevation_layer: str = "elevation"
    elevation_threshold: float = 0.25
    threshold: float = 0.55
    obstacle_radius: float = 0.25
    max_obstacles: int = 32
    min_distance: float = 0.35
    max_distance: float = 6.0
    stride: int = 2
    stale_timeout: float = 1.0
    static_dedupe_distance: float = 0.6
    last_msg: Any | None = None
    last_time: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ElevationMapObstacleAdapter":
        cfg = config.get("elevation_map", {})
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            required=bool(cfg.get("required", False)),
            layer=str(cfg.get("layer", "traversability")),
            mode=str(cfg.get("mode", "traversability")),
            elevation_layer=str(cfg.get("elevation_layer", "elevation")),
            elevation_threshold=float(cfg.get("elevation_threshold", 0.25)),
            threshold=float(cfg.get("traversability_threshold", cfg.get("threshold", 0.55))),
            obstacle_radius=float(cfg.get("obstacle_radius", 0.25)),
            max_obstacles=int(cfg.get("max_obstacles", 32)),
            min_distance=float(cfg.get("min_distance", 0.35)),
            max_distance=float(cfg.get("max_distance", 6.0)),
            stride=max(int(cfg.get("stride", 2)), 1),
            stale_timeout=float(cfg.get("stale_timeout", 1.0)),
            static_dedupe_distance=max(float(cfg.get("static_dedupe_distance", 0.6)), 0.0),
        )

    def update(self, msg: Any, now: float | None = None) -> None:
        self.last_msg = msg
        self.last_time = time.monotonic() if now is None else float(now)

    def obstacles_for_state(self, state: np.ndarray, now: float | None = None) -> np.ndarray:
        if not self.has_fresh_map(now=now):
            return np.empty((0, 7), dtype=np.float32)
        return grid_map_traversability_obstacles(
            self.last_msg,
            state_xy=np.asarray(state, dtype=np.float32).reshape(6)[:2],
            layer=self.layer,
            mode=self.mode,
            elevation_layer=self.elevation_layer,
            elevation_threshold=self.elevation_threshold,
            threshold=self.threshold,
            obstacle_radius=self.obstacle_radius,
            max_obstacles=self.max_obstacles,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            stride=self.stride,
        )

    def has_fresh_map(self, now: float | None = None) -> bool:
        if not self.enabled or self.last_msg is None:
            return False
        current = time.monotonic() if now is None else float(now)
        return current - self.last_time <= self.stale_timeout


@dataclass(frozen=True)
class GroundArtifactFilterConfig:
    enabled: bool = False
    min_cost: float = 20.0
    max_height_abs: float = 0.08
    max_roughness: float = 0.05
    max_slope: float = 0.05
    occupied_threshold: int = 50
    clear_value: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GroundArtifactFilterConfig":
        cfg = config.get("ground_artifact_filter", {})
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            min_cost=max(float(cfg.get("min_cost", 20.0)), 0.0),
            max_height_abs=max(float(cfg.get("max_height_abs", 0.08)), 0.0),
            max_roughness=max(float(cfg.get("max_roughness", 0.05)), 0.0),
            max_slope=max(float(cfg.get("max_slope", 0.05)), 0.0),
            occupied_threshold=int(cfg.get("occupied_threshold", 50)),
            clear_value=max(float(cfg.get("clear_value", 0.0)), 0.0),
        )


@dataclass(frozen=True)
class LocalCostmapFootprintConfig:
    enabled: bool = False
    radius: float = 0.35
    safety_margin: float = 0.05
    sample_count: int = 8

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LocalCostmapFootprintConfig":
        cfg = config.get("footprint", {})
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            radius=max(float(cfg.get("radius", 0.35)), 0.0),
            safety_margin=max(float(cfg.get("safety_margin", 0.05)), 0.0),
            sample_count=max(int(cfg.get("sample_count", 8)), 4),
        )


@dataclass(frozen=True)
class LocalCostmapConfig:
    enabled: bool = False
    required: bool = False
    topic: str = "/msg_local_reward"
    layer: str = "reward_cost"
    stale_timeout: float = 1.0
    cost_weight: float = 35.0
    cost_power: float = 2.0
    unknown_cost: float = 100.0
    max_cost: float = 100.0
    cost_threshold: float = 0.0
    unknown_clear_radius: float = 1.0
    unknown_clear_value: float = 0.0
    ground_artifact_filter: GroundArtifactFilterConfig = field(default_factory=GroundArtifactFilterConfig)
    footprint: LocalCostmapFootprintConfig = field(default_factory=LocalCostmapFootprintConfig)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LocalCostmapConfig":
        cfg = config.get("local_costmap", {})
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            required=bool(cfg.get("required", False)),
            topic=str(cfg.get("topic", "/msg_local_reward")),
            layer=str(cfg.get("layer", "reward_cost")),
            stale_timeout=max(float(cfg.get("stale_timeout", 1.0)), 0.0),
            cost_weight=max(float(cfg.get("cost_weight", 35.0)), 0.0),
            cost_power=max(float(cfg.get("cost_power", 2.0)), 0.1),
            unknown_cost=max(float(cfg.get("unknown_cost", 100.0)), 0.0),
            max_cost=max(float(cfg.get("max_cost", 100.0)), 1e-6),
            cost_threshold=max(float(cfg.get("cost_threshold", 0.0)), 0.0),
            unknown_clear_radius=max(float(cfg.get("unknown_clear_radius", 1.0)), 0.0),
            unknown_clear_value=max(float(cfg.get("unknown_clear_value", 0.0)), 0.0),
            ground_artifact_filter=GroundArtifactFilterConfig.from_config(cfg),
            footprint=LocalCostmapFootprintConfig.from_config(cfg),
        )


@dataclass
class LocalCostmapAdapter:
    cfg: LocalCostmapConfig
    last_msg: Any | None = None
    last_time: float = 0.0

    def update(self, msg: Any, now: float | None = None) -> None:
        self.last_msg = msg
        self.last_time = time.monotonic() if now is None else float(now)

    def has_fresh_map(self, now: float | None = None) -> bool:
        if not self.cfg.enabled or self.last_msg is None:
            return False
        current = time.monotonic() if now is None else float(now)
        return current - self.last_time <= self.cfg.stale_timeout

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        if not self.has_fresh_map(now=now):
            return {"enabled": False}
        msg = self.last_msg
        info = msg.occupancy.info
        width = int(info.width)
        height = int(info.height)
        resolution = float(info.resolution)
        cell_count = width * height
        values = self._layer_values(msg)
        if width <= 0 or height <= 0 or resolution <= 0.0 or values.size != cell_count:
            return {"enabled": False}
        unknown_mask = self._unknown_mask(msg, values, cell_count)
        data = np.asarray(values, dtype=np.float32).reshape(-1).copy()
        invalid_values = ~np.isfinite(data)
        unknown_mask |= invalid_values
        data[invalid_values] = float(self.cfg.unknown_cost)
        raw_data = data.copy()
        if self.cfg.cost_threshold > 0.0:
            data[data < self.cfg.cost_threshold] = 0.0
        if self.cfg.cost_threshold > 0.0 and width > 2 and height > 2:
            grid = data.reshape(height, width)
            # 3x3 average smoothing to remove scan-ring artifacts
            smoothed = np.zeros_like(grid)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    src_i = np.clip(np.arange(height) + di, 0, height - 1)
                    src_j = np.clip(np.arange(width) + dj, 0, width - 1)
                    smoothed += grid[src_i[:, None], src_j[None, :]]
            grid = smoothed / 9.0
            # re-apply threshold after smoothing
            grid[grid < self.cfg.cost_threshold] = 0.0
            data = grid.reshape(-1)
        data, artifact_cells = self._filter_ground_artifacts(msg, data, unknown_mask, cell_count)
        return {
            "enabled": True,
            "origin": np.asarray(
                [float(info.origin.position.x), float(info.origin.position.y)],
                dtype=np.float32,
            ),
            "resolution": resolution,
            "width": width,
            "height": height,
            "data": data,
            "unknown_mask": unknown_mask,
            "weight": float(self.cfg.cost_weight),
            "power": float(self.cfg.cost_power),
            "unknown_cost": float(self.cfg.unknown_cost),
            "max_cost": float(self.cfg.max_cost),
            "unknown_clear_radius": float(self.cfg.unknown_clear_radius),
            "unknown_clear_value": float(self.cfg.unknown_clear_value),
            "ground_artifact_cells": int(artifact_cells),
            "raw_mean_cost": float(np.mean(raw_data)) if raw_data.size else 0.0,
            "raw_max_cost": float(np.max(raw_data)) if raw_data.size else 0.0,
            "raw_high_cost_ratio": float(np.mean(raw_data >= self.cfg.ground_artifact_filter.min_cost)) if raw_data.size else 0.0,
            "filtered_mean_cost": float(np.mean(data)) if data.size else 0.0,
            "filtered_max_cost": float(np.max(data)) if data.size else 0.0,
            "filtered_high_cost_ratio": float(np.mean(data >= self.cfg.ground_artifact_filter.min_cost)) if data.size else 0.0,
            "footprint_enabled": bool(self.cfg.footprint.enabled),
            "footprint_radius": float(self.cfg.footprint.radius),
            "footprint_safety_margin": float(self.cfg.footprint.safety_margin),
            "footprint_sample_count": int(self.cfg.footprint.sample_count),
        }

    def _layer_values(self, msg: Any) -> np.ndarray:
        layer = str(self.cfg.layer)
        if layer in {"occupancy", "occupancy_data"}:
            return np.asarray(msg.occupancy.data, dtype=np.float32)
        if not hasattr(msg, layer):
            return np.empty(0, dtype=np.float32)
        return np.asarray(getattr(msg, layer), dtype=np.float32)

    def _unknown_mask(self, msg: Any, values: np.ndarray, cell_count: int) -> np.ndarray:
        mask = np.zeros(cell_count, dtype=bool)
        occupancy = np.asarray(getattr(msg.occupancy, "data", []), dtype=np.int16).reshape(-1)
        if occupancy.size == cell_count:
            mask |= occupancy < 0
        if str(self.cfg.layer) in {"occupancy", "occupancy_data"}:
            value_arr = np.asarray(values, dtype=np.float32).reshape(-1)
            mask |= value_arr < 0.0
        return mask

    def _filter_ground_artifacts(
        self,
        msg: Any,
        data: np.ndarray,
        unknown_mask: np.ndarray,
        cell_count: int,
    ) -> tuple[np.ndarray, int]:
        cfg = self.cfg.ground_artifact_filter
        if not cfg.enabled or data.size != cell_count:
            return data, 0
        height = self._message_float_layer(msg, "height", cell_count)
        roughness = self._message_float_layer(msg, "roughness", cell_count)
        slope = self._message_float_layer(msg, "cost_map", cell_count)
        occupancy = np.asarray(getattr(msg.occupancy, "data", []), dtype=np.int16).reshape(-1)
        if height is None or roughness is None or slope is None or occupancy.size != cell_count:
            return data, 0
        observed = (~unknown_mask) & (occupancy >= 0) & (occupancy < int(cfg.occupied_threshold))
        flat_ground = (
            observed
            & (data >= float(cfg.min_cost))
            & (np.abs(height) <= float(cfg.max_height_abs))
            & (roughness <= float(cfg.max_roughness))
            & (slope <= float(cfg.max_slope))
        )
        if not np.any(flat_ground):
            return data, 0
        filtered = data.copy()
        filtered[flat_ground] = float(cfg.clear_value)
        return filtered, int(np.count_nonzero(flat_ground))

    @staticmethod
    def _message_float_layer(msg: Any, name: str, cell_count: int) -> np.ndarray | None:
        if not hasattr(msg, name):
            return None
        values = np.asarray(getattr(msg, name), dtype=np.float32).reshape(-1)
        if values.size != cell_count:
            return None
        return values


@dataclass(frozen=True)
class LocalGoalConfig:
    enabled: bool = False
    lookahead: float = 6.0
    lateral_offsets: tuple[float, ...] = (0.0, 1.4, -1.4, 2.4, -2.4)
    recenter_gain: float = 0.25
    corridor_buffer: float = 0.45
    robot_radius: float = 0.35
    safety_dist: float = 0.10
    obstacle_weight: float = 80.0
    lateral_weight: float = 0.15
    goal_weight: float = 0.05
    switch_distance: float = 2.0
    final_approach_distance: float = 6.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LocalGoalConfig":
        cfg = config.get("local_goal", {})
        robot = config.get("robot", {})
        offsets = tuple(float(v) for v in cfg.get("lateral_offsets", [0.0, 1.4, -1.4, 2.4, -2.4]))
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            lookahead=float(cfg.get("lookahead", 6.0)),
            lateral_offsets=offsets,
            recenter_gain=float(np.clip(cfg.get("recenter_gain", 0.25), 0.0, 1.0)),
            corridor_buffer=float(cfg.get("corridor_buffer", 0.45)),
            robot_radius=float(cfg.get("robot_radius", robot.get("radius", 0.35))),
            safety_dist=float(cfg.get("safety_dist", robot.get("safety_dist", 0.10))),
            obstacle_weight=float(cfg.get("obstacle_weight", 80.0)),
            lateral_weight=float(cfg.get("lateral_weight", 0.15)),
            goal_weight=float(cfg.get("goal_weight", 0.05)),
            switch_distance=float(cfg.get("switch_distance", 2.0)),
            final_approach_distance=float(cfg.get("final_approach_distance", 6.0)),
        )


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    point = np.asarray(point, dtype=np.float32).reshape(2)
    start = np.asarray(start, dtype=np.float32).reshape(2)
    end = np.asarray(end, dtype=np.float32).reshape(2)
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-9:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def select_obstacle_aware_goal(
    state: np.ndarray,
    global_goal: np.ndarray,
    obstacles: np.ndarray,
    cfg: LocalGoalConfig,
) -> np.ndarray:
    goal = np.asarray(global_goal, dtype=np.float32).reshape(6).copy()
    if not cfg.enabled or obstacles.size == 0:
        return goal
    state_xy = np.asarray(state, dtype=np.float32).reshape(6)[:2]
    goal_xy = goal[:2]
    to_goal = goal_xy - state_xy
    distance_to_goal = float(np.linalg.norm(to_goal))
    if distance_to_goal <= max(cfg.switch_distance, cfg.final_approach_distance):
        if distance_to_goal > cfg.switch_distance:
            goal[2] = math.atan2(float(to_goal[1]), float(to_goal[0]))
        return goal
    direction = to_goal / max(distance_to_goal, 1e-6)
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    lookahead = min(float(cfg.lookahead), distance_to_goal)
    base_goal = state_xy + direction * lookahead
    # In the Scout random-obstacle task the global goal often lies behind the
    # obstacle row. Re-centering too aggressively pulls the local target out of
    # a clear bypass lane, so only bleed lateral offset back toward the global
    # line gradually.
    base_goal[1] = state_xy[1] + direction[1] * lookahead * cfg.recenter_gain
    obstacles = np.asarray(obstacles, dtype=np.float32).reshape(-1, 7)

    best_xy = base_goal
    best_score = float("inf")
    for offset in cfg.lateral_offsets:
        candidate_xy = base_goal + lateral * float(offset)
        score = cfg.lateral_weight * float(offset * offset) + cfg.goal_weight * float(
            np.sum((candidate_xy - goal_xy) ** 2)
        )
        for obstacle in obstacles:
            clearance = (
                point_segment_distance(obstacle[:2], state_xy, candidate_xy)
                - float(obstacle[2])
                - cfg.robot_radius
                - cfg.safety_dist
            )
            margin = cfg.corridor_buffer - clearance
            if margin > 0.0:
                score += cfg.obstacle_weight * margin * margin
        if score < best_score:
            best_score = score
            best_xy = candidate_xy

    local_goal = goal.copy()
    local_goal[:2] = best_xy.astype(np.float32)
    local_delta = best_xy - state_xy
    if float(np.linalg.norm(local_delta)) > 1e-6:
        local_goal[2] = math.atan2(float(local_delta[1]), float(local_delta[0]))
    return local_goal


@dataclass(frozen=True)
class GlobalPathConfig:
    enabled: bool = False
    resolution: float = 0.25
    padding: float = 3.0
    lookahead: float = 3.0
    obstacle_inflation: float = 0.25
    replan_steps: int = 10
    replan_distance: float = 1.0
    max_grid_cells: int = 160000
    simplify_stride: int = 3
    smoothing_iterations: int = 0
    smoothing_alpha: float = 0.25
    smoothing_sample_resolution: float = 0.10
    cost_max_points: int = 48
    yaw_mode: str = "line_of_sight"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GlobalPathConfig":
        cfg = config.get("global_path", {})
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            resolution=max(float(cfg.get("resolution", 0.25)), 0.05),
            padding=max(float(cfg.get("padding", 3.0)), 0.5),
            lookahead=max(float(cfg.get("lookahead", 3.0)), 0.2),
            obstacle_inflation=max(float(cfg.get("obstacle_inflation", 0.25)), 0.0),
            replan_steps=max(int(cfg.get("replan_steps", 10)), 1),
            replan_distance=max(float(cfg.get("replan_distance", 1.0)), 0.0),
            max_grid_cells=max(int(cfg.get("max_grid_cells", 160000)), 1000),
            simplify_stride=max(int(cfg.get("simplify_stride", 3)), 1),
            smoothing_iterations=max(int(cfg.get("smoothing_iterations", 0)), 0),
            smoothing_alpha=float(np.clip(cfg.get("smoothing_alpha", 0.25), 0.0, 0.9)),
            smoothing_sample_resolution=max(float(cfg.get("smoothing_sample_resolution", 0.10)), 0.02),
            cost_max_points=max(int(cfg.get("cost_max_points", 48)), 2),
            yaw_mode=str(cfg.get("yaw_mode", "line_of_sight")).lower(),
        )


def plan_global_path_astar(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles: np.ndarray,
    cfg: GlobalPathConfig,
    *,
    robot_radius: float,
    safety_dist: float,
) -> np.ndarray:
    start_xy = np.asarray(start_xy, dtype=np.float32).reshape(2)
    goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    obstacles = np.asarray(obstacles, dtype=np.float32).reshape(-1, 7)
    points = [start_xy, goal_xy]
    if obstacles.size:
        points.append(obstacles[:, :2])
    all_points = np.vstack(points)
    min_xy = np.min(all_points, axis=0) - float(cfg.padding)
    max_xy = np.max(all_points, axis=0) + float(cfg.padding)
    resolution = float(cfg.resolution)
    dims = np.ceil((max_xy - min_xy) / resolution).astype(int) + 1
    if int(dims[0] * dims[1]) > int(cfg.max_grid_cells):
        scale = math.sqrt(float(dims[0] * dims[1]) / float(cfg.max_grid_cells))
        resolution *= scale
        dims = np.ceil((max_xy - min_xy) / resolution).astype(int) + 1
    occupied = np.zeros((int(dims[0]), int(dims[1])), dtype=bool)
    for obstacle in obstacles:
        radius = float(obstacle[2]) + float(robot_radius) + float(safety_dist) + float(cfg.obstacle_inflation)
        center_idx = np.rint((obstacle[:2] - min_xy) / resolution).astype(int)
        radius_cells = int(math.ceil(radius / resolution))
        x0 = max(center_idx[0] - radius_cells, 0)
        x1 = min(center_idx[0] + radius_cells + 1, occupied.shape[0])
        y0 = max(center_idx[1] - radius_cells, 0)
        y1 = min(center_idx[1] + radius_cells + 1, occupied.shape[1])
        xs = (np.arange(x0, x1, dtype=np.float32) * resolution + min_xy[0]) - float(obstacle[0])
        ys = (np.arange(y0, y1, dtype=np.float32) * resolution + min_xy[1]) - float(obstacle[1])
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        occupied[x0:x1, y0:y1] |= (xx * xx + yy * yy) <= radius * radius
    start_idx = tuple(np.rint((start_xy - min_xy) / resolution).astype(int))
    goal_idx = tuple(np.rint((goal_xy - min_xy) / resolution).astype(int))
    start_idx = _nearest_free_cell(occupied, start_idx)
    goal_idx = _nearest_free_cell(occupied, goal_idx)
    if start_idx is None or goal_idx is None:
        return np.empty((0, 2), dtype=np.float32)
    index_path = _astar_grid(occupied, start_idx, goal_idx)
    if not index_path:
        return np.empty((0, 2), dtype=np.float32)
    path = np.asarray([[min_xy[0] + ix * resolution, min_xy[1] + iy * resolution] for ix, iy in index_path], dtype=np.float32)
    path[0] = start_xy
    path[-1] = goal_xy
    path = _simplify_path(path, stride=cfg.simplify_stride)
    return _smooth_path_with_clearance(
        path,
        obstacles,
        clearance_margin=float(robot_radius) + float(safety_dist),
        iterations=cfg.smoothing_iterations,
        alpha=cfg.smoothing_alpha,
        sample_resolution=cfg.smoothing_sample_resolution,
    )


def select_path_lookahead_goal(
    state: np.ndarray,
    global_goal: np.ndarray,
    path: np.ndarray,
    lookahead: float,
    *,
    yaw_mode: str = "line_of_sight",
) -> np.ndarray:
    goal = np.asarray(global_goal, dtype=np.float32).reshape(6).copy()
    path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    if len(path) == 0:
        return goal
    state_xy = np.asarray(state, dtype=np.float32).reshape(6)[:2]
    start_idx, projected, _distance = project_point_to_path(state_xy, path)
    remaining = float(lookahead)
    target = path[-1]
    tangent = path[-1] - path[-2] if len(path) >= 2 else target - state_xy
    if start_idx < len(path) - 1:
        segment_tail = path[start_idx + 1] - projected
        tail_length = float(np.linalg.norm(segment_tail))
        if tail_length > 1e-6:
            if remaining <= tail_length:
                target = projected + segment_tail * (remaining / tail_length)
                goal[:2] = target.astype(np.float32)
                _set_path_goal_yaw(goal, state_xy, segment_tail, yaw_mode=yaw_mode)
                return goal
            remaining -= tail_length
    for idx in range(start_idx + 1, len(path) - 1):
        segment = path[idx + 1] - path[idx]
        length = float(np.linalg.norm(segment))
        if length <= 1e-6:
            continue
        if remaining <= length:
            target = path[idx] + segment * (remaining / length)
            tangent = segment
            break
        remaining -= length
    goal[:2] = target.astype(np.float32)
    _set_path_goal_yaw(goal, state_xy, tangent, yaw_mode=yaw_mode)
    return goal


def _set_path_goal_yaw(goal: np.ndarray, state_xy: np.ndarray, tangent: np.ndarray, *, yaw_mode: str) -> None:
    if str(yaw_mode).lower() in {"path_tangent", "tangent"} and float(np.linalg.norm(tangent)) > 1e-6:
        goal[2] = math.atan2(float(tangent[1]), float(tangent[0]))
        return
    delta = goal[:2] - state_xy
    if float(np.linalg.norm(delta)) > 1e-6:
        goal[2] = math.atan2(float(delta[1]), float(delta[0]))


def project_point_to_path(point: np.ndarray, path: np.ndarray) -> tuple[int, np.ndarray, float]:
    point = np.asarray(point, dtype=np.float32).reshape(2)
    path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    if len(path) == 0:
        return 0, point.copy(), 0.0
    if len(path) == 1:
        return 0, path[0].copy(), float(np.linalg.norm(point - path[0]))
    best_idx = 0
    best_projection = path[0].copy()
    best_distance = float("inf")
    for idx, (start, end) in enumerate(zip(path[:-1], path[1:])):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-9:
            projection = start
        else:
            t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
            projection = start + t * segment
        distance = float(np.linalg.norm(point - projection))
        if distance < best_distance:
            best_idx = idx
            best_projection = projection.astype(np.float32, copy=False)
            best_distance = distance
    return best_idx, best_projection, best_distance


def _nearest_free_cell(occupied: np.ndarray, cell: tuple[int, int]) -> tuple[int, int] | None:
    x = int(np.clip(cell[0], 0, occupied.shape[0] - 1))
    y = int(np.clip(cell[1], 0, occupied.shape[1] - 1))
    if not occupied[x, y]:
        return (x, y)
    max_radius = max(occupied.shape)
    for radius in range(1, max_radius):
        for ix in range(max(x - radius, 0), min(x + radius + 1, occupied.shape[0])):
            for iy in range(max(y - radius, 0), min(y + radius + 1, occupied.shape[1])):
                if max(abs(ix - x), abs(iy - y)) == radius and not occupied[ix, iy]:
                    return (ix, iy)
    return None


def _astar_grid(occupied: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    moves = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]
    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (_grid_heuristic(start, goal), 0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_cost = {start: 0.0}
    while open_heap:
        _priority, cost, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct_grid_path(came_from, current)
        if cost > best_cost.get(current, float("inf")):
            continue
        for dx, dy, step_cost in moves:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor[0] < 0 or neighbor[0] >= occupied.shape[0] or neighbor[1] < 0 or neighbor[1] >= occupied.shape[1]:
                continue
            if occupied[neighbor]:
                continue
            new_cost = cost + step_cost
            if new_cost >= best_cost.get(neighbor, float("inf")):
                continue
            best_cost[neighbor] = new_cost
            came_from[neighbor] = current
            heapq.heappush(open_heap, (new_cost + _grid_heuristic(neighbor, goal), new_cost, neighbor))
    return []


def _grid_heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _reconstruct_grid_path(came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _simplify_path(path: np.ndarray, *, stride: int) -> np.ndarray:
    if len(path) <= 2 or stride <= 1:
        return path.astype(np.float32, copy=False)
    keep = list(range(0, len(path), stride))
    if keep[-1] != len(path) - 1:
        keep.append(len(path) - 1)
    return path[keep].astype(np.float32, copy=False)


def _smooth_path_with_clearance(
    path: np.ndarray,
    obstacles: np.ndarray,
    *,
    clearance_margin: float,
    iterations: int,
    alpha: float,
    sample_resolution: float,
) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    if len(path) <= 2 or iterations <= 0 or alpha <= 0.0:
        return path.astype(np.float32, copy=False)
    obstacles = np.asarray(obstacles, dtype=np.float32).reshape(-1, 7)
    if obstacles.size == 0:
        return _laplacian_smooth_path(path, iterations=iterations, alpha=alpha)

    original = path.astype(np.float32, copy=True)
    dense = _resample_path(original, spacing=sample_resolution)
    for scale in (1.0, 0.5, 0.25):
        candidate = _laplacian_smooth_path(dense, iterations=iterations, alpha=alpha * scale)
        if _path_has_clearance(
            candidate,
            obstacles,
            clearance_margin=clearance_margin,
            sample_resolution=sample_resolution,
        ):
            return candidate
    return original


def _resample_path(path: np.ndarray, *, spacing: float) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    if len(path) <= 2:
        return path.astype(np.float32, copy=False)
    spacing = max(float(spacing), 1e-3)
    points = [path[0].copy()]
    for start, end in zip(path[:-1], path[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-6:
            continue
        count = max(int(math.floor(length / spacing)), 1)
        for idx in range(1, count + 1):
            ratio = min(float(idx) * spacing / length, 1.0)
            point = start + segment * ratio
            if float(np.linalg.norm(point - points[-1])) > 1e-6:
                points.append(point.astype(np.float32, copy=False))
        if float(np.linalg.norm(end - points[-1])) > 1e-6:
            points.append(end.copy())
    points[-1] = path[-1].copy()
    return np.asarray(points, dtype=np.float32)


def _limit_path_points(path: np.ndarray, *, max_points: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    max_points = max(int(max_points), 2)
    if len(path) <= max_points:
        return path.astype(np.float32, copy=False)
    indices = np.rint(np.linspace(0, len(path) - 1, max_points)).astype(int)
    indices[0] = 0
    indices[-1] = len(path) - 1
    return path[np.unique(indices)].astype(np.float32, copy=False)


def _laplacian_smooth_path(path: np.ndarray, *, iterations: int, alpha: float) -> np.ndarray:
    smoothed = np.asarray(path, dtype=np.float32).reshape(-1, 2).copy()
    for _ in range(iterations):
        previous = smoothed.copy()
        smoothed[1:-1] = previous[1:-1] + float(alpha) * (
            0.5 * (previous[:-2] + previous[2:]) - previous[1:-1]
        )
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed.astype(np.float32, copy=False)


def _path_has_clearance(
    path: np.ndarray,
    obstacles: np.ndarray,
    *,
    clearance_margin: float,
    sample_resolution: float,
) -> bool:
    if obstacles.size == 0:
        return True
    path = np.asarray(path, dtype=np.float32).reshape(-1, 2)
    obstacles = np.asarray(obstacles, dtype=np.float32).reshape(-1, 7)
    for start, end in zip(path[:-1], path[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        samples = max(int(math.ceil(length / float(sample_resolution))), 1)
        for ratio in np.linspace(0.0, 1.0, samples + 1, dtype=np.float32):
            point = start + segment * float(ratio)
            distances = np.linalg.norm(obstacles[:, :2] - point, axis=1)
            required = obstacles[:, 2] + float(clearance_margin) - 0.05
            if bool(np.any(distances < required)):
                return False
    return True


class MujocoClosedLoopRecorder:
    def __init__(
        self,
        *,
        results_path: str | Path,
        config: dict[str, Any],
        controller_name: str,
        backend: str,
        seed: int,
    ) -> None:
        self.results_path = Path(results_path)
        self.config = config
        self.controller_name = str(controller_name)
        self.backend = str(backend)
        self.seed = int(seed)
        self.goal = optional_goal_from_config(config)
        self.minimum_distance = float(config["simulation"].get("minimum_distance", 0.5))
        self.dt = 1.0 / float(config["simulation"].get("sampling_rate", 10.0))
        self.rows: list[dict[str, Any]] = []
        self.global_path_rows: list[dict[str, Any]] = []
        self._next_path_plan_id = 0

    def record_step(
        self,
        *,
        state: np.ndarray,
        raw_control: np.ndarray,
        commanded_control: np.ndarray,
        measured_control: np.ndarray,
        terrain_features: np.ndarray,
        terrain_risk: float,
        mppi_time_ms: float,
        min_cost: float,
        planning_goal: np.ndarray | None = None,
        active_path_plan_id: int | None = None,
    ) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(6)
        raw = np.asarray(raw_control, dtype=np.float32).reshape(3)
        command = np.asarray(commanded_control, dtype=np.float32).reshape(3)
        measured = np.asarray(measured_control, dtype=np.float32).reshape(3)
        terrain = np.asarray(terrain_features, dtype=np.float32).reshape(4)
        local_goal = None if planning_goal is None else np.asarray(planning_goal, dtype=np.float32).reshape(6)
        residual = measured - command
        self.rows.append(
            {
                "state": state.copy(),
                "planning_goal": None if local_goal is None else local_goal.copy(),
                "active_path_plan_id": active_path_plan_id,
                "raw_control": raw.copy(),
                "commanded_control": command.copy(),
                "measured_control": measured.copy(),
                "exec_residual": residual.astype(np.float32, copy=False),
                "terrain_features": terrain.copy(),
                "terrain_risk": float(terrain_risk),
                "mppi_time_ms": float(mppi_time_ms),
                "min_cost": float(min_cost),
            }
        )

    def record_global_path(self, *, step: int, path: np.ndarray) -> int | None:
        waypoints = np.asarray(path, dtype=np.float32).reshape(-1, 2)
        if len(waypoints) == 0:
            return None
        plan_id = self._next_path_plan_id
        self._next_path_plan_id += 1
        for waypoint_index, waypoint in enumerate(waypoints):
            self.global_path_rows.append(
                {
                    "plan_id": plan_id,
                    "step": int(step),
                    "waypoint_index": int(waypoint_index),
                    "x": float(waypoint[0]),
                    "y": float(waypoint[1]),
                }
            )
        return plan_id

    def write(
        self,
        *,
        final_state: np.ndarray,
        reached_goal: bool,
        failed: bool,
        failure_reason: str | None,
    ) -> dict[str, Any]:
        self.results_path.mkdir(parents=True, exist_ok=True)
        final_state = np.asarray(final_state, dtype=np.float32).reshape(6)
        self._write_config()
        self._write_trajectory(final_state)
        self._write_controls()
        self._write_residuals()
        self._write_terrain()
        self._write_planning_goals()
        self._write_global_path()
        self._write_series("time_results.csv", "mppi_time_ms")
        self._write_series("costs.csv", "min_cost")
        self._write_trajectory_overlay(final_state)
        summary = self._summary(final_state, reached_goal, failed, failure_reason)
        (self.results_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (self.results_path / "test_summary.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(summary, stream, sort_keys=False)
        experiment_summary = {
            "profile_path": self.config.get("mujoco", {}).get("profile_path"),
            "experiment_name": self.config["results"].get("run_name", "mujoco_scout"),
            "controller_name": self.controller_name,
            "method": "learned_fdm" if bool(self.config.get("fdm", {}).get("enabled", False)) else "nominal",
            "seed": self.seed,
            "backend": self.backend,
            "results_path": str(self.results_path),
            "steps": summary["steps"],
            "reached_goal": bool(reached_goal),
            "failed": bool(failed),
            "run_time": summary["run_time"],
            "metrics": summary,
            "artifacts": collect_run_artifacts(self.results_path),
        }
        experiment_summary["artifacts"]["experiment_summary"] = str(self.results_path / "experiment_summary.json")
        (self.results_path / "experiment_summary.json").write_text(
            json.dumps(experiment_summary, indent=2),
            encoding="utf-8",
        )
        return summary

    def _write_config(self) -> None:
        with (self.results_path / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.config, stream, sort_keys=False)

    def _write_trajectory(self, final_state: np.ndarray) -> None:
        rows = []
        for idx, row in enumerate(self.rows):
            rows.append(_trajectory_row(idx, row["state"], self.goal))
        rows.append(_trajectory_row(len(self.rows), final_state, self.goal))
        pd.DataFrame(rows).to_csv(self.results_path / "trajectory.csv", index=False)
        pd.DataFrame(rows).rename(columns={"vx": "dx", "vy": "dy"}).to_csv(
            self.results_path / "results.csv",
            index=False,
        )

    def _write_planning_goals(self) -> None:
        rows = []
        for idx, row in enumerate(self.rows):
            goal = row.get("planning_goal")
            if goal is None:
                continue
            rows.append(
                {
                    "step": idx,
                    "plan_id": row.get("active_path_plan_id"),
                    "x": goal[0],
                    "y": goal[1],
                    "theta": goal[2],
                }
            )
        pd.DataFrame(rows).to_csv(self.results_path / "planning_goals.csv", index=False)

    def _write_global_path(self) -> None:
        pd.DataFrame(self.global_path_rows).to_csv(self.results_path / "global_path.csv", index=False)

    def _write_controls(self) -> None:
        rows = [
            {"step": idx, "vx_cmd": row["commanded_control"][0], "vy_cmd": row["commanded_control"][1], "wz_cmd": row["commanded_control"][2]}
            for idx, row in enumerate(self.rows)
        ]
        pd.DataFrame(rows).to_csv(self.results_path / "controls.csv", index=False)
        raw_rows = [
            {"step": idx, "vx_cmd": row["raw_control"][0], "vy_cmd": row["raw_control"][1], "wz_cmd": row["raw_control"][2]}
            for idx, row in enumerate(self.rows)
        ]
        pd.DataFrame(raw_rows).to_csv(self.results_path / "raw_controls.csv", index=False)

    def _write_residuals(self) -> None:
        rows = []
        for idx, row in enumerate(self.rows):
            cmd = row["commanded_control"]
            measured = row["measured_control"]
            delta = row["exec_residual"]
            norm = float(np.linalg.norm(delta))
            rows.append(
                {
                    "step": idx,
                    "cmd_vx": cmd[0],
                    "cmd_vy": cmd[1],
                    "cmd_wz": cmd[2],
                    "real_vx": measured[0],
                    "real_vy": measured[1],
                    "real_wz": measured[2],
                    "oracle_du_vx": delta[0],
                    "oracle_du_vy": delta[1],
                    "oracle_du_wz": delta[2],
                    "oracle_du_norm": norm,
                    "exec_du_vx": delta[0],
                    "exec_du_vy": delta[1],
                    "exec_du_wz": delta[2],
                    "exec_du_norm": norm,
                    "du_vx": delta[0],
                    "du_vy": delta[1],
                    "du_wz": delta[2],
                    "du_norm": norm,
                }
            )
        pd.DataFrame(rows).to_csv(self.results_path / "residuals.csv", index=False)

    def _write_terrain(self) -> None:
        threshold = float(self.config.get("mppi", {}).get("terrain_risk_threshold", 0.0))
        rows = []
        for idx, row in enumerate(self.rows):
            state = row["state"]
            features = row["terrain_features"]
            risk = float(row["terrain_risk"])
            excess = max(risk - threshold, 0.0)
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
                    "risk_excess": excess,
                    "risk_exposed": bool(risk > threshold),
                }
            )
        pd.DataFrame(rows).to_csv(self.results_path / "terrain.csv", index=False)

    def _write_series(self, filename: str, key: str) -> None:
        pd.DataFrame({key: [row[key] for row in self.rows]}).to_csv(self.results_path / filename, index=False)

    def _write_trajectory_overlay(self, final_state: np.ndarray) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except Exception:
            return

        states = np.asarray([row["state"] for row in self.rows] + [final_state], dtype=np.float32)
        if states.size == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
        if self.goal is not None:
            ax.plot([states[0, 0], self.goal[0]], [states[0, 1], self.goal[1]], "--", color="#999999", lw=1.2, label="start-goal line")
        if self.global_path_rows:
            paths = pd.DataFrame(self.global_path_rows)
            first_plan_id = int(paths["plan_id"].min())
            last_plan_id = int(paths["plan_id"].max())
            first_path = paths[paths["plan_id"] == first_plan_id].sort_values("waypoint_index")
            ax.plot(first_path["x"], first_path["y"], color="#2ca02c", lw=1.8, alpha=0.75, label="A* global path")
            if last_plan_id != first_plan_id:
                last_path = paths[paths["plan_id"] == last_plan_id].sort_values("waypoint_index")
                ax.plot(last_path["x"], last_path["y"], color="#17becf", lw=1.2, alpha=0.65, label="latest replan")
        planning_goals = [row["planning_goal"] for row in self.rows if row.get("planning_goal") is not None]
        if planning_goals:
            local = np.asarray(planning_goals, dtype=np.float32)
            ax.scatter(local[:, 0], local[:, 1], s=8, color="#ff7f0e", alpha=0.22, label="MPPI lookahead goals")
        ax.plot(states[:, 0], states[:, 1], color="#1f77b4", lw=2.2, label="executed trajectory")
        ax.scatter([states[0, 0]], [states[0, 1]], s=70, marker="o", color="#2ca02c", label="start", zorder=5)
        if self.goal is not None:
            ax.scatter([self.goal[0]], [self.goal[1]], s=90, marker="*", color="#ff7f0e", label="goal", zorder=5)
        ax.scatter([states[-1, 0]], [states[-1, 1]], s=70, marker="x", color="#111111", label="final", zorder=6)
        if self.goal is not None:
            ax.add_patch(Circle((float(self.goal[0]), float(self.goal[1])), self.minimum_distance, fill=False, ls=":", lw=1.4, color="#ff7f0e"))
        ax.set_title("MuJoCo cube omni: external path + MPPI")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        margin = 1.5
        xs = [float(np.min(states[:, 0])), float(np.max(states[:, 0]))]
        ys = [float(np.min(states[:, 1])), float(np.max(states[:, 1]))]
        if self.goal is not None:
            xs.append(float(self.goal[0]))
            ys.append(float(self.goal[1]))
        if self.global_path_rows:
            path_rows = pd.DataFrame(self.global_path_rows)
            xs.extend([float(path_rows["x"].min()), float(path_rows["x"].max())])
            ys.extend([float(path_rows["y"].min()), float(path_rows["y"].max())])
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(self.results_path / "trajectory_overlay.png")
        fig.savefig(self.results_path / "trajectory.png")
        plt.close(fig)

    def _summary(self, final_state: np.ndarray, reached_goal: bool, failed: bool, failure_reason: str | None) -> dict[str, Any]:
        states = [row["state"] for row in self.rows] + [final_state]
        path_length = _path_length(states)
        controls = np.asarray([row["commanded_control"] for row in self.rows], dtype=np.float32)
        residuals = np.asarray([row["exec_residual"] for row in self.rows], dtype=np.float32)
        risks = np.asarray([row["terrain_risk"] for row in self.rows], dtype=np.float32)
        return {
            "world_mode": "mujoco",
            "controller_type": self.controller_name,
            "success": bool(reached_goal),
            "reached_goal": bool(reached_goal),
            "failed": bool(failed),
            "failure_reason": failure_reason,
            "goal_termination_disabled": bool(self.config["simulation"].get("disable_goal_termination", False)),
            "init_pose": states[0].tolist() if states else final_state.tolist(),
            "goal": None if self.goal is None else self.goal.tolist(),
            "steps": len(self.rows),
            "final_distance": None if self.goal is None else float(np.linalg.norm(self.goal[:2] - final_state[:2])),
            "path_length": path_length,
            "arrival_time": len(self.rows) * self.dt if reached_goal else None,
            "run_time": len(self.rows) * self.dt,
            "mean_mppi_time_ms": _mean([row["mppi_time_ms"] for row in self.rows]),
            "max_mppi_time_ms": _max([row["mppi_time_ms"] for row in self.rows]),
            "controls_csv": "executed_controls",
            "mean_residual_norm": _mean_norm(residuals),
            "max_residual_norm": _max_norm(residuals),
            "mean_cmd_real_error": _mean_norm(residuals),
            "max_cmd_real_error": _max_norm(residuals),
            "mean_terrain_risk": float(np.mean(risks)) if risks.size else 0.0,
            "max_terrain_risk": float(np.max(risks)) if risks.size else 0.0,
            "cumulative_terrain_risk": float(np.sum(risks)) if risks.size else 0.0,
            "control_smoothness": _smoothness(controls),
            "control_jerk": _jerk(controls),
            "lateral_usage": float(np.mean(controls[:, 1] * controls[:, 1])) if controls.size else 0.0,
            **self._path_tracking_metrics(),
        }

    def _path_tracking_metrics(self) -> dict[str, float | None]:
        if not self.rows or not self.global_path_rows:
            return {
                "path_tracking_mean_m": None,
                "path_tracking_p90_m": None,
                "path_tracking_max_m": None,
            }
        paths: dict[int, list[tuple[int, np.ndarray]]] = {}
        for row in self.global_path_rows:
            paths.setdefault(int(row["plan_id"]), []).append(
                (int(row["waypoint_index"]), np.asarray([row["x"], row["y"]], dtype=np.float32))
            )
        ordered_paths = {
            plan_id: np.asarray([point for _idx, point in sorted(points, key=lambda item: item[0])], dtype=np.float32)
            for plan_id, points in paths.items()
        }
        distances = []
        for row in self.rows:
            plan_id = row.get("active_path_plan_id")
            if plan_id is None or int(plan_id) not in ordered_paths:
                continue
            _idx, _projected, distance = project_point_to_path(row["state"][:2], ordered_paths[int(plan_id)])
            distances.append(distance)
        if not distances:
            return {
                "path_tracking_mean_m": None,
                "path_tracking_p90_m": None,
                "path_tracking_max_m": None,
            }
        values = np.asarray(distances, dtype=np.float32)
        return {
            "path_tracking_mean_m": float(np.mean(values)),
            "path_tracking_p90_m": float(np.percentile(values, 90)),
            "path_tracking_max_m": float(np.max(values)),
        }


def run_mujoco_closed_loop_profile(
    profile_path: str | Path,
    *,
    controller_name: str | None = None,
    seed: int | None = None,
    backend: str | None = None,
    results_dir: str | Path | None = None,
    max_steps: int | None = None,
    odom_timeout: float | None = None,
    spin_once_timeout: float = 0.1,
) -> dict[str, Any]:
    config, metadata = build_experiment_config(
        profile_path,
        controller_name=controller_name,
        seed=seed,
        backend=backend,
        results_dir=results_dir,
    )
    config.setdefault("mujoco", {})["profile_path"] = str(profile_path)
    if max_steps is not None:
        config["simulation"]["max_steps"] = int(max_steps)
    if odom_timeout is not None:
        config.setdefault("mujoco", {})["odom_timeout"] = float(odom_timeout)
    validate_learned_fdm_artifacts(config, metadata)
    return _run_ros_node(config, metadata, spin_once_timeout=spin_once_timeout)


def _run_ros_node(config: dict[str, Any], metadata: dict[str, Any], *, spin_once_timeout: float) -> dict[str, Any]:
    import rclpy

    rclpy.init(args=None)
    node = MujocoClosedLoopNode(config, metadata)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node._node, timeout_sec=spin_once_timeout)
    finally:
        result = node.result
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return result


class MujocoClosedLoopNode:
    def __init__(self, config: dict[str, Any], metadata: dict[str, Any]) -> None:
        import rclpy
        from geometry_msgs.msg import Point, PoseStamped, Twist
        from nav_msgs.msg import Odometry, Path as RosPath
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        try:
            from visualization_msgs.msg import Marker, MarkerArray
        except Exception:
            Marker = None
            MarkerArray = None

        class _Node(Node):
            pass

        self._node = _Node("fdm_mppi_mujoco_closed_loop")
        self._rclpy = rclpy
        self._twist_type = Twist
        self._point_type = Point
        self._pose_stamped_type = PoseStamped
        self._path_type = RosPath
        self._marker_type = Marker
        self._marker_array_type = MarkerArray
        self.config = config
        self.metadata = metadata
        self.goal = optional_goal_from_config(config)
        self.goal_relief_center_follows_goal = runtime_goal_relief_center_follows_goal(config)
        if self.goal is not None:
            sync_runtime_goal_relief_center(
                config,
                self.goal,
                follow_goal=self.goal_relief_center_follows_goal,
            )
        self.minimum_distance = float(config["simulation"]["minimum_distance"])
        self.goal_termination_distance = float(
            config["simulation"].get("goal_termination_distance", max(self.minimum_distance - 0.1, 0.0))
        )
        self.max_steps = int(config["simulation"]["max_steps"])
        mujoco_cfg = config.get("mujoco", {})
        self.drive_mode = str(mujoco_cfg.get("drive_mode", "differential")).lower()
        self.odom_timeout = float(mujoco_cfg.get("odom_timeout", 1.0))
        self.startup_odom_timeout = float(mujoco_cfg.get("startup_odom_timeout", 0.0))
        self.odom_topic = str(mujoco_cfg.get("odom_topic", "/scout1/odom"))
        self.cmd_vel_topic = str(mujoco_cfg.get("cmd_vel_topic", "/scout1/cmd_vel"))
        self.terrain = TerrainField.from_config(config.get("terrain"))
        self.controller = create_omni_controller(config, seed=int(metadata["seed"]))
        self.static_obstacles = np.asarray(config.get("obstacles", {}).get("virtual", []), dtype=np.float32).reshape(-1, 7)
        self.elevation_map = ElevationMapObstacleAdapter.from_config(config)
        self.local_costmap = LocalCostmapAdapter(LocalCostmapConfig.from_config(config))
        self.external_path = ExternalPathAdapter(ExternalPathConfig.from_config(config))
        self.local_goal = LocalGoalConfig.from_config(config)
        self.global_path = GlobalPathConfig.from_config(config)
        self.command_filter = CommandFilterConfig.from_config(config)
        goal_topic_cfg = config.get("goal_topic", {})
        self.goal_topic_enabled = bool(goal_topic_cfg.get("enabled", False))
        self.goal_topic = str(goal_topic_cfg.get("topic", "/move_base_simple/goal"))
        self.command_dt = 1.0 / float(config["simulation"]["sampling_rate"])
        self.previous_command = np.zeros(3, dtype=np.float32)
        self.path_waypoints = np.empty((0, 2), dtype=np.float32)
        self.path_plan_id: int | None = None
        self.path_replan_step = -1
        self.path_replan_state = np.empty(2, dtype=np.float32)
        self.results_path = create_results_path(config["results"])
        self.recorder = MujocoClosedLoopRecorder(
            results_path=self.results_path,
            config=config,
            controller_name=str(metadata["controller_name"]),
            backend=str(config["mppi"].get("backend", "numpy")).lower(),
            seed=int(metadata["seed"]),
        )
        self.last_state: np.ndarray | None = None
        self.start_time = time.monotonic()
        self.last_odom_time = 0.0
        self.waiting_for_odom_logged = False
        self.waiting_for_goal_logged = False
        self.steps = 0
        self.failed = False
        self.failure_reason: str | None = None
        self.finished = False
        self.result: dict[str, Any] = {}
        self.publisher = self._node.create_publisher(Twist, self.cmd_vel_topic, 10)
        rviz_cfg = config.get("rviz", {})
        self.rviz_enabled = bool(rviz_cfg.get("enabled", False)) and Marker is not None and MarkerArray is not None
        self.rviz_frame_id = str(rviz_cfg.get("frame_id", "map"))
        self.rviz_sample_count = max(int(rviz_cfg.get("sample_count", 50)), 0)
        self.rviz_sample_topic = str(rviz_cfg.get("sample_topic", "/fdm_mppi/sample_rollouts"))
        self.rviz_sample_path_prefix = str(rviz_cfg.get("sample_path_topic_prefix", "/fdm_mppi/sample_rollout_"))
        self.rviz_selected_path_topic = str(rviz_cfg.get("selected_path_topic", "/fdm_mppi/selected_rollout"))
        self.rviz_history_path_topic = str(rviz_cfg.get("history_path_topic", "/fdm_mppi/history_path"))
        self.rviz_history_max_points = max(int(rviz_cfg.get("history_max_points", 3000)), 0)
        self.rviz_robot_topic = str(rviz_cfg.get("robot_topic", "/fdm_mppi/cube_robot"))
        self.rviz_robot_scale = tuple(float(v) for v in rviz_cfg.get("robot_scale", [0.7, 0.7, 0.35]))
        self.rviz_history_path = np.empty((0, 6), dtype=np.float32)
        self.rollout_marker_pub = None
        self.sample_path_pubs = []
        self.selected_path_pub = None
        self.history_path_pub = None
        self.robot_marker_pub = None
        if self.rviz_enabled:
            self.rollout_marker_pub = self._node.create_publisher(MarkerArray, self.rviz_sample_topic, 10)
            self.sample_path_pubs = [
                self._node.create_publisher(RosPath, f"{self.rviz_sample_path_prefix}{idx:02d}", 10)
                for idx in range(self.rviz_sample_count)
            ]
            self.selected_path_pub = self._node.create_publisher(RosPath, self.rviz_selected_path_topic, 10)
            self.history_path_pub = self._node.create_publisher(RosPath, self.rviz_history_path_topic, 10)
            self.robot_marker_pub = self._node.create_publisher(Marker, self.rviz_robot_topic, 10)
        elif bool(rviz_cfg.get("enabled", False)):
            self._node.get_logger().warning("rviz.enabled=true but visualization_msgs is unavailable; disabling markers")
        self.subscription = self._node.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)
        self.elevation_subscription = None
        if self.elevation_map.enabled:
            from grid_map_msgs.msg import GridMap

            topic = str(config.get("elevation_map", {}).get("topic", "/elevation_mapping_node/elevation_map"))
            self.elevation_subscription = self._node.create_subscription(GridMap, topic, self._on_grid_map, 10)
        self.local_costmap_subscription = None
        if self.local_costmap.cfg.enabled:
            try:
                from elevation_msgs.msg import OccupancyElevation
            except Exception as exc:
                raise RuntimeError(
                    "local_costmap.enabled=true requires elevation_msgs/OccupancyElevation; "
                    "source Geomapping_ros2 install/setup.bash before running MPPI"
                ) from exc
            self.local_costmap_subscription = self._node.create_subscription(
                OccupancyElevation,
                self.local_costmap.cfg.topic,
                self._on_local_costmap,
                10,
            )
        self.external_path_subscription = None
        if self.external_path.cfg.enabled:
            external_path_msg_type = self._external_path_message_type(RosPath)
            external_path_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.external_path_subscription = self._node.create_subscription(
                external_path_msg_type,
                self.external_path.cfg.path_topic,
                self._on_external_path,
                external_path_qos,
            )
        self.goal_subscription = None
        if self.goal_topic_enabled:
            self.goal_subscription = self._node.create_subscription(
                PoseStamped,
                self.goal_topic,
                self._on_goal,
                10,
            )
        self.timer = self._node.create_timer(1.0 / float(config["simulation"]["sampling_rate"]), self._on_timer)
        self._node.get_logger().info(f"MuJoCo closed-loop controller started: odom={self.odom_topic} cmd={self.cmd_vel_topic}")

    def __getattr__(self, name: str):
        return getattr(self._node, name)

    def _external_path_message_type(self, ros_path_type: Any) -> Any:
        message_type = self.external_path.cfg.message_type.replace(" ", "").lower()
        if message_type in {"nav_msgs/path", "nav_msgs/msg/path", "path"}:
            return ros_path_type
        if message_type in {
            "traversability_mapping/polynome",
            "traversability_mapping/msg/polynome",
            "polynome",
        }:
            try:
                from traversability_mapping.msg import Polynome
            except Exception as exc:
                raise RuntimeError(
                    "external_path.message_type=traversability_mapping/Polynome requires "
                    "traversability_mapping messages in the sourced ROS2 overlay"
                ) from exc
            return Polynome
        raise ValueError(
            "Unsupported external_path.message_type="
            f"{self.external_path.cfg.message_type!r}; expected nav_msgs/Path or traversability_mapping/Polynome"
        )

    def _on_odom(self, message: Any) -> None:
        self.last_state = project_mujoco_state(odom_message_to_state(message), self.drive_mode)
        self.last_odom_time = time.monotonic()

    def _on_grid_map(self, message: Any) -> None:
        self.elevation_map.update(message)

    def _on_local_costmap(self, message: Any) -> None:
        self.local_costmap.update(message)

    def _on_external_path(self, message: Any) -> None:
        self.external_path.update(message)

    def _on_goal(self, message: Any) -> None:
        self.goal = pose_stamped_message_to_goal(message)
        sync_runtime_goal_relief_center(
            self.config,
            self.goal,
            self.terrain,
            self.controller,
            follow_goal=self.goal_relief_center_follows_goal,
        )
        self.recorder.goal = self.goal.copy()
        self.path_waypoints = np.empty((0, 2), dtype=np.float32)
        self.path_plan_id = None
        self.external_path.clear()
        self._node.get_logger().info(
            f"Updated MPPI goal from {self.goal_topic}: "
            f"x={self.goal[0]:.2f} y={self.goal[1]:.2f} yaw={self.goal[2]:.2f}"
        )

    def _on_timer(self) -> None:
        if self.finished:
            return
        if self.last_state is None:
            now = time.monotonic()
            if self.last_odom_time == 0.0 and not initial_odom_timeout_expired(
                self.start_time,
                now,
                self.startup_odom_timeout,
            ):
                if not self.waiting_for_odom_logged:
                    self._node.get_logger().info(f"Waiting for first odom on {self.odom_topic}")
                    self.waiting_for_odom_logged = True
                return
            self._finish(True, "no odom received")
            return
        if time.monotonic() - self.last_odom_time > self.odom_timeout:
            self._finish(True, "odom timeout")
            return
        if runtime_goal_required(self.config) and self.goal is None:
            self.previous_command[:] = 0.0
            self.publisher.publish(self._twist_type())
            if not self.waiting_for_goal_logged:
                self._node.get_logger().info(f"Waiting for RViz goal on {self.goal_topic}")
                self.waiting_for_goal_logged = True
            return
        if self.steps >= self.max_steps:
            self._finish(False, None)
            return
        if self.goal is not None and not bool(self.config["simulation"].get("disable_goal_termination", False)):
            goal_dist = float(np.linalg.norm(self.goal[:2] - self.last_state[:2]))
            if goal_reached_xy(self.last_state, self.goal, self.goal_termination_distance + 0.1):
                self._node.get_logger().info(
                    f"Goal reached at distance {goal_dist:.3f}m (threshold={self.goal_termination_distance:.3f}m)"
                )
                if runtime_goal_required(self.config):
                    self._on_goal_reached()
                else:
                    self._finish(False, None)
                return
            elif goal_dist < self.goal_termination_distance + 0.5:
                self._node.get_logger().debug(f"Approaching goal: distance={goal_dist:.3f}m")

        state = self.last_state.copy()
        if self.elevation_map.enabled and self.elevation_map.required and not self.elevation_map.has_fresh_map():
            self.previous_command[:] = 0.0
            self.publisher.publish(self._twist_type())
            return
        local_costmap_snapshot = self.local_costmap.snapshot()
        using_local_costmap = bool(local_costmap_snapshot.get("enabled", False))
        if self.local_costmap.cfg.enabled and self.local_costmap.cfg.required and not using_local_costmap:
            self.previous_command[:] = 0.0
            self.publisher.publish(self._twist_type())
            return
        if (
            self.external_path.cfg.enabled
            and self.external_path.cfg.stop_on_stale
            and not self.external_path.has_fresh_path()
        ):
            self.previous_command[:] = 0.0
            self.publisher.publish(self._twist_type())
            return
        map_obstacles = self.elevation_map.obstacles_for_state(state)
        obstacles = merge_static_and_map_obstacles(
            self.static_obstacles,
            map_obstacles,
            dedupe_distance=self.elevation_map.static_dedupe_distance,
        )
        tracking_obstacles = (
            np.empty((0, 7), dtype=np.float32)
            if using_local_costmap or self.external_path.has_fresh_path()
            else obstacles
        )
        start = time.time()
        planning_goal = self.goal if self.goal is not None else path_terminal_goal(self.external_path.path(), state)
        active_path = np.empty((0, 2), dtype=np.float32)
        active_path_plan_id = None
        final_cfg = self.config.get("final_controller", {})
        disable_final_with_local_costmap = bool(final_cfg.get("disable_when_local_costmap", True))
        use_final_controller = not (using_local_costmap and disable_final_with_local_costmap)
        final_u = None if self.goal is None or not use_final_controller else final_approach_control(state, self.goal, self.config)
        optimal_u = None
        sample_u = None
        if final_u is None:
            if self.local_costmap.cfg.enabled and not self.external_path.has_fresh_path():
                planning_goal = self.goal if self.goal is not None else np.asarray(state, dtype=np.float32).reshape(6).copy()
            else:
                planning_goal, active_path, active_path_plan_id = self._planning_goal(state, obstacles)
            raw_u, optimal_u, sample_u, _normalizer, min_cost = self.controller.compute_control(
                state,
                [
                    None,
                    None,
                    None,
                    planning_goal,
                    tracking_obstacles,
                    len(tracking_obstacles),
                    active_path,
                    local_costmap_snapshot,
                    self.goal,
                ],
            )
            if self.goal is not None and self.drive_mode not in {"omni", "omni_freejoint", "holonomic"}:
                goal_vec = self.goal[:2] - state[:2]
                goal_dist = float(np.linalg.norm(goal_vec))
                if goal_dist > 1.5:
                    target_yaw = math.atan2(float(goal_vec[1]), float(goal_vec[0]))
                    heading_err = angle_diff(target_yaw, float(state[2]))
                    if abs(heading_err) > math.radians(45.0):
                        max_wz = float(self.config["robot"]["max_wz"])
                        raw_u = np.array([0.0, 0.0, np.sign(heading_err) * max_wz], dtype=np.float32)
                        optimal_u = raw_u.copy()
                        sample_u = None
        else:
            raw_u = final_u
            min_cost = 0.0
        elapsed_ms = (time.time() - start) * 1000.0
        filtered_u = filter_mujoco_command(
            raw_u,
            self.previous_command,
            self.command_filter,
            self.command_dt,
            distance_to_goal=None if self.goal is None else float(np.linalg.norm(self.goal[:2] - state[:2])),
        )
        self.previous_command = filtered_u.copy()
        twist, command = mujoco_twist_command(filtered_u, drive_mode=self.drive_mode, twist_factory=self._twist_type)
        self.publisher.publish(twist)
        self._publish_rviz_markers(state, sample_u, optimal_u)
        terrain_features = self.terrain.feature(float(state[0]), float(state[1]))
        terrain_risk = self.terrain.risk_cost(float(state[0]), float(state[1]), features=terrain_features)
        self.recorder.record_step(
            state=state,
            raw_control=raw_u,
            commanded_control=command,
            measured_control=state[3:6],
            terrain_features=terrain_features,
            terrain_risk=terrain_risk,
            mppi_time_ms=elapsed_ms,
            min_cost=min_cost,
            planning_goal=planning_goal,
            active_path_plan_id=active_path_plan_id,
        )
        self.steps += 1
        if not np.isfinite(state).all() or not np.isfinite(command).all():
            self._finish(True, "non-finite state or command")

    def _publish_rviz_markers(
        self,
        state: np.ndarray,
        sample_u: np.ndarray | None,
        optimal_u: np.ndarray | None,
    ) -> None:
        if not self.rviz_enabled:
            return
        assert self._marker_type is not None
        assert self._marker_array_type is not None
        stamp = self._node.get_clock().now().to_msg()
        self.rviz_history_path = append_history_state(
            self.rviz_history_path,
            state,
            max_points=self.rviz_history_max_points,
        )
        if self.robot_marker_pub is not None:
            self.robot_marker_pub.publish(self._cube_marker(state))
        if self.history_path_pub is not None:
            self.history_path_pub.publish(self._path_message(self.rviz_history_path, stamp=stamp, z=0.16))
        if self.rollout_marker_pub is None:
            return
        marker_array = self._marker_array_type()
        delete_all = self._marker_type()
        delete_all.action = self._marker_type.DELETEALL
        marker_array.markers.append(delete_all)
        marker_id = 0
        max_control = np.asarray(self.config.get("robot", {}).get("max_control", []), dtype=np.float32)
        if max_control.size != 3:
            robot = self.config.get("robot", {})
            max_control = np.asarray(
                [robot.get("max_vx", 1.0), robot.get("max_vy", 1.0), robot.get("max_wz", 1.0)],
                dtype=np.float32,
            )
        if sample_u is not None:
            sample_arr = np.asarray(sample_u, dtype=np.float32)
            samples = sample_arr.reshape(-1, sample_arr.shape[-2], 3)
            for sample_index, controls in enumerate(samples[: self.rviz_sample_count]):
                rollout = predict_omni_rollout(state, controls, self.command_dt, max_control)
                if sample_index < len(self.sample_path_pubs):
                    self.sample_path_pubs[sample_index].publish(self._path_message(rollout, stamp=stamp))
                marker_array.markers.append(
                    self._line_strip_marker(
                        marker_id=marker_id,
                        namespace="sample_rollouts",
                        states=rollout,
                        rgba=(0.5, 0.5, 0.5, 0.35),
                        width=0.02,
                    )
                )
                marker_id += 1
        if optimal_u is not None:
            rollout = predict_omni_rollout(state, optimal_u, self.command_dt, max_control)
            if self.selected_path_pub is not None:
                self.selected_path_pub.publish(self._path_message(rollout, stamp=stamp))
            marker_array.markers.append(
                self._line_strip_marker(
                    marker_id=marker_id,
                    namespace="optimal_rollout",
                    states=rollout,
                    rgba=(1.0, 0.0, 0.0, 0.95),
                    width=0.06,
                )
            )
        self.rollout_marker_pub.publish(marker_array)

    def _path_message(self, states: np.ndarray, *, stamp: Any, z: float = 0.10):
        path = self._path_type()
        path.header.frame_id = self.rviz_frame_id
        path.header.stamp = stamp
        for state in np.asarray(states, dtype=np.float32).reshape(-1, 6):
            pose = self._pose_stamped_type()
            pose.header.frame_id = self.rviz_frame_id
            pose.header.stamp = stamp
            pose.pose.position.x = float(state[0])
            pose.pose.position.y = float(state[1])
            pose.pose.position.z = float(z)
            qx, qy, qz, qw = yaw_to_quaternion(float(state[2]))
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)
        return path

    def _line_strip_marker(
        self,
        *,
        marker_id: int,
        namespace: str,
        states: np.ndarray,
        rgba: tuple[float, float, float, float],
        width: float,
    ):
        marker = self._marker_type()
        marker.header.frame_id = self.rviz_frame_id
        marker.header.stamp = self._node.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.type = self._marker_type.LINE_STRIP
        marker.action = self._marker_type.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(width)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (float(v) for v in rgba)
        for state in np.asarray(states, dtype=np.float32).reshape(-1, 6):
            point = self._point_type()
            point.x = float(state[0])
            point.y = float(state[1])
            point.z = 0.08
            marker.points.append(point)
        return marker

    def _cube_marker(self, state: np.ndarray):
        state = np.asarray(state, dtype=np.float32).reshape(6)
        marker = self._marker_type()
        marker.header.frame_id = self.rviz_frame_id
        marker.header.stamp = self._node.get_clock().now().to_msg()
        marker.ns = "cube_omni"
        marker.id = 0
        marker.type = self._marker_type.CUBE
        marker.action = self._marker_type.ADD
        marker.pose.position.x = float(state[0])
        marker.pose.position.y = float(state[1])
        marker.pose.position.z = self.rviz_robot_scale[2] * 0.5
        qx, qy, qz, qw = yaw_to_quaternion(float(state[2]))
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x, marker.scale.y, marker.scale.z = self.rviz_robot_scale
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.25
        marker.color.a = 0.75
        return marker

    def _finish(self, failed: bool, reason: str | None) -> None:
        if self.finished:
            return
        self.finished = True
        self.failed = bool(failed)
        self.failure_reason = reason
        self.previous_command[:] = 0.0
        self.publisher.publish(self._twist_type())
        final_state = self.last_state if self.last_state is not None else np.asarray(self.config["simulation"]["initial_state"], dtype=np.float32)
        reached = False if self.goal is None else goal_reached_xy(
            final_state,
            self.goal,
            self.goal_termination_distance + 0.1,
        )
        summary = self.recorder.write(
            final_state=final_state,
            reached_goal=reached,
            failed=self.failed,
            failure_reason=self.failure_reason,
        )
        self.result = {
            **self.metadata,
            "backend": str(self.config["mppi"].get("backend", "numpy")).lower(),
            "results_path": str(self.results_path),
            "steps": summary["steps"],
            "reached_goal": reached,
            "failed": self.failed,
            "run_time": summary["run_time"],
            "artifacts": collect_run_artifacts(self.results_path),
            "metrics": summary,
        }

    def _on_goal_reached(self) -> None:
        self._node.get_logger().info("Goal reached. Waiting for next goal.")
        self.previous_command[:] = 0.0
        self.publisher.publish(self._twist_type())
        final_state = self.last_state if self.last_state is not None else np.asarray(self.config["simulation"]["initial_state"], dtype=np.float32)
        reached = goal_reached_xy(final_state, self.goal, self.minimum_distance)
        self.recorder.write(
            final_state=final_state,
            reached_goal=reached,
            failed=False,
            failure_reason=None,
        )
        self.goal = None
        self.waiting_for_goal_logged = False
        self.steps = 0
        self.recorder.rows = []
        self.recorder.global_path_rows = []
        self.recorder.goal = None
        self.recorder._next_path_plan_id = 0
        self.path_waypoints = np.empty((0, 2), dtype=np.float32)
        self.path_plan_id = None
        self.external_path.clear()
        self.rviz_history_path = np.empty((0, 6), dtype=np.float32)
        if hasattr(self.controller, "nominal_u"):
            self.controller.nominal_u = np.zeros_like(self.controller.nominal_u)
        if hasattr(self.controller, "previous_control"):
            self.controller.previous_control = np.zeros(3, dtype=np.float32)

    def _planning_goal(self, state: np.ndarray, obstacles: np.ndarray) -> tuple[np.ndarray, np.ndarray, int | None]:
        external_waypoints = self.external_path.path()
        if len(external_waypoints):
            reference_goal = self.goal if self.goal is not None else path_terminal_goal(external_waypoints, state)
            planning_goal, active_path = select_external_path_goal(
                state,
                reference_goal,
                external_waypoints,
                self.external_path.cfg,
            )
            return planning_goal, active_path, None
        if self.goal is None:
            return (
                np.asarray(state, dtype=np.float32).reshape(6).copy(),
                np.empty((0, 2), dtype=np.float32),
                None,
            )
        if self.global_path.enabled:
            if self._should_replan_path(state):
                self.path_waypoints = plan_global_path_astar(
                    np.asarray(state, dtype=np.float32)[:2],
                    self.goal[:2],
                    obstacles,
                    self.global_path,
                    robot_radius=float(self.config.get("robot", {}).get("radius", 0.35)),
                    safety_dist=float(self.config.get("robot", {}).get("safety_dist", 0.10)),
                )
                self.path_replan_step = self.steps
                self.path_replan_state = np.asarray(state, dtype=np.float32)[:2].copy()
                recorded_id = self.recorder.record_global_path(step=self.steps, path=self.path_waypoints)
                if recorded_id is not None:
                    self.path_plan_id = recorded_id
            if len(self.path_waypoints):
                return (
                    select_path_lookahead_goal(
                        state,
                        self.goal,
                        self.path_waypoints,
                        self.global_path.lookahead,
                        yaw_mode=self.global_path.yaw_mode,
                    ),
                    _limit_path_points(self.path_waypoints, max_points=self.global_path.cost_max_points),
                    self.path_plan_id,
                )
        return (
            select_obstacle_aware_goal(state, self.goal, obstacles, self.local_goal),
            np.empty((0, 2), dtype=np.float32),
            None,
        )

    def _should_replan_path(self, state: np.ndarray) -> bool:
        if len(self.path_waypoints) == 0:
            return True
        if self.steps - self.path_replan_step >= self.global_path.replan_steps:
            return True
        state_xy = np.asarray(state, dtype=np.float32)[:2]
        if self.path_replan_state.size != 2:
            return True
        return bool(np.linalg.norm(state_xy - self.path_replan_state) >= self.global_path.replan_distance)

    def destroy_node(self) -> bool:
        return self._node.destroy_node()


def _simple_twist():
    return SimpleNamespace(
        linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )


def _trajectory_row(step: int, state: np.ndarray, goal: np.ndarray | None) -> dict[str, Any]:
    return {
        "step": step,
        "x": state[0],
        "y": state[1],
        "theta": state[2],
        "vx": state[3],
        "vy": state[4],
        "wz": state[5],
        "x_des": np.nan if goal is None else goal[0],
        "y_des": np.nan if goal is None else goal[1],
        "theta_des": np.nan if goal is None else goal[2],
    }


def _path_length(states: list[np.ndarray]) -> float:
    if len(states) < 2:
        return 0.0
    deltas = np.diff(np.asarray([state[:2] for state in states], dtype=np.float32), axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _mean_norm(values: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(values, axis=1))) if values.size else 0.0


def _max_norm(values: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(values, axis=1))) if values.size else 0.0


def _smoothness(controls: np.ndarray) -> float:
    if len(controls) < 2:
        return 0.0
    deltas = np.diff(controls, axis=0)
    return float(np.mean(np.sum(deltas * deltas, axis=1)))


def _jerk(controls: np.ndarray) -> float:
    if len(controls) < 3:
        return 0.0
    jerks = controls[2:] - 2.0 * controls[1:-1] + controls[:-2]
    return float(np.mean(np.sum(jerks * jerks, axis=1)))
