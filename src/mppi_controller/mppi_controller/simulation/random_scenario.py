"""Random start/goal sampling for oracle simulation scenarios."""

from __future__ import annotations

import numpy as np


def sample_start_goal(config: dict, obstacles: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    scenario = config.get("scenario", config)
    seed = int(scenario.get("random_seed", 123))
    x_min, x_max = (float(value) for value in scenario.get("x_range", [5.0, 95.0]))
    y_min, y_max = (float(value) for value in scenario.get("y_range", [5.0, 95.0]))
    distance_min, distance_max = (float(value) for value in scenario.get("distance_range", [5.0, 20.0]))
    min_obstacle_clearance = float(scenario.get("min_obstacle_clearance", 2.0))
    max_attempts = int(scenario.get("max_attempts", 5000))
    start_yaw = float(scenario.get("start_yaw", 0.0))
    goal_yaw = float(scenario.get("goal_yaw", 0.0))

    obstacle_array = np.asarray(obstacles, dtype=np.float32).reshape(-1, 7) if obstacles is not None else None
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        start_xy = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)],
            dtype=np.float32,
        )
        distance = float(rng.uniform(distance_min, distance_max))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        goal_xy = start_xy + distance * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

        if not (x_min <= goal_xy[0] <= x_max and y_min <= goal_xy[1] <= y_max):
            continue
        if obstacle_array is not None and (
            not _has_obstacle_clearance(start_xy, obstacle_array, min_obstacle_clearance)
            or not _has_obstacle_clearance(goal_xy, obstacle_array, min_obstacle_clearance)
        ):
            continue

        start_state = np.array([start_xy[0], start_xy[1], start_yaw, 0.0, 0.0, 0.0], dtype=np.float32)
        goal_state = np.array([goal_xy[0], goal_xy[1], goal_yaw, 0.0, 0.0, 0.0], dtype=np.float32)
        return start_state, goal_state

    raise ValueError(f"Unable to sample valid start/goal after {max_attempts} attempts")


def _has_obstacle_clearance(point_xy: np.ndarray, obstacles: np.ndarray, clearance: float) -> bool:
    for obstacle in obstacles:
        if np.linalg.norm(point_xy - obstacle[:2]) <= float(obstacle[2]) + clearance:
            return False
    return True
