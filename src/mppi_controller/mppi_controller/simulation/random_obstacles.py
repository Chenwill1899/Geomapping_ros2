"""Random obstacle generation for oracle simulation scenarios."""

from __future__ import annotations

import numpy as np


def generate_random_obstacles(config: dict, start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    num_random = int(config.get("num_random", 0))
    seed = int(config.get("random_seed", 0))
    radius_min, radius_max = (float(value) for value in config.get("radius_range", [0.5, 1.0]))
    x_min, x_max = (float(value) for value in config.get("x_range", [0.0, 1.0]))
    y_min, y_max = (float(value) for value in config.get("y_range", [0.0, 1.0]))
    min_gap = float(config.get("min_obstacle_gap", 0.0))
    min_start_goal_clearance = float(config.get("min_start_goal_clearance", 0.0))
    max_attempts = int(config.get("max_attempts", 5000))

    rng = np.random.default_rng(seed)
    start_xy = np.asarray(start_xy, dtype=np.float32)[:2]
    goal_xy = np.asarray(goal_xy, dtype=np.float32)[:2]
    obstacles: list[list[float]] = []

    attempts = 0
    while len(obstacles) < num_random and attempts < max_attempts:
        attempts += 1
        radius = float(rng.uniform(radius_min, radius_max))
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        center = np.array([x, y], dtype=np.float32)

        if np.linalg.norm(center - start_xy) <= radius + min_start_goal_clearance:
            continue
        if np.linalg.norm(center - goal_xy) <= radius + min_start_goal_clearance:
            continue

        valid = True
        for existing in obstacles:
            existing_center = np.array(existing[:2], dtype=np.float32)
            existing_radius = float(existing[2])
            if np.linalg.norm(center - existing_center) <= radius + existing_radius + min_gap:
                valid = False
                break
        if not valid:
            continue

        obstacles.append([x, y, radius, 0.0, 0.0, 0.0, 0.0])

    if len(obstacles) != num_random:
        raise ValueError(
            "Random obstacle parameters are too crowded: "
            f"generated {len(obstacles)} of {num_random} after {max_attempts} attempts"
        )

    return np.asarray(obstacles, dtype=np.float32).reshape(-1, 7)
