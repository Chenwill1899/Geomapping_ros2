"""Collect a single sequence FDM episode on random terrain."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml

from mppi_controller.config import load_config
from mppi_controller.core.terrain import TerrainField
from mppi_controller.data.collect_oracle_episode import collect_oracle_episode
from mppi_controller.simulation.random_terrain import RandomTerrainGenerator


def _mark_binary_risk(terrain_risk: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Return binary array where once threshold exceeded, all subsequent are 1."""
    binary = np.zeros_like(terrain_risk, dtype=np.float32)
    if terrain_risk.size == 0:
        return binary
    idx = int(np.argmax(terrain_risk > threshold))
    if terrain_risk[idx] > threshold:
        binary[idx:] = 1.0
    return binary


def _sample_start_goal(
    rng: np.random.Generator,
    map_bounds: tuple[float, float, float, float],
    min_distance: float = 10.0,
    max_attempts: int = 100,
    terrain: TerrainField | None = None,
    max_start_goal_risk: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample start and goal positions at least min_distance apart.

    Args:
        rng: random generator
        map_bounds: (x_min, x_max, y_min, y_max)
        min_distance: minimum Euclidean distance between start and goal
        max_attempts: max sampling attempts before raising
        terrain: if provided, start and goal must both have risk_cost <= max_start_goal_risk
        max_start_goal_risk: maximum terrain risk for valid start/goal positions
    """
    x_min, x_max, y_min, y_max = map_bounds
    for _ in range(max_attempts):
        start = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float32
        )
        goal = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float32
        )
        if np.linalg.norm(goal - start) < min_distance:
            continue
        if terrain is not None:
            if terrain.risk_cost(float(start[0]), float(start[1])) > max_start_goal_risk:
                continue
            if terrain.risk_cost(float(goal[0]), float(goal[1])) > max_start_goal_risk:
                continue
        return start, goal
    raise RuntimeError(
        f"Could not sample start/goal pair within {max_attempts} attempts "
        f"(terrain risk threshold={max_start_goal_risk})"
    )


def _terrain_to_config(terrain: TerrainField) -> dict:
    """Build a config dict from a TerrainField's public attributes."""
    return {
        "enabled": terrain.enabled,
        "slope_scale": terrain.slope_scale,
        "slope_wave": terrain.slope_wave,
        "roughness_scale": terrain.roughness_scale,
        "roughness_wave": terrain.roughness_wave,
        "friction_base": terrain.friction_base,
        "friction_slope_scale": terrain.friction_slope_scale,
        "friction_roughness_scale": terrain.friction_roughness_scale,
        "risk_weights": terrain.risk_weights,
        "goal_relief": terrain.goal_relief,
        "noise_enabled": terrain.noise_enabled,
        "noise_seed": terrain.noise_seed,
        "noise_grid_size": terrain.noise_grid_size,
        "noise_scale": terrain.noise_scale,
        "noise_smooth_passes": terrain.noise_smooth_passes,
        "noise_roughness_weight": terrain.noise_roughness_weight,
        "noise_friction_weight": terrain.noise_friction_weight,
        "noise_slope_weight": terrain.noise_slope_weight,
        "noise_x_range": terrain.noise_x_range,
        "noise_y_range": terrain.noise_y_range,
        "patches": terrain.patches,
    }


def _convert_tuples_to_lists(obj):
    """Recursively convert tuples to lists for YAML serialization."""
    if isinstance(obj, tuple):
        return [_convert_tuples_to_lists(v) for v in obj]
    if isinstance(obj, list):
        return [_convert_tuples_to_lists(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _convert_tuples_to_lists(v) for k, v in obj.items()}
    return obj


def collect_sequence_fdm_episode(
    *,
    base_config_path: str | Path,
    episode_id: int,
    terrain_seed: int,
    output_dir: str | Path,
    map_bounds: tuple[float, float, float, float] = (-20.0, 20.0, -20.0, 20.0),
    min_start_goal_distance: float = 10.0,
    risk_threshold: float = 0.6,
    num_patches_range: tuple[int, int] = (3, 6),
    num_trajectories: int = 1,
) -> list[dict]:
    """Generate random terrain, run MPPI, and save an episode with binary risk labels.

    When num_trajectories > 1, the same terrain is reused but each trajectory
    gets independent start/goal positions and a different MPPI noise seed.
    Returns a list of metadata dicts, one per trajectory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate random terrain once (before the loop)
    rng = np.random.default_rng(terrain_seed)
    terrain_gen = RandomTerrainGenerator(
        map_bounds=map_bounds,
        num_patches_range=num_patches_range,
    )
    terrain = terrain_gen.generate(seed=terrain_seed)

    # Sample obstacles once (same for all trajectories)
    rng_obs = np.random.default_rng(terrain_seed + 50000)
    num_obs = int(rng_obs.integers(3, 7))
    obs_list = []
    for _ in range(num_obs):
        ox = float(rng_obs.uniform(map_bounds[0] + 2.0, map_bounds[1] - 2.0))
        oy = float(rng_obs.uniform(map_bounds[2] + 2.0, map_bounds[3] - 2.0))
        radius = float(rng_obs.uniform(0.8, 2.0))
        obs_list.append([ox, oy, radius, 0.0, 0.0, 0.0, 0.0])

    all_metadata: list[dict] = []

    for jj in range(num_trajectories):
        # Sample start and goal independently per trajectory
        start_goal_rng = np.random.default_rng(terrain_seed + jj * 1000)
        start_xy, goal_xy = _sample_start_goal(
            start_goal_rng, map_bounds,
            min_distance=min_start_goal_distance,
            terrain=terrain,
            max_start_goal_risk=0.5,
        )

        # Load and override base config
        config = load_config(base_config_path)
        config["terrain"] = _terrain_to_config(terrain)
        config.setdefault("mppi", {})["backend"] = "torch"
        config["simulation"]["max_steps"] = 500
        # Disable expensive visualization to speed up collection
        config.setdefault("results", {})["enable_plots"] = False
        config.setdefault("results", {})["enable_animation"] = False
        # Enable terrain risk avoidance so MPPI avoids high-risk patches
        mppi_cfg = config.setdefault("mppi", {})
        mppi_cfg["terrain_risk_weight"] = 1.0
        mppi_cfg["terrain_risk_threshold"] = risk_threshold
        mppi_cfg["terrain_risk_mode"] = "excess"
        config["obstacles"] = {
            "num_max": num_obs,
            "static_enabled": True,
            "virtual": obs_list,
        }

        initial_state = list(config["simulation"]["initial_state"])
        goal_state = list(config["simulation"]["goal"])
        initial_state[0] = float(start_xy[0])
        initial_state[1] = float(start_xy[1])
        goal_state[0] = float(goal_xy[0])
        goal_state[1] = float(goal_xy[1])
        config["simulation"]["initial_state"] = initial_state
        config["simulation"]["goal"] = goal_state

        # Output filename: episode_{id:06d}_traj_{jj:02d}.npz
        output_path = output_dir / f"episode_{int(episode_id):06d}_traj_{jj:02d}.npz"

        # Write temporary config (convert tuples to lists for YAML compatibility)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            yaml.dump(_convert_tuples_to_lists(config), temp_file)
            temp_config_path = Path(temp_file.name)

        try:
            # Collect oracle episode; MPPI seed = terrain_seed + jj for noise diversity
            metadata = collect_oracle_episode(
                config_path=temp_config_path,
                episode_id=episode_id,
                seed=terrain_seed + jj,
                output_path=output_path,
            )

            # Load saved NPZ, augment, and re-save
            data = dict(np.load(output_path, allow_pickle=True))
            for key in data:
                if isinstance(data[key], np.ndarray) and data[key].dtype == object:
                    data[key] = data[key].item()

            terrain_risk = data["terrain_risk"]
            binary_risk = _mark_binary_risk(terrain_risk, threshold=risk_threshold)
            data["binary_risk"] = binary_risk
            data["terrain_seed"] = np.asarray(int(terrain_seed), dtype=np.int64)
            data["start_xy"] = start_xy.astype(np.float32)
            data["goal_xy"] = goal_xy.astype(np.float32)
            data["traj_idx"] = np.asarray(int(jj), dtype=np.int64)

            np.savez_compressed(output_path, **data)

            metadata["binary_risk"] = binary_risk.tolist()
            metadata["terrain_seed"] = int(terrain_seed)
            metadata["traj_idx"] = int(jj)
            metadata["start_xy"] = start_xy.tolist()
            metadata["goal_xy"] = goal_xy.tolist()
            metadata["output_path"] = str(output_path)

            all_metadata.append(metadata)
        finally:
            temp_config_path.unlink(missing_ok=True)

    return all_metadata


def build_sequence_fdm_windows(
    episode_path: str | Path,
    horizon_steps: int,
    stride: int = 1,
    map_bounds: tuple[float, float, float, float] = (-20.0, 20.0, -20.0, 20.0),
    terrain_grid_size: int = 9,
    terrain_grid_span: float = 18.0,
) -> list[dict]:
    """Extract sliding windows from a collected episode for sequence FDM training.

    Returns a list of dicts, each containing one training sample.
    """
    from mppi_controller.core.terrain_grid import sample_terrain_risk_grid_np

    episode = np.load(episode_path, allow_pickle=True)
    states = episode["states"].astype(np.float32)
    controls = episode["cmd_controls"].astype(np.float32)
    binary_risk = episode["binary_risk"].astype(np.float32)
    terrain_seed = int(episode["terrain_seed"].item())
    episode.close()

    # Reconstruct terrain for grid sampling
    generator = RandomTerrainGenerator(map_bounds=map_bounds)
    terrain = generator.generate(seed=terrain_seed)

    T = len(states)
    windows: list[dict] = []
    for t in range(0, T - horizon_steps, stride):
        state_t = states[t]
        controls_t = controls[t : t + horizon_steps]
        terrain_grid = sample_terrain_risk_grid_np(
            terrain, float(state_t[0]), float(state_t[1]),
            size=terrain_grid_size, span=terrain_grid_span,
        )
        target_states = states[t + 1 : t + 1 + horizon_steps]
        target_risk = binary_risk[t + 1 : t + 1 + horizon_steps]

        windows.append({
            "state": state_t,
            "controls": controls_t,
            "terrain_grid": terrain_grid,
            "target_states": target_states,
            "target_risk": target_risk,
            "episode_id": str(episode_path),
            "timestep": t,
        })
    return windows


def build_sequence_fdm_windows_from_raw_episode(
    episode_dir: str | Path,
    horizon_steps: int,
    stride: int = 1,
    costmap_grid_size: int = 9,
    costmap_grid_span: float = 18.0,
    costmap_max_age_s: float = 0.75,
    costmap_max_value: float = 100.0,
    risk_threshold: float = 0.6,
) -> list[dict]:
    """Extract sequence-FDM windows from a raw Geomapping episode directory.

    The map input is costmap-only: it samples the saved ``reward_cost`` layer
    from ``local_costmap.npz`` and ignores height/roughness/cost_map layers.
    """
    from mppi_controller.core.sequence_fdm_v2 import COSTMAP_GRID_DIM, COSTMAP_GRID_SIZE
    from mppi_controller.core.terrain_grid import sample_local_costmap_grid_np

    episode_dir = Path(episode_dir)
    if int(costmap_grid_size) * int(costmap_grid_size) != COSTMAP_GRID_DIM:
        raise ValueError(
            f"costmap_grid_size must be {COSTMAP_GRID_SIZE} so the flattened input is {COSTMAP_GRID_DIM}-D"
        )
    odom = _read_numeric_csv(episode_dir / "odom.csv")
    cmd = _read_numeric_csv(episode_dir / "cmd.csv")
    goal = _read_episode_goal(episode_dir)
    path_records = _read_path_records(episode_dir / "tltrajectory.jsonl")
    if not path_records:
        path_records = _read_path_records(episode_dir / "frontend_path.jsonl")
    localmap = np.load(episode_dir / "local_costmap.npz")
    try:
        stamps = np.asarray(odom["stamp"], dtype=np.float64)
        states = np.stack(
            [
                np.asarray(odom["x"], dtype=np.float32),
                np.asarray(odom["y"], dtype=np.float32),
                np.asarray(odom["yaw"], dtype=np.float32),
                np.asarray(odom["vx"], dtype=np.float32),
                np.asarray(odom["vy"], dtype=np.float32),
                np.asarray(odom["wz"], dtype=np.float32),
            ],
            axis=1,
        )
        commands = _commands_at_stamps(cmd, stamps)
        costmap_stamps = np.asarray(localmap["stamp"], dtype=np.float64)
        reward_cost = np.asarray(localmap["reward_cost"], dtype=np.float32)
        origins = np.asarray(localmap["origin"], dtype=np.float32)
        resolutions = np.asarray(localmap["resolution"], dtype=np.float32)
        widths = np.asarray(localmap["width"], dtype=np.int32)
        heights = np.asarray(localmap["height"], dtype=np.int32)
    finally:
        localmap.close()

    horizon = int(horizon_steps)
    if horizon <= 0:
        raise ValueError("horizon_steps must be positive")
    if costmap_stamps.size == 0 or reward_cost.size == 0:
        return []

    windows: list[dict] = []
    max_start = len(states) - horizon
    for start in range(0, max_start, int(stride)):
        map_index = _nearest_index(costmap_stamps, stamps[start])
        if map_index is None or abs(float(costmap_stamps[map_index] - stamps[start])) > float(costmap_max_age_s):
            continue
        state_t = states[start]
        costmap_grid = sample_local_costmap_grid_np(
            reward_cost[map_index],
            origin=origins[map_index],
            resolution=float(resolutions[map_index]),
            width=int(widths[map_index]),
            height=int(heights[map_index]),
            x=float(state_t[0]),
            y=float(state_t[1]),
            size=int(costmap_grid_size),
            span=float(costmap_grid_span),
            max_value=float(costmap_max_value),
        )
        target_states = states[start + 1 : start + 1 + horizon]
        path_points, path_length = _path_at_stamp(path_records, stamps[start])
        goal_path_features = _goal_path_features(state_t, goal, path_points, path_length)
        target_risk = _target_risk_from_costmaps(
            stamps[start + 1 : start + 1 + horizon],
            target_states,
            costmap_stamps=costmap_stamps,
            reward_cost=reward_cost,
            origins=origins,
            resolutions=resolutions,
            widths=widths,
            heights=heights,
            max_age_s=float(costmap_max_age_s),
            max_value=float(costmap_max_value),
            threshold=float(risk_threshold),
        )
        windows.append(
            {
                "state": state_t.astype(np.float32, copy=False),
                "controls": commands[start : start + horizon].astype(np.float32, copy=False),
                "costmap_grid": costmap_grid.astype(np.float32, copy=False),
                # Kept as a compatibility alias for the existing V2 trainer.
                "terrain_grid": costmap_grid.astype(np.float32, copy=False),
                "goal_path_features": goal_path_features.astype(np.float32, copy=False),
                "target_states": target_states.astype(np.float32, copy=False),
                "target_risk": target_risk.astype(np.float32, copy=False),
                "episode_id": str(episode_dir),
                "timestep": int(start),
                "costmap_stamp": float(costmap_stamps[map_index]),
                "costmap_age_s": float(abs(costmap_stamps[map_index] - stamps[start])),
            }
        )
    return windows


def _target_risk_from_costmaps(
    stamps: np.ndarray,
    states: np.ndarray,
    *,
    costmap_stamps: np.ndarray,
    reward_cost: np.ndarray,
    origins: np.ndarray,
    resolutions: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    max_age_s: float,
    max_value: float,
    threshold: float,
) -> np.ndarray:
    from mppi_controller.core.terrain_grid import sample_local_costmap_grid_np

    risks = np.zeros(len(states), dtype=np.float32)
    for idx, state in enumerate(states):
        map_index = _nearest_index(costmap_stamps, stamps[idx])
        if map_index is None or abs(float(costmap_stamps[map_index] - stamps[idx])) > max_age_s:
            continue
        sampled = sample_local_costmap_grid_np(
            reward_cost[map_index],
            origin=origins[map_index],
            resolution=float(resolutions[map_index]),
            width=int(widths[map_index]),
            height=int(heights[map_index]),
            x=float(state[0]),
            y=float(state[1]),
            size=1,
            span=0.0,
            max_value=max_value,
        )
        risks[idx] = 1.0 if float(sampled[0]) >= threshold else 0.0
    return risks


def _read_episode_goal(episode_dir: Path) -> np.ndarray:
    episode_path = episode_dir / "episode.json"
    if not episode_path.exists():
        return np.zeros(3, dtype=np.float32)
    data = json.loads(episode_path.read_text(encoding="utf-8"))
    goal = data.get("goal", {})
    return np.asarray(
        [
            float(goal.get("x", 0.0)),
            float(goal.get("y", 0.0)),
            float(goal.get("yaw", 0.0)),
        ],
        dtype=np.float32,
    )


def _read_path_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "stamp" in record and isinstance(record.get("points"), list):
                records.append(record)
    return sorted(records, key=lambda item: float(item["stamp"]))


def _path_at_stamp(records: list[dict], stamp: float) -> tuple[np.ndarray, float]:
    if not records:
        return np.empty((0, 2), dtype=np.float32), 0.0
    stamps = np.asarray([float(record["stamp"]) for record in records], dtype=np.float64)
    index = _nearest_index(stamps, float(stamp))
    if index is None:
        return np.empty((0, 2), dtype=np.float32), 0.0
    record = records[index]
    points = np.asarray(record.get("points", []), dtype=np.float32).reshape(-1, 2)
    length = float(record.get("length_m", 0.0))
    if length <= 0.0 and len(points) >= 2:
        length = float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))
    return points, length


def _goal_path_features(
    state: np.ndarray,
    goal: np.ndarray,
    path_points: np.ndarray,
    path_length: float,
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(6)
    goal = np.asarray(goal, dtype=np.float32).reshape(3)
    rel_goal = _world_to_body_delta(state, goal[:2])
    goal_distance = float(np.linalg.norm(goal[:2] - state[:2]))
    goal_yaw_error = _angle_diff(float(goal[2]), float(state[2]))

    path_available = 1.0 if path_points.shape[0] >= 2 else 0.0
    if path_available:
        lookahead = _world_to_body_delta(state, path_points[min(1, path_points.shape[0] - 1)])
        path_end = _world_to_body_delta(state, path_points[-1])
    else:
        lookahead = np.zeros(2, dtype=np.float32)
        path_end = np.zeros(2, dtype=np.float32)
        path_length = 0.0
    return np.asarray(
        [
            rel_goal[0],
            rel_goal[1],
            goal_distance,
            goal_yaw_error,
            lookahead[0],
            lookahead[1],
            path_end[0],
            path_end[1],
            float(path_length),
            path_available,
        ],
        dtype=np.float32,
    )


def _world_to_body_delta(state: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    dx = float(point_xy[0]) - float(state[0])
    dy = float(point_xy[1]) - float(state[1])
    yaw = float(state[2])
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    return np.asarray(
        [
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy,
        ],
        dtype=np.float32,
    )


def _angle_diff(a: float, b: float) -> float:
    return float((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi)


def _read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        return {}
    columns = {name: [] for name in rows[0]}
    for row in rows:
        for name, value in row.items():
            columns[name].append(float(value))
    return {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


def _commands_at_stamps(cmd: dict[str, np.ndarray], stamps: np.ndarray) -> np.ndarray:
    if not cmd:
        return np.zeros((len(stamps), 3), dtype=np.float32)
    cmd_stamps = np.asarray(cmd["stamp"], dtype=np.float64)
    command_values = np.stack(
        [
            np.asarray(cmd["linear_x"], dtype=np.float32),
            np.asarray(cmd["linear_y"], dtype=np.float32),
            np.asarray(cmd["angular_z"], dtype=np.float32),
        ],
        axis=1,
    )
    indices = np.searchsorted(cmd_stamps, stamps, side="right") - 1
    indices = np.clip(indices, 0, len(command_values) - 1)
    return command_values[indices].astype(np.float32, copy=False)


def _nearest_index(values: np.ndarray, target: float) -> int | None:
    if values.size == 0:
        return None
    index = int(np.searchsorted(values, float(target), side="left"))
    candidates = []
    if index < values.size:
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    return min(candidates, key=lambda idx: abs(float(values[idx] - target)))
