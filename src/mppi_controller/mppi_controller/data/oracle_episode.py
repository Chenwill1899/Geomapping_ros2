"""Convert one oracle simulation result directory into FDM episode arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STATE_COLUMNS = ["x", "y", "theta", "vx", "vy", "wz"]
CMD_COLUMNS = ["cmd_vx", "cmd_vy", "cmd_wz"]
REAL_COLUMNS = ["real_vx", "real_vy", "real_wz"]
EXEC_RESIDUAL_COLUMNS = ["exec_du_vx", "exec_du_vy", "exec_du_wz"]
ORACLE_RESIDUAL_COLUMNS = ["oracle_du_vx", "oracle_du_vy", "oracle_du_wz"]
TERRAIN_COLUMNS = ["slope_f", "slope_l", "roughness", "friction"]


def build_episode_npz(results_path: Path, episode_id: int, output_path: Path) -> dict:
    results_path = Path(results_path)
    output_path = Path(output_path)
    trajectory = pd.read_csv(results_path / "trajectory.csv")
    residuals = pd.read_csv(results_path / "residuals.csv")
    terrain = pd.read_csv(results_path / "terrain.csv")
    summary = json.loads((results_path / "summary.json").read_text(encoding="utf-8"))

    _require_columns(trajectory, STATE_COLUMNS + ["step"], "trajectory.csv")
    _require_columns(residuals, CMD_COLUMNS + REAL_COLUMNS + EXEC_RESIDUAL_COLUMNS + ORACLE_RESIDUAL_COLUMNS, "residuals.csv")
    _require_columns(terrain, TERRAIN_COLUMNS + ["risk_cost"], "terrain.csv")
    if len(trajectory) < 2:
        raise ValueError("trajectory.csv must contain at least 2 rows")

    num_transitions = len(residuals)
    if len(terrain) < num_transitions:
        raise ValueError(
            "terrain.csv must contain at least len(residuals) rows: "
            f"trajectory={len(trajectory)}, residuals={len(residuals)}, terrain={len(terrain)}"
        )
    if len(trajectory) < num_transitions + 1:
        raise ValueError(
            "trajectory.csv must contain one more row than residuals.csv to provide next_states: "
            f"trajectory={len(trajectory)}, residuals={len(residuals)}, terrain={len(terrain)}"
        )

    states = trajectory.loc[: num_transitions - 1, STATE_COLUMNS].to_numpy(dtype=np.float32)
    next_states = trajectory.loc[1:num_transitions, STATE_COLUMNS].to_numpy(dtype=np.float32)
    residual_rows = residuals.iloc[:num_transitions]
    terrain_rows = terrain.iloc[:num_transitions]
    cmd_controls = residual_rows.loc[:, CMD_COLUMNS].to_numpy(dtype=np.float32)
    real_controls = residual_rows.loc[:, REAL_COLUMNS].to_numpy(dtype=np.float32)
    exec_residuals = residual_rows.loc[:, EXEC_RESIDUAL_COLUMNS].to_numpy(dtype=np.float32)
    oracle_residuals = residual_rows.loc[:, ORACLE_RESIDUAL_COLUMNS].to_numpy(dtype=np.float32)
    terrain_features = terrain_rows.loc[:, TERRAIN_COLUMNS].to_numpy(dtype=np.float32)
    terrain_risk = terrain_rows.loc[:, "risk_cost"].to_numpy(dtype=np.float32)
    steps = trajectory.loc[: num_transitions - 1, "step"].to_numpy(dtype=np.int64)
    episode_ids = np.full(num_transitions, int(episode_id), dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states,
        next_states=next_states,
        cmd_controls=cmd_controls,
        real_controls=real_controls,
        exec_residuals=exec_residuals,
        oracle_residuals=oracle_residuals,
        terrain_features=terrain_features,
        terrain_risk=terrain_risk,
        episode_ids=episode_ids,
        steps=steps,
        success=np.asarray(bool(summary.get("success", summary.get("reached_goal", False)))),
        failed=np.asarray(bool(summary.get("failed", False))),
        start_goal_distance=np.asarray(float(summary.get("start_goal_distance", np.nan)), dtype=np.float32),
        final_distance=np.asarray(float(summary.get("final_distance", np.nan)), dtype=np.float32),
        min_obstacle_clearance=np.asarray(_optional_float(summary.get("min_obstacle_clearance")), dtype=np.float32),
    )

    return {
        "episode_id": int(episode_id),
        "num_transitions": int(num_transitions),
        "success": bool(summary.get("success", summary.get("reached_goal", False))),
        "failed": bool(summary.get("failed", False)),
        "start_goal_distance": float(summary.get("start_goal_distance", np.nan)),
        "final_distance": float(summary.get("final_distance", np.nan)),
        "min_obstacle_clearance": _optional_float(summary.get("min_obstacle_clearance")),
        "results_path": str(results_path),
        "output_path": str(output_path),
    }


def _require_columns(frame: pd.DataFrame, columns: list[str], filename: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{filename} missing required columns: {missing}")


def _optional_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)
