"""CSV/YAML result writers shared by CLI and ROS 2 node simulation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml


def save_results(
    state_history: Sequence[np.ndarray],
    desired_state_history: Sequence[np.ndarray],
    state_cost_history: Sequence[float],
    control_cost_history: Sequence[float],
    control_history: Sequence[np.ndarray],
    results_path: Path,
) -> None:
    rows = []
    for state, target, state_cost, control_cost, control in zip(
        state_history,
        desired_state_history,
        state_cost_history,
        control_cost_history,
        control_history,
    ):
        rows.append(
            {
                "x": state[0],
                "y": state[1],
                "theta": state[2],
                "vx": state[3],
                "vy": state[4],
                "dx": state[3],
                "dy": state[4],
                "x_des": target[0],
                "y_des": target[1],
                "theta_des": target[2],
                "state_cost": state_cost,
                "control_cost": control_cost,
                "v": control[0],
                "w": control[1],
            }
        )
    pd.DataFrame(rows).to_csv(results_path / "results.csv", index=False)


def save_obs_results(
    obs_num_max: int,
    results_path: Path,
    obs_state_history: Sequence[Sequence[np.ndarray]],
) -> None:
    rows = []
    for states in obs_state_history:
        row = {}
        for idx in range(obs_num_max):
            if idx < len(states):
                obstacle = np.asarray(states[idx])
            else:
                obstacle = np.zeros(7, dtype=np.float32)
            row.update(
                {
                    f"x{idx}": obstacle[0],
                    f"y{idx}": obstacle[1],
                    f"r{idx}": obstacle[2],
                    f"b{idx}": obstacle[3],
                    f"theta{idx}": obstacle[4],
                    f"dx{idx}": obstacle[5],
                    f"dy{idx}": obstacle[6],
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(results_path / "obs_results.csv", index=False)


def save_time_results(results_path: Path, mppi_time_history: Sequence[float]) -> None:
    pd.DataFrame({"mppi_time_ms": mppi_time_history}).to_csv(
        results_path / "time_results.csv", index=False
    )


def save_summary(results_path: Path, summary: dict) -> None:
    with (results_path / "test_summary.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(summary, stream, sort_keys=False)
