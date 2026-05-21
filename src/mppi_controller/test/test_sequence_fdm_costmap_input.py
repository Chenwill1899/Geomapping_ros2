from __future__ import annotations

import csv
import json

import numpy as np
import torch


def test_raw_episode_windows_use_reward_cost_grid_without_feature_layers(tmp_path):
    from mppi_controller.data.sequence_fdm_collector import build_sequence_fdm_windows_from_raw_episode

    episode_dir = tmp_path / "episode_000"
    episode_dir.mkdir()
    with (episode_dir / "odom.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stamp", "rel_t", "x", "y", "yaw", "vx", "vy", "wz"])
        writer.writeheader()
        for step in range(4):
            writer.writerow(
                {
                    "stamp": float(step),
                    "rel_t": float(step),
                    "x": float(step),
                    "y": 0.0,
                    "yaw": 0.0,
                    "vx": 1.0,
                    "vy": 0.0,
                    "wz": 0.0,
                }
            )
    with (episode_dir / "cmd.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stamp", "rel_t", "linear_x", "linear_y", "angular_z"])
        writer.writeheader()
        for step in range(4):
            writer.writerow({"stamp": float(step), "rel_t": float(step), "linear_x": 1.0, "linear_y": 0.0, "angular_z": 0.0})
    (episode_dir / "episode.json").write_text(
        json.dumps({"goal": {"id": "g0", "x": 3.0, "y": 4.0, "yaw": 0.0}}),
        encoding="utf-8",
    )
    with (episode_dir / "tltrajectory.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "stamp": 0.0,
                    "rel_t": 0.0,
                    "length_m": 2.0,
                    "points": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                }
            )
            + "\n"
        )

    np.savez_compressed(
        episode_dir / "local_costmap.npz",
        stamp=np.asarray([0.0], dtype=np.float64),
        origin=np.asarray([[-4.0, -4.0]], dtype=np.float32),
        resolution=np.asarray([1.0], dtype=np.float32),
        width=np.asarray([9], dtype=np.int32),
        height=np.asarray([9], dtype=np.int32),
        reward_cost=np.arange(81, dtype=np.float32).reshape(1, 81),
        height_layer=np.empty((1, 0), dtype=np.float32),
        height_data=np.empty((1, 0), dtype=np.float32),
        roughness=np.empty((1, 0), dtype=np.float32),
        cost_map=np.empty((1, 0), dtype=np.float32),
    )

    windows = build_sequence_fdm_windows_from_raw_episode(
        episode_dir,
        horizon_steps=2,
        stride=1,
        costmap_grid_size=9,
        costmap_grid_span=8.0,
        costmap_max_age_s=10.0,
        costmap_max_value=100.0,
    )

    assert len(windows) == 2
    first = windows[0]
    assert first["costmap_grid"].shape == (81,)
    assert first["terrain_grid"].shape == (81,)
    assert first["goal_path_features"].shape == (10,)
    np.testing.assert_allclose(first["costmap_grid"], first["terrain_grid"])
    np.testing.assert_allclose(first["controls"], np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(first["target_states"][:, 0], np.asarray([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(
        first["goal_path_features"],
        np.asarray([3.0, 4.0, 5.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0], dtype=np.float32),
    )
    assert float(first["costmap_grid"].max()) <= 0.801
    assert float(first["costmap_grid"].max()) > 0.0


def test_sequence_fdm_v2_dataset_prefers_costmap_grid_key():
    from mppi_controller.core.sequence_fdm_v2 import GOAL_PATH_FEATURE_DIM
    from mppi_controller.training.sequence_fdm_v2 import SequenceFdmDataset

    costmap_grid = np.arange(81, dtype=np.float32)
    goal_path_features = np.arange(GOAL_PATH_FEATURE_DIM, dtype=np.float32)
    dataset = SequenceFdmDataset(
        [
            {
                "state": np.zeros(6, dtype=np.float32),
                "controls": np.zeros((2, 3), dtype=np.float32),
                "costmap_grid": costmap_grid,
                "goal_path_features": goal_path_features,
                "target_states": np.zeros((2, 6), dtype=np.float32),
                "target_risk": np.zeros(2, dtype=np.float32),
            }
        ],
        horizon_steps=2,
    )

    _state, _controls, grid, features, _target_states, _target_risk = dataset[0]

    assert isinstance(grid, torch.Tensor)
    np.testing.assert_allclose(grid.numpy(), costmap_grid)
    np.testing.assert_allclose(features.numpy(), goal_path_features)


def test_sequence_fdm_v2_uses_state_control_mlp_costmap_cnn_and_goal_path_features():
    from mppi_controller.core.sequence_fdm_v2 import (
        COSTMAP_GRID_DIM,
        GOAL_PATH_FEATURE_DIM,
        SequenceFdmMlpV2,
        build_feature_names_v2,
    )

    model = SequenceFdmMlpV2(horizon_steps=2, hidden_dims=[8])
    state = torch.zeros((3, 6), dtype=torch.float32)
    controls = torch.zeros((3, 2, 3), dtype=torch.float32)
    costmap_grid = torch.zeros(COSTMAP_GRID_DIM, dtype=torch.float32).unsqueeze(0).expand(3, -1)
    goal_path_features = torch.zeros((3, GOAL_PATH_FEATURE_DIM), dtype=torch.float32)

    states_pred, risk_logits = model(state, controls, costmap_grid, goal_path_features)

    assert hasattr(model, "state_control_encoder")
    assert hasattr(model, "costmap_encoder")
    assert len(build_feature_names_v2(2)) == 6 + 2 * 3 + COSTMAP_GRID_DIM + GOAL_PATH_FEATURE_DIM
    assert states_pred.shape == (3, 2, 6)
    assert risk_logits.shape == (3, 2)
