from __future__ import annotations

import json

import numpy as np
import pytest
import torch


class _FakeHfdm(torch.nn.Module):
    def forward(self, history, local_map, actions):
        batch, horizon, _ = actions.shape
        pose = torch.zeros((batch, horizon, 4), dtype=actions.dtype)
        pose[..., 0] = torch.cumsum(actions[..., 0], dim=1) * 0.1
        pose[..., 1] = torch.cumsum(actions[..., 1], dim=1) * 0.1
        pose[..., 2] = torch.sin(torch.cumsum(actions[..., 2], dim=1) * 0.1)
        pose[..., 3] = torch.cos(torch.cumsum(actions[..., 2], dim=1) * 0.1)
        risk = torch.zeros((batch, horizon, 6), dtype=actions.dtype)
        risk[..., 5] = torch.clamp(actions[..., 0], min=0.0, max=1.0)
        return pose, risk, actions


def _write_fake_export(path, *, horizon: int = 3) -> None:
    path.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(
        _FakeHfdm(),
        (
            torch.zeros(1, 10, 10),
            torch.zeros(1, 1, 64, 64),
            torch.zeros(1, horizon, 3),
        ),
    )
    traced.save(str(path / "fdm_ts.pt"))
    (path / "fdm_metadata.json").write_text(
        json.dumps(
            {
                "horizon": horizon,
                "history_len": 10,
                "map_size": 64,
                "map_channels": 1,
                "dt": 0.1,
                "risk_names": [
                    "collision",
                    "high_cost",
                    "stuck",
                    "tracking_failure",
                    "untraversable",
                    "any",
                ],
                "map_max_cost": 100.0,
            }
        ),
        encoding="utf-8",
    )


def _minimal_config(model_dir) -> dict:
    return {
        "simulation": {
            "sampling_rate": 10.0,
            "time_horizon": 0.3,
            "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "goal": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "minimum_distance": 0.5,
            "max_steps": 10,
        },
        "mppi": {
            "backend": "torch",
            "state_dim": 6,
            "control_dim": 3,
            "num_trajectories": 4,
            "draw_num_traj": 2,
            "lambda": 1.0,
            "weights": [1.0, 1.0, 0.1, 0.0, 0.0, 0.0],
            "std_normal": [0.1, 0.1, 0.1],
            "control_weight": 0.0,
            "smooth_weight": 0.0,
            "accel_weight": 0.0,
            "jerk_weight": 0.0,
            "path_tracking_weight": 0.0,
            "path_progress_weight": 0.0,
            "goal_progress_weight": 0.0,
            "heading_to_goal_weight": 0.0,
        },
        "robot": {
            "state_dim": 6,
            "max_vx": 1.0,
            "min_vx": 0.0,
            "max_vy": 0.3,
            "max_wz": 1.2,
            "max_ax": 10.0,
            "max_ay": 10.0,
            "max_awz": 10.0,
            "velocity_lag_beta": 0.0,
            "radius": 0.55,
            "safety_dist": 0.2,
        },
        "cbf": {},
        "obstacles": {"virtual": []},
        "results": {},
        "fdm": {
            "enabled": True,
            "mode": "high_level_fdm",
            "model_dir": str(model_dir),
            "device": "cpu",
            "risk_weight": 10.0,
            "risk_threshold": 0.4,
            "risk_power": 2.0,
        },
    }


def test_high_level_fdm_runtime_loads_torchscript_and_metadata(tmp_path):
    from mppi_controller.core.high_level_fdm_runtime import HighLevelFdmRuntime

    _write_fake_export(tmp_path, horizon=3)
    runtime = HighLevelFdmRuntime.from_model_dir(tmp_path, device="cpu")

    pose, risk, applied = runtime.predict(
        torch.zeros(2, 10, 10),
        torch.zeros(2, 1, 64, 64),
        torch.zeros(2, 3, 3),
    )

    assert runtime.metadata.horizon == 3
    assert runtime.metadata.history_len == 10
    assert runtime.risk_any_index == 5
    assert pose.shape == (2, 3, 4)
    assert risk.shape == (2, 3, 6)
    assert applied.shape == (2, 3, 3)


def test_robot_frame_hfdm_pose_converts_to_world_frame(tmp_path):
    from mppi_controller.controllers.mppi_omni_high_level_fdm_torch import (
        MppiOmniHighLevelFdmTorch,
    )

    rel_pose = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]]])
    states = MppiOmniHighLevelFdmTorch.relative_pose_to_world_states(
        np.asarray([2.0, 3.0, np.pi / 2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        rel_pose,
    )

    np.testing.assert_allclose(states[0, 1, :3].cpu().numpy(), [2.0, 4.0, np.pi / 2.0], atol=1e-5)
    np.testing.assert_allclose(states[0, 2, :3].cpu().numpy(), [1.0, 4.0, np.pi], atol=1e-5)


def test_learned_any_risk_cost_penalizes_risky_candidates(tmp_path):
    from mppi_controller.controllers.mppi_omni_high_level_fdm_torch import (
        MppiOmniHighLevelFdmTorch,
    )

    _write_fake_export(tmp_path, horizon=3)
    controller = MppiOmniHighLevelFdmTorch.from_config(_minimal_config(tmp_path), seed=1)
    low = torch.zeros((1, 3, 6), dtype=torch.float32)
    high = torch.zeros((1, 3, 6), dtype=torch.float32)
    high[..., 5] = 0.9

    low_cost = controller._learned_risk_cost_torch(low)
    high_cost = controller._learned_risk_cost_torch(high)

    assert float(low_cost[0]) == 0.0
    assert float(high_cost[0]) > float(low_cost[0])
    assert float(high_cost[0]) == pytest.approx(10.0 * 3 * (0.9 - 0.4) ** 2)


def test_create_omni_controller_selects_high_level_fdm(tmp_path):
    from mppi_controller.controllers.mppi_omni_high_level_fdm_torch import (
        MppiOmniHighLevelFdmTorch,
    )
    from mppi_controller.simulation.omni_runner import create_omni_controller

    _write_fake_export(tmp_path, horizon=3)
    controller = create_omni_controller(_minimal_config(tmp_path), seed=1)

    assert isinstance(controller, MppiOmniHighLevelFdmTorch)
