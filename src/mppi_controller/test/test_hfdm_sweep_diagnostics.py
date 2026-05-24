from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "tools" / "analyze_hfdm_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_hfdm_sweep", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analyze_run_reports_path_and_control_oscillation_metrics(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    native = run_dir / "native_run"
    native.mkdir(parents=True)
    (run_dir / "odom.csv").write_text(
        "\n".join(
            [
                "t,x,y,yaw,vx,vy,wz",
                "0,0,0,0,0.5,0.00,0.0",
                "1,5,5,0.785398,0.5,0.20,0.2",
                "2,10,0,0,0.5,-0.10,-0.1",
                "3,10,0,0,0.5,0.30,0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (native / "controls.csv").write_text(
        "\n".join(
            [
                "step,vx_cmd,vy_cmd,wz_cmd",
                "0,0.5,0.00,0.0",
                "1,0.5,0.20,0.2",
                "2,0.5,-0.10,-0.1",
                "3,0.5,0.30,0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (native / "planning_goals.csv").write_text(
        "\n".join(
            [
                "step,plan_id,x,y,theta",
                "0,,10,0,0",
                "1,,10,0,0",
                "2,,10,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (native / "summary.json").write_text('{"reached_goal": true, "arrival_time": 3.0}', encoding="utf-8")

    metrics = module.analyze_run(run_dir, goal=(10.0, 0.0))

    assert metrics["path_length_m"] == pytest.approx(2.0 * math.sqrt(50.0))
    assert metrics["straight_ratio"] == pytest.approx(math.sqrt(2.0))
    assert metrics["max_line_deviation_m"] == pytest.approx(5.0)
    assert metrics["total_heading_change_rad"] == pytest.approx(math.pi / 2.0, rel=1e-5)
    assert metrics["odom_wz_sign_switches"] == 2
    assert metrics["control_wz_sign_switches"] == 2
    assert metrics["mean_abs_vy"] == pytest.approx(0.15)
    assert metrics["mean_abs_wz"] == pytest.approx(0.1)
    assert metrics["planning_goal_unique_count"] == 1
    assert metrics["planning_goal_is_constant_final_goal"] is True


def test_build_tuned_profile_keeps_no_frontend_and_adds_mild_regularization() -> None:
    module = _load_module()
    source = {
        "external_path": {"enabled": True},
        "global_path": {"enabled": True},
        "final_controller": {"disable_when_local_costmap": False},
        "command_filter": {"enabled": False, "alpha": 0.0, "lateral_deadband": 0.0, "yaw_deadband": 0.0},
        "mppi": {
            "smooth_weight": 0.0,
            "accel_weight": 0.0,
            "lateral_weight": 0.0,
            "yaw_rate_weight": 0.0,
            "jerk_weight": 0.0,
            "update_smoothing_alpha": [0.0, 0.0, 0.0],
            "path_tracking_weight": 2.0,
            "path_progress_weight": 2.0,
            "learned_risk_weight": 4.0,
        },
        "controllers": [{"fdm": {"mode": "high_level_fdm"}}],
    }

    tuned = module.build_tuned_no_frontend_profile(source)

    assert tuned["external_path"]["enabled"] is False
    assert tuned["global_path"]["enabled"] is False
    assert tuned["final_controller"]["disable_when_local_costmap"] is True
    assert tuned["mppi"]["path_tracking_weight"] == pytest.approx(0.0)
    assert tuned["mppi"]["path_progress_weight"] == pytest.approx(0.0)
    assert tuned["mppi"]["smooth_weight"] == pytest.approx(0.6)
    assert tuned["mppi"]["accel_weight"] == pytest.approx(0.12)
    assert tuned["mppi"]["lateral_weight"] == pytest.approx(0.12)
    assert tuned["mppi"]["yaw_rate_weight"] == pytest.approx(0.12)
    assert tuned["mppi"]["jerk_weight"] == pytest.approx(0.2)
    assert tuned["mppi"]["update_smoothing_alpha"] == pytest.approx([0.12, 0.45, 0.45])
    assert tuned["command_filter"]["enabled"] is True
    assert tuned["controllers"][0]["fdm"]["mode"] == "high_level_fdm"
