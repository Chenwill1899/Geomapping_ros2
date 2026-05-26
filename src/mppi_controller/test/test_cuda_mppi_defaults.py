from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(relative_path: str) -> dict:
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def _assert_nominal_cuda_profile(relative_path: str) -> None:
    profile = _load_yaml(relative_path)
    assert profile["default_controller"] == "nominal_cuda"
    controllers = {controller["name"]: controller for controller in profile["controllers"]}
    assert "nominal_numpy" not in controllers
    assert controllers["nominal_cuda"]["method"] == "nominal"
    assert controllers["nominal_cuda"]["backend"] == "cuda"


def test_frontend_profile_defaults_to_cuda_controller():
    _assert_nominal_cuda_profile("src/mppi_controller/configs/mujoco_rviz_goal.yaml")


def test_frontend_profile_keeps_obstacle_safety_cost_enabled():
    profiles = [
        _load_yaml("src/mppi_controller/configs/mujoco_rviz_goal.yaml"),
        _load_yaml("src/mppi_controller/configs/mujoco_rviz_goal_no_frontend.yaml"),
    ]

    for profile in profiles:
        assert profile["robot"]["safety_dist"] > 0.0
        assert profile["mppi"]["obstacle_weight"] > 0.0
        assert profile["mppi"]["obstacle_collision_weight"] > 0.0
        assert profile["mppi"]["obstacle_soft_weight"] > 0.0
        assert profile["mppi"]["obstacle_influence_dist"] > 0.0
        assert profile["local_costmap"]["footprint"]["enabled"] is True
        assert profile["local_costmap"]["footprint"]["safety_margin"] > 0.0
        assert profile["dynamic_obstacles"]["enabled"] is True
        assert profile["dynamic_obstacles"]["topic"] == "/dyn_obstacle"


def test_frontend_profile_recovers_from_external_path_stagnation():
    profile = _load_yaml("src/mppi_controller/configs/mujoco_rviz_goal.yaml")
    recovery = profile["external_path"]["stagnation_recovery"]

    assert recovery["enabled"] is True
    assert recovery["patience_steps"] > 0
    assert recovery["min_progress"] > 0.0
    assert recovery["recovery_steps"] > 0


def test_frontend_profile_rejects_bad_external_path_with_global_fallback():
    profile = _load_yaml("src/mppi_controller/configs/mujoco_rviz_goal.yaml")

    assert profile["external_path"]["min_forward_projection"] >= 2.0
    assert profile["global_path"]["enabled"] is True
    assert profile["global_path"]["lookahead"] >= profile["external_path"]["lookahead"]
    assert profile["global_path"]["obstacle_inflation"] > 0.0


def test_no_frontend_profile_defaults_to_cuda_controller():
    _assert_nominal_cuda_profile("src/mppi_controller/configs/mujoco_rviz_goal_no_frontend.yaml")


def test_navigation_entrypoints_default_to_cuda_controller():
    entrypoints = [
        "tools/geomapping_nav_trial.py",
        "src/ausim_geomapping_adapter/ausim_geomapping_adapter/pipeline.py",
        "src/traversability_mapping/launch/ausim_cube_mppi.launch.py",
        "src/mppi_controller/launch/mppi_closed_loop.launch.py",
    ]
    for relative_path in entrypoints:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "nominal_cuda" in text, relative_path
        assert "nominal_numpy" not in text, relative_path


def test_navigation_trial_uses_sparse_obstacle_override_by_default():
    obstacle_cfg = _load_yaml("src/mppi_controller/configs/obstacle_scout_sparse.yaml")
    script_text = (ROOT / "tools/geomapping_nav_trial.py").read_text(encoding="utf-8")

    assert obstacle_cfg["obstacle_count"] == 20
    assert obstacle_cfg["radius"] > 0.3
    assert obstacle_cfg["box_size"] > 0.3
    assert obstacle_cfg["range"]["x_max"] - obstacle_cfg["range"]["x_min"] > 20.0
    assert obstacle_cfg["range"]["y_max"] - obstacle_cfg["range"]["y_min"] > 10.0
    assert "obstacle_scout_sparse.yaml" in script_text
    assert "default=str(DEFAULT_OBSTACLE_CONFIG)" in script_text


def test_hfdm_profile_uses_no_frontend_mild_noise_jerk_defaults():
    profile = _load_yaml("src/mppi_controller/configs/mujoco_rviz_goal_hfdm_h25.yaml")
    mppi = profile["mppi"]
    controller = profile["controllers"][0]

    assert profile["default_controller"] == "learned_hfdm_h25"
    assert mppi["learned_risk_weight"] > 0.0
    assert mppi["terrain_risk_weight"] == 0.0
    assert profile["local_costmap"]["cost_weight"] == 0.0
    assert profile["external_path"]["enabled"] is False
    assert profile["global_path"]["enabled"] is False
    assert profile["final_controller"]["disable_when_local_costmap"] is True
    assert mppi["std_normal"] == [0.5, 0.19, 0.25]
    assert mppi["path_tracking_weight"] == 0.0
    assert mppi["path_tracking_tolerance"] == 0.2
    assert mppi["path_progress_weight"] == 0.0
    assert mppi["smooth_weight"] == 0.028
    assert mppi["accel_weight"] == 0.0
    assert mppi["lateral_weight"] == 0.04
    assert mppi["yaw_rate_weight"] == 0.04
    assert mppi["jerk_weight"] == 0.012
    assert mppi["update_smoothing_alpha"] == [0.0, 0.03, 0.04]
    assert profile["command_filter"]["enabled"] is True
    assert profile["command_filter"]["alpha"] == 0.02
    assert profile["command_filter"]["max_ax"] == 2.0
    assert profile["command_filter"]["max_ay"] == 1.0
    assert profile["command_filter"]["max_awz"] == 2.5
    assert profile["command_filter"]["lateral_deadband"] == 0.008
    assert profile["command_filter"]["yaw_deadband"] == 0.008
    assert controller["fdm"]["model_dir"] == "/home/mexxiie/prj/high_level_fdm/runs/geomapping_data1_h25/export"
