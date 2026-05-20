import numpy as np


def test_reactive_avoidance_reduces_velocity_toward_close_obstacle():
    from mppi_controller.mujoco_closed_loop import apply_reactive_obstacle_avoidance

    config = {
        "mujoco": {"drive_mode": "omni_freejoint"},
        "robot": {"radius": 0.55, "safety_dist": 0.15, "max_vx": 1.0, "max_vy": 0.3, "max_wz": 1.2},
        "reactive_avoidance": {
            "enabled": True,
            "influence_dist": 0.8,
            "brake_gain": 0.8,
            "lateral_gain": 0.0,
            "max_lateral_adjust": 0.25,
            "min_speed_scale": 0.1,
        },
    }
    state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    command = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.asarray([[1.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)

    adjusted = apply_reactive_obstacle_avoidance(state, command, obstacles, config)

    assert adjusted[0] < command[0]
    assert adjusted[1] == 0.0


def test_reactive_avoidance_adds_lateral_away_from_offset_obstacle():
    from mppi_controller.mujoco_closed_loop import apply_reactive_obstacle_avoidance

    config = {
        "mujoco": {"drive_mode": "omni_freejoint"},
        "robot": {"radius": 0.55, "safety_dist": 0.15, "max_vx": 1.0, "max_vy": 0.3, "max_wz": 1.2},
        "reactive_avoidance": {
            "enabled": True,
            "influence_dist": 0.8,
            "brake_gain": 0.0,
            "lateral_gain": 0.4,
            "max_lateral_adjust": 0.25,
            "min_speed_scale": 0.1,
        },
    }
    state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    command = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)
    obstacles = np.asarray([[1.0, 0.2, 0.3, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)

    adjusted = apply_reactive_obstacle_avoidance(state, command, obstacles, config)

    assert adjusted[1] < 0.0


def test_reactive_avoidance_can_ignore_configured_safety_band():
    from mppi_controller.mujoco_closed_loop import apply_reactive_obstacle_avoidance

    config = {
        "mujoco": {"drive_mode": "omni_freejoint"},
        "robot": {"radius": 0.55, "safety_dist": 0.20, "max_vx": 1.0, "max_vy": 0.3, "max_wz": 1.2},
        "reactive_avoidance": {
            "enabled": True,
            "include_safety_dist": False,
            "influence_dist": 0.08,
            "brake_gain": 0.5,
            "lateral_gain": 0.0,
            "max_lateral_adjust": 0.05,
            "min_speed_scale": 0.1,
        },
    }
    state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    command = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    outside_geometry_band = np.asarray([[0.95, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)
    inside_geometry_band = np.asarray([[0.90, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)

    unchanged = apply_reactive_obstacle_avoidance(state, command, outside_geometry_band, config)
    adjusted = apply_reactive_obstacle_avoidance(state, command, inside_geometry_band, config)

    np.testing.assert_allclose(unchanged, command)
    assert adjusted[0] < command[0]
