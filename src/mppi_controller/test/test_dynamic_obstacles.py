import numpy as np


class _Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class _Pose:
    def __init__(self, x=0.0, y=0.0):
        self.position = _Point(x, y)


class _Vector:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class _Box:
    def __init__(self, x, y, sx, sy):
        self.center = _Pose(x, y)
        self.size = _Vector(sx, sy)


class _BoxArray:
    def __init__(self, boxes):
        self.boxes = boxes


def test_bounding_box_array_to_obstacles_uses_conservative_xy_radius():
    from mppi_controller.mujoco_closed_loop import bounding_box_array_to_obstacles

    message = _BoxArray([_Box(2.0, 3.0, 0.6, 0.8)])

    obstacles = bounding_box_array_to_obstacles(message, safety_padding=0.1)

    assert obstacles.shape == (1, 7)
    np.testing.assert_allclose(obstacles[0, :2], [2.0, 3.0])
    assert obstacles[0, 2] == np.float32(0.6)
    assert obstacles[0, 3] == np.float32(0.6)


def test_bounding_box_array_to_obstacles_can_pad_compact_boxes_more():
    from mppi_controller.mujoco_closed_loop import bounding_box_array_to_obstacles

    message = _BoxArray([_Box(0.0, 0.0, 0.3, 0.3), _Box(1.0, 0.0, 0.6, 0.6)])

    obstacles = bounding_box_array_to_obstacles(message, safety_padding=0.05, box_safety_padding=0.06)

    assert obstacles[0, 2] == np.float32(0.5 * np.hypot(0.3, 0.3) + 0.06)
    assert obstacles[1, 2] == np.float32(0.5 * np.hypot(0.6, 0.6) + 0.05)


def test_experiment_config_keeps_dynamic_obstacle_section():
    from mppi_controller.experiment import build_experiment_config

    config, _metadata = build_experiment_config("src/mppi_controller/configs/mujoco_rviz_goal.yaml")

    assert config["dynamic_obstacles"]["enabled"] is True
    assert config["dynamic_obstacles"]["topic"] == "/dyn_obstacle"
