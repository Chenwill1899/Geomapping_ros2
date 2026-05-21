import math
from types import SimpleNamespace

import numpy as np

import mppi_controller.mujoco_closed_loop as closed_loop
from mppi_controller.mujoco_closed_loop import (
    CommandFilterConfig,
    MotionPolicyConfig,
    MotionPolicyState,
    MujocoClosedLoopNode,
    apply_motion_policy_to_command,
)
from mppi_controller.experiment import build_experiment_config


def test_mujoco_profile_preserves_motion_policy_after_build():
    config, _metadata = build_experiment_config("src/mppi_controller/configs/mujoco_rviz_goal.yaml")

    motion_policy = config["motion_policy"]
    assert motion_policy["allow_reverse"] is False
    assert motion_policy["rotate_then_translate"]["enabled"] is True
    assert motion_policy["rotate_then_translate"]["enter_angle_deg"] == 100.0
    assert motion_policy["rotate_then_translate"]["exit_angle_deg"] == 45.0


def test_rotate_then_translate_enters_when_goal_is_behind_robot():
    cfg = MotionPolicyConfig(
        allow_reverse=False,
        rotate_then_translate_enabled=True,
        enter_angle_rad=math.radians(100.0),
        exit_angle_rad=math.radians(45.0),
        min_distance=0.5,
        wz_gain=2.0,
        max_wz=1.2,
    )
    policy_state = MotionPolicyState()
    robot_state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.asarray([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw_command = np.asarray([0.6, 0.2, 0.0], dtype=np.float32)

    command = apply_motion_policy_to_command(robot_state, goal, raw_command, cfg, policy_state)

    assert policy_state.rotate_then_translate_active is True
    np.testing.assert_allclose(command[:2], [0.0, 0.0], atol=1e-6)
    assert math.isclose(abs(float(command[2])), 1.2, rel_tol=0.0, abs_tol=1e-6)


def test_rotate_then_translate_start_check_is_one_shot_per_goal():
    cfg = MotionPolicyConfig(
        allow_reverse=False,
        rotate_then_translate_enabled=True,
        enter_angle_rad=math.radians(100.0),
        exit_angle_rad=math.radians(45.0),
        min_distance=0.5,
        wz_gain=2.0,
        max_wz=1.2,
    )
    policy_state = MotionPolicyState()
    robot_state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    front_goal = np.asarray([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    behind_goal = np.asarray([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw_command = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)

    first_command = apply_motion_policy_to_command(robot_state, front_goal, raw_command, cfg, policy_state)
    second_command = apply_motion_policy_to_command(robot_state, behind_goal, raw_command, cfg, policy_state)

    assert policy_state.rotate_then_translate_active is False
    np.testing.assert_allclose(first_command, raw_command, atol=1e-6)
    np.testing.assert_allclose(second_command, raw_command, atol=1e-6)


def test_rotate_then_translate_does_not_reenter_after_initial_turn_releases():
    cfg = MotionPolicyConfig(
        allow_reverse=False,
        rotate_then_translate_enabled=True,
        enter_angle_rad=math.radians(100.0),
        exit_angle_rad=math.radians(45.0),
        min_distance=0.5,
        wz_gain=2.0,
        max_wz=1.2,
    )
    policy_state = MotionPolicyState(rotate_then_translate_active=True)
    goal = np.asarray([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw_command = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)

    released = apply_motion_policy_to_command(
        np.asarray([0.0, 0.0, math.pi, 0.0, 0.0, 0.0], dtype=np.float32),
        goal,
        raw_command,
        cfg,
        policy_state,
    )
    reentered = apply_motion_policy_to_command(
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        goal,
        raw_command,
        cfg,
        policy_state,
    )

    assert policy_state.rotate_then_translate_active is False
    np.testing.assert_allclose(released, raw_command, atol=1e-6)
    np.testing.assert_allclose(reentered, raw_command, atol=1e-6)


def test_closed_loop_motion_policy_uses_navigation_goal_not_local_planning_goal(monkeypatch):
    state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    navigation_goal = np.asarray([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    local_planning_goal = np.asarray([0.2, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    captured = {}

    def capture_motion_policy(policy_state, policy_goal, command, cfg, state_store):
        captured["goal"] = np.asarray(policy_goal, dtype=np.float32).copy()
        return np.asarray(command, dtype=np.float32)

    monkeypatch.setattr(closed_loop, "apply_reactive_obstacle_avoidance", lambda state, command, obstacles, config: command)
    monkeypatch.setattr(closed_loop, "apply_motion_policy_to_command", capture_motion_policy)
    monkeypatch.setattr(closed_loop, "filter_mujoco_command", lambda command, previous, cfg, dt, distance_to_goal=None: command)
    monkeypatch.setattr(closed_loop, "mujoco_twist_command", lambda command, drive_mode="differential", twist_factory=None: (object(), command))
    monkeypatch.setattr(closed_loop, "update_external_path_recovery_state", lambda *args, **kwargs: (False, False))

    node = SimpleNamespace(
        finished=False,
        last_state=state.copy(),
        last_odom_time=closed_loop.time.monotonic(),
        odom_timeout=1_000_000.0,
        config={
            "simulation": {
                "disable_goal_termination": True,
                "max_steps": 5,
                "minimum_distance": 0.5,
            },
            "goal_topic": {"required": False},
        },
        steps=0,
        max_steps=5,
        goal=navigation_goal,
        goal_termination_distance=0.5,
        elevation_map=SimpleNamespace(
            enabled=False,
            required=False,
            static_dedupe_distance=0.0,
            obstacles_for_state=lambda _state: np.empty((0, 7), dtype=np.float32),
        ),
        local_costmap=SimpleNamespace(
            cfg=SimpleNamespace(enabled=False, required=False),
            snapshot=lambda: {"enabled": False},
        ),
        external_path=SimpleNamespace(
            cfg=SimpleNamespace(enabled=False, stop_on_stale=False),
            has_fresh_path=lambda: False,
            path=lambda: np.empty((0, 2), dtype=np.float32),
        ),
        static_obstacles=np.empty((0, 7), dtype=np.float32),
        dynamic_obstacles=SimpleNamespace(
            cfg=SimpleNamespace(dedupe_distance=0.0),
            obstacles=lambda: np.empty((0, 7), dtype=np.float32),
        ),
        external_path_recovery=SimpleNamespace(),
        controller=SimpleNamespace(
            compute_control=lambda _state, _args: (
                np.asarray([0.6, 0.0, 0.0], dtype=np.float32),
                None,
                None,
                None,
                0.0,
            )
        ),
        motion_policy=MotionPolicyConfig(
            allow_reverse=False,
            rotate_then_translate_enabled=True,
            enter_angle_rad=math.radians(100.0),
            exit_angle_rad=math.radians(45.0),
        ),
        motion_policy_state=MotionPolicyState(),
        previous_command=np.zeros(3, dtype=np.float32),
        command_filter=SimpleNamespace(),
        command_dt=0.1,
        drive_mode="differential",
        publisher=SimpleNamespace(publish=lambda _message: None),
        _twist_type=lambda: object(),
        _planning_goal=lambda _state, _obstacles: (local_planning_goal, np.empty((0, 2), dtype=np.float32), None),
        _global_path_planning_goal=lambda _state, _obstacles: (
            local_planning_goal,
            np.empty((0, 2), dtype=np.float32),
            None,
        ),
        _publish_rviz_markers=lambda *args, **kwargs: None,
        terrain=SimpleNamespace(feature=lambda _x, _y: {}, risk_cost=lambda _x, _y, features=None: 0.0),
        recorder=SimpleNamespace(record_step=lambda **kwargs: None),
    )

    MujocoClosedLoopNode._on_timer(node)

    np.testing.assert_allclose(captured["goal"], navigation_goal)


def test_closed_loop_in_place_turn_does_not_blend_previous_linear_command(monkeypatch):
    state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    navigation_goal = np.asarray([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    captured = {}

    monkeypatch.setattr(closed_loop, "apply_reactive_obstacle_avoidance", lambda state, command, obstacles, config: command)
    monkeypatch.setattr(
        closed_loop,
        "mujoco_twist_command",
        lambda command, drive_mode="differential", twist_factory=None: (
            object(),
            captured.setdefault("command", np.asarray(command, dtype=np.float32).copy()),
        ),
    )
    monkeypatch.setattr(closed_loop, "update_external_path_recovery_state", lambda *args, **kwargs: (False, False))

    node = SimpleNamespace(
        finished=False,
        last_state=state.copy(),
        last_odom_time=closed_loop.time.monotonic(),
        odom_timeout=1_000_000.0,
        config={
            "simulation": {
                "disable_goal_termination": True,
                "max_steps": 5,
                "minimum_distance": 0.5,
            },
            "goal_topic": {"required": False},
        },
        steps=0,
        max_steps=5,
        goal=navigation_goal,
        goal_termination_distance=0.5,
        elevation_map=SimpleNamespace(
            enabled=False,
            required=False,
            static_dedupe_distance=0.0,
            obstacles_for_state=lambda _state: np.empty((0, 7), dtype=np.float32),
        ),
        local_costmap=SimpleNamespace(
            cfg=SimpleNamespace(enabled=False, required=False),
            snapshot=lambda: {"enabled": False},
        ),
        external_path=SimpleNamespace(
            cfg=SimpleNamespace(enabled=False, stop_on_stale=False),
            has_fresh_path=lambda: False,
            path=lambda: np.empty((0, 2), dtype=np.float32),
        ),
        static_obstacles=np.empty((0, 7), dtype=np.float32),
        dynamic_obstacles=SimpleNamespace(
            cfg=SimpleNamespace(dedupe_distance=0.0),
            obstacles=lambda: np.empty((0, 7), dtype=np.float32),
        ),
        external_path_recovery=SimpleNamespace(),
        controller=SimpleNamespace(
            compute_control=lambda _state, _args: (
                np.asarray([0.6, 0.0, 0.0], dtype=np.float32),
                None,
                None,
                None,
                0.0,
            )
        ),
        motion_policy=MotionPolicyConfig(
            allow_reverse=False,
            rotate_then_translate_enabled=True,
            enter_angle_rad=math.radians(100.0),
            exit_angle_rad=math.radians(45.0),
            wz_gain=2.0,
            max_wz=1.2,
        ),
        motion_policy_state=MotionPolicyState(),
        previous_command=np.asarray([0.9, 0.0, 0.0], dtype=np.float32),
        command_filter=CommandFilterConfig(enabled=True, alpha=0.8, max_ax=0.1, max_awz=1.2),
        command_dt=0.1,
        drive_mode="differential",
        publisher=SimpleNamespace(publish=lambda _message: None),
        _twist_type=lambda: object(),
        _planning_goal=lambda _state, _obstacles: (navigation_goal, np.empty((0, 2), dtype=np.float32), None),
        _global_path_planning_goal=lambda _state, _obstacles: (
            navigation_goal,
            np.empty((0, 2), dtype=np.float32),
            None,
        ),
        _publish_rviz_markers=lambda *args, **kwargs: None,
        terrain=SimpleNamespace(feature=lambda _x, _y: {}, risk_cost=lambda _x, _y, features=None: 0.0),
        recorder=SimpleNamespace(record_step=lambda **kwargs: None),
    )

    MujocoClosedLoopNode._on_timer(node)

    assert float(captured["command"][0]) == 0.0
    assert float(node.previous_command[0]) == 0.0
    assert abs(float(captured["command"][2])) > 0.0


def test_rotate_then_translate_latches_until_heading_is_inside_exit_angle():
    cfg = MotionPolicyConfig(
        allow_reverse=False,
        rotate_then_translate_enabled=True,
        enter_angle_rad=math.radians(100.0),
        exit_angle_rad=math.radians(45.0),
        min_distance=0.5,
        wz_gain=2.0,
        max_wz=1.2,
    )
    policy_state = MotionPolicyState(rotate_then_translate_active=True)
    goal = np.asarray([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw_command = np.asarray([0.6, 0.0, 0.0], dtype=np.float32)

    still_turning = apply_motion_policy_to_command(
        np.asarray([0.0, 0.0, math.radians(130.0), 0.0, 0.0, 0.0], dtype=np.float32),
        goal,
        raw_command,
        cfg,
        policy_state,
    )
    assert policy_state.rotate_then_translate_active is True
    np.testing.assert_allclose(still_turning[:2], [0.0, 0.0], atol=1e-6)
    assert abs(float(still_turning[2])) > 0.0

    released = apply_motion_policy_to_command(
        np.asarray([0.0, 0.0, math.radians(170.0), 0.0, 0.0, 0.0], dtype=np.float32),
        goal,
        raw_command,
        cfg,
        policy_state,
    )
    assert policy_state.rotate_then_translate_active is False
    np.testing.assert_allclose(released, raw_command, atol=1e-6)


def test_motion_policy_clamps_reverse_when_reverse_is_disabled():
    cfg = MotionPolicyConfig(
        allow_reverse=False,
        rotate_then_translate_enabled=False,
        enter_angle_rad=math.radians(100.0),
        exit_angle_rad=math.radians(45.0),
        min_distance=0.5,
        wz_gain=2.0,
        max_wz=1.2,
    )
    policy_state = MotionPolicyState()
    robot_state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.asarray([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw_command = np.asarray([-0.4, 0.1, -0.2], dtype=np.float32)

    command = apply_motion_policy_to_command(robot_state, goal, raw_command, cfg, policy_state)

    np.testing.assert_allclose(command, [0.0, 0.1, -0.2], atol=1e-6)
