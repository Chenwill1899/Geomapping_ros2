import numpy as np

from mppi_controller.mujoco_closed_loop import (
    ExternalPathConfig,
    ExternalPathRecoveryState,
    GlobalPathConfig,
    MujocoClosedLoopNode,
    select_external_path_goal,
    update_external_path_recovery_state,
)


def test_short_external_path_behind_robot_falls_back_to_global_goal():
    state = np.asarray([10.0, 2.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    global_goal = np.asarray([18.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    path = np.asarray([[9.9, 2.28], [9.64, 2.26]], dtype=np.float32)
    cfg = ExternalPathConfig(
        enabled=True,
        lookahead=3.0,
        min_remaining_length=1.5,
        min_goal_distance=1.0,
        final_goal_bypass_distance=2.0,
    )

    planning_goal, active_path = select_external_path_goal(state, global_goal, path, cfg)

    np.testing.assert_allclose(planning_goal[:2], global_goal[:2])
    assert active_path.size == 0


def test_valid_external_path_still_uses_lookahead_goal():
    state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    global_goal = np.asarray([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    path = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [8.0, 0.0]], dtype=np.float32)
    cfg = ExternalPathConfig(
        enabled=True,
        lookahead=3.0,
        min_remaining_length=1.5,
        min_goal_distance=1.0,
        final_goal_bypass_distance=2.0,
    )

    planning_goal, active_path = select_external_path_goal(state, global_goal, path, cfg)

    np.testing.assert_allclose(planning_goal[:2], np.asarray([3.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(active_path, path)


def test_external_path_goal_behind_global_progress_falls_back_to_global_goal():
    state = np.asarray([16.0, 4.6, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    global_goal = np.asarray([18.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    path = np.asarray([[16.0, 4.6], [15.2, 3.8]], dtype=np.float32)
    cfg = ExternalPathConfig(
        enabled=True,
        lookahead=3.0,
        min_remaining_length=1.5,
        min_goal_distance=1.0,
        final_goal_bypass_distance=2.0,
        min_forward_projection=0.0,
    )

    planning_goal, active_path = select_external_path_goal(state, global_goal, path, cfg)

    np.testing.assert_allclose(planning_goal[:2], global_goal[:2])
    assert active_path.size == 0


def test_external_path_config_parses_stagnation_recovery():
    cfg = ExternalPathConfig.from_config(
        {
            "external_path": {
                "enabled": True,
                "stagnation_recovery": {
                    "enabled": True,
                    "patience_steps": 12,
                    "min_progress": 0.3,
                    "recovery_steps": 8,
                },
            }
        }
    )

    assert cfg.stagnation_recovery_enabled is True
    assert cfg.stagnation_patience_steps == 12
    assert cfg.stagnation_min_progress == 0.3
    assert cfg.stagnation_recovery_steps == 8


def test_external_path_recovery_triggers_after_stagnation():
    cfg = ExternalPathConfig(
        enabled=True,
        stagnation_recovery_enabled=True,
        stagnation_patience_steps=2,
        stagnation_min_progress=0.2,
        stagnation_recovery_steps=3,
    )
    state = ExternalPathRecoveryState()

    bypass, triggered = update_external_path_recovery_state(
        state,
        cfg,
        distance_to_goal=10.0,
        has_fresh_external_path=True,
    )
    assert (bypass, triggered) == (False, False)

    bypass, triggered = update_external_path_recovery_state(
        state,
        cfg,
        distance_to_goal=9.95,
        has_fresh_external_path=True,
    )
    assert (bypass, triggered) == (False, False)

    bypass, triggered = update_external_path_recovery_state(
        state,
        cfg,
        distance_to_goal=9.90,
        has_fresh_external_path=True,
    )
    assert (bypass, triggered) == (True, True)

    bypass, triggered = update_external_path_recovery_state(
        state,
        cfg,
        distance_to_goal=9.88,
        has_fresh_external_path=True,
    )
    assert (bypass, triggered) == (True, False)


def test_global_path_fallback_avoids_boundary_detour_when_external_path_is_bad():
    node = object.__new__(MujocoClosedLoopNode)
    node.goal = np.asarray([18.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    node.global_path = GlobalPathConfig(
        enabled=True,
        resolution=0.2,
        padding=2.0,
        lookahead=3.0,
        obstacle_inflation=0.15,
        simplify_stride=3,
        smoothing_iterations=1,
        smoothing_alpha=0.15,
    )
    node.config = {"robot": {"radius": 0.55, "safety_dist": 0.20}}
    node.steps = 0
    node.path_waypoints = np.empty((0, 2), dtype=np.float32)
    node.path_replan_step = -100
    node.path_replan_state = np.asarray([np.inf, np.inf], dtype=np.float32)
    node.path_plan_id = None

    class Recorder:
        def record_global_path(self, *, step, path):
            self.step = step
            self.path = path
            return 7

    node.recorder = Recorder()
    obstacles = np.asarray(
        [
            [0.90, 0.10, 0.34, 0.0, 0.0, 0.0, 0.0],
            [12.88, 2.73, 0.34, 0.0, 0.0, 0.0, 0.0],
            [13.55, 3.61, 0.34, 0.0, 0.0, 0.0, 0.0],
            [13.83, 4.85, 0.34, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    planning_goal, active_path, plan_id = node._global_path_planning_goal(
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        obstacles,
    )

    assert plan_id == 7
    assert len(active_path) > 2
    assert float(np.min(node.path_waypoints[:, 1])) > -1.0
    assert planning_goal[0] > 0.5
    assert planning_goal[1] > 0.1
