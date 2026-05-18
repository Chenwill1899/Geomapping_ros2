import numpy as np

from mppi_controller.mujoco_closed_loop import ExternalPathConfig, select_external_path_goal


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
