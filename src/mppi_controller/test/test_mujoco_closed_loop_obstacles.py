import numpy as np


def test_tracking_obstacles_are_kept_with_costmap_or_external_path():
    from mppi_controller.mujoco_closed_loop import tracking_obstacles_for_cost

    obstacles = np.asarray([[1.0, 2.0, 0.4, 0.0, 0.0, 0.0, 0.16]], dtype=np.float32)

    with_costmap = tracking_obstacles_for_cost(
        obstacles,
        using_local_costmap=True,
        has_fresh_external_path=False,
    )
    with_external_path = tracking_obstacles_for_cost(
        obstacles,
        using_local_costmap=False,
        has_fresh_external_path=True,
    )

    np.testing.assert_allclose(with_costmap, obstacles)
    np.testing.assert_allclose(with_external_path, obstacles)


def test_rollout_preview_uses_command_filter_limits():
    from mppi_controller.mujoco_closed_loop import CommandFilterConfig, filter_control_sequence_for_rollout

    cfg = CommandFilterConfig(
        enabled=True,
        alpha=0.0,
        max_ax=1.0,
        max_ay=0.5,
        max_awz=2.0,
        drive_mode="omni_freejoint",
    )
    controls = np.asarray(
        [
            [1.0, 0.5, 1.0],
            [1.0, 0.5, 1.0],
        ],
        dtype=np.float32,
    )

    filtered = filter_control_sequence_for_rollout(
        controls,
        np.zeros(3, dtype=np.float32),
        cfg,
        0.1,
    )

    np.testing.assert_allclose(filtered[0], [0.1, 0.05, 0.2], atol=1e-6)
    np.testing.assert_allclose(filtered[1], [0.2, 0.1, 0.4], atol=1e-6)
