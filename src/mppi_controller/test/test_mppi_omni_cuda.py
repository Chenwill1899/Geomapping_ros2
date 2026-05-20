import numpy as np
import pytest


def _cuda_available() -> bool:
    try:
        import pycuda.driver as cuda

        cuda.init()
        return cuda.Device.count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="CUDA device is not available")


def test_cuda_controller_reads_progress_heading_and_forward_limits():
    from mppi_controller.experiment import build_experiment_config
    from mppi_controller.controllers.mppi_omni_cuda import MppiOmniCuda

    config, _metadata = build_experiment_config("src/mppi_controller/configs/mujoco_rviz_goal.yaml")

    controller = MppiOmniCuda.from_config(config, seed=123)

    assert controller.goal_progress_weight == pytest.approx(44.0)
    assert controller.heading_to_goal_weight == pytest.approx(0.20)
    assert controller.heading_to_goal_min_distance == pytest.approx(0.6)
    assert controller.path_tracking_weight == pytest.approx(3.0)
    assert controller.path_progress_weight == pytest.approx(0.8)
    assert controller.obstacle_weight == pytest.approx(200.0)
    assert controller.obstacle_collision_weight == pytest.approx(10000.0)
    assert controller.obstacle_soft_weight == pytest.approx(16.0)
    assert controller.obstacle_influence_dist == pytest.approx(1.00)
    assert controller.safety_dist == pytest.approx(0.20)
    np.testing.assert_allclose(controller.update_smoothing_alpha, [0.34, 0.84, 0.82])
    assert controller.min_control[0] == pytest.approx(0.0)


def test_cuda_goal_progress_cost_prefers_forward_motion_when_goal_is_far():
    from mppi_controller.controllers.mppi_omni_cuda import MppiOmniCuda

    controller = MppiOmniCuda(
        dt=0.1,
        horizon_steps=10,
        num_samples=2,
        lambda_=1.0,
        noise_std=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        max_vx=1.0,
        max_vy=0.0,
        max_wz=0.0,
        min_vx=0.0,
        goal_xy_weight=0.0,
        yaw_weight=0.0,
        control_weight=0.0,
        smooth_weight=0.0,
        accel_weight=0.0,
        jerk_weight=0.0,
        goal_progress_weight=10.0,
        heading_to_goal_weight=0.0,
        robot_radius=0.0,
        safety_dist=0.0,
        draw_num_traj=2,
        seed=123,
    )
    initial_state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.asarray([10.0, 0.0, 0.0], dtype=np.float32)
    controls = np.zeros((2, 10, 3), dtype=np.float32)
    controls[1, :, 0] = 1.0

    costs = controller.trajectory_cost_batch(
        initial_state,
        controls,
        goal,
        np.zeros((0, 7), dtype=np.float32),
    )

    assert costs[1] < costs[0]


def test_cuda_controller_resets_nominal_controls_on_goal_change():
    from mppi_controller.controllers.mppi_omni_cuda import MppiOmniCuda

    controller = MppiOmniCuda(
        dt=0.1,
        horizon_steps=3,
        num_samples=2,
        lambda_=1.0,
        noise_std=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        max_vx=1.0,
        max_vy=0.0,
        max_wz=0.0,
        min_vx=0.0,
        goal_change_reset_distance=0.75,
        goal_change_reset_yaw=1.0,
        robot_radius=0.0,
        safety_dist=0.0,
        draw_num_traj=2,
        seed=123,
    )
    controller.nominal_u.fill(0.5)

    controller._reset_nominal_on_goal_change(np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.all(controller.nominal_u == pytest.approx(0.5))

    controller._reset_nominal_on_goal_change(np.asarray([2.0, 0.0, 0.0], dtype=np.float32))

    np.testing.assert_allclose(controller.nominal_u, 0.0)


def test_cuda_controller_smooths_nominal_updates_by_axis():
    from mppi_controller.controllers.mppi_omni_cuda import MppiOmniCuda

    controller = MppiOmniCuda(
        dt=0.1,
        horizon_steps=2,
        num_samples=2,
        lambda_=1.0,
        noise_std=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        max_vx=1.0,
        max_vy=1.0,
        max_wz=1.0,
        min_vx=0.0,
        update_smoothing_alpha=[0.5, 0.0, 0.25],
        robot_radius=0.0,
        safety_dist=0.0,
        draw_num_traj=2,
        seed=123,
    )
    previous = np.zeros((2, 3), dtype=np.float32)
    updated = np.ones((2, 3), dtype=np.float32)

    first = controller._smooth_nominal_update(updated, previous)
    second = controller._smooth_nominal_update(updated, previous)

    np.testing.assert_allclose(first, updated)
    np.testing.assert_allclose(second, [[0.5, 1.0, 0.75], [0.5, 1.0, 0.75]])


def test_cuda_path_tracking_cost_prefers_rollouts_near_active_path():
    from mppi_controller.controllers.mppi_omni_cuda import MppiOmniCuda

    controller = MppiOmniCuda(
        dt=0.1,
        horizon_steps=10,
        num_samples=2,
        lambda_=1.0,
        noise_std=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        max_vx=1.0,
        max_vy=1.0,
        max_wz=0.0,
        min_vx=0.0,
        goal_xy_weight=0.0,
        yaw_weight=0.0,
        control_weight=0.0,
        smooth_weight=0.0,
        accel_weight=0.0,
        jerk_weight=0.0,
        path_tracking_weight=10.0,
        path_tracking_tolerance=0.05,
        robot_radius=0.0,
        safety_dist=0.0,
        draw_num_traj=2,
        seed=123,
    )
    initial_state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
    path = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    controls = np.zeros((2, 10, 3), dtype=np.float32)
    controls[:, :, 0] = 1.0
    controls[1, :, 1] = 1.0

    costs = controller.trajectory_cost_batch(
        initial_state,
        controls,
        goal,
        np.zeros((0, 7), dtype=np.float32),
        path=path,
    )

    assert costs[0] < costs[1]


def test_cuda_local_costmap_cost_penalizes_high_cost_cells():
    from mppi_controller.controllers.mppi_omni_cuda import MppiOmniCuda

    controller = MppiOmniCuda(
        dt=0.1,
        horizon_steps=10,
        num_samples=2,
        lambda_=1.0,
        noise_std=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        max_vx=1.0,
        max_vy=1.0,
        max_wz=0.0,
        min_vx=0.0,
        goal_xy_weight=0.0,
        yaw_weight=0.0,
        control_weight=0.0,
        smooth_weight=0.0,
        accel_weight=0.0,
        jerk_weight=0.0,
        robot_radius=0.0,
        safety_dist=0.0,
        draw_num_traj=2,
        seed=123,
    )
    data = np.zeros((12, 12), dtype=np.float32)
    data[0, 1] = 100.0
    costmap = {
        "enabled": True,
        "origin": np.asarray([0.0, -1.0], dtype=np.float32),
        "resolution": 0.1,
        "width": 12,
        "height": 12,
        "data": data.reshape(-1),
        "unknown_mask": np.zeros(data.size, dtype=bool),
        "weight": 20.0,
        "power": 2.0,
        "unknown_cost": 100.0,
        "max_cost": 100.0,
        "unknown_clear_radius": 0.0,
        "unknown_clear_value": 0.0,
        "footprint_enabled": False,
        "footprint_radius": 0.0,
        "footprint_safety_margin": 0.0,
        "footprint_sample_count": 8,
    }
    initial_state = np.asarray([0.0, -0.95, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.asarray([1.0, -0.95, 0.0], dtype=np.float32)
    controls = np.zeros((2, 10, 3), dtype=np.float32)
    controls[:, :, 0] = 1.0
    controls[1, :, 1] = 1.0

    costs = controller.trajectory_cost_batch(
        initial_state,
        controls,
        goal,
        np.zeros((0, 7), dtype=np.float32),
        costmap=costmap,
    )

    assert costs[1] < costs[0]


def test_cuda_collision_weight_adds_strong_overlap_penalty():
    from mppi_controller.controllers.mppi_omni_cuda import MppiOmniCuda

    controller = MppiOmniCuda(
        dt=0.1,
        horizon_steps=10,
        num_samples=2,
        lambda_=1.0,
        noise_std=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        max_vx=1.0,
        max_vy=1.0,
        max_wz=0.0,
        min_vx=0.0,
        goal_xy_weight=0.0,
        yaw_weight=0.0,
        control_weight=0.0,
        smooth_weight=0.0,
        accel_weight=0.0,
        jerk_weight=0.0,
        obstacle_weight=0.0,
        obstacle_collision_weight=1200.0,
        robot_radius=0.55,
        safety_dist=0.0,
        draw_num_traj=2,
        seed=123,
    )
    initial_state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.asarray([[0.6, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    controls = np.zeros((2, 10, 3), dtype=np.float32)
    controls[:, :, 0] = 1.0
    controls[1, :, 1] = 1.0

    costs = controller.trajectory_cost_batch(initial_state, controls, goal, obstacles)

    assert costs[0] > costs[1]
