import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "tools" / "geomapping_dataset_collect.py"


def load_collector_module():
    spec = importlib.util.spec_from_file_location("geomapping_dataset_collect", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_goal_yaml_parses_required_fields_in_order(tmp_path):
    collect = load_collector_module()
    goals_path = tmp_path / "goals.yaml"
    goals_path.write_text(
        yaml.safe_dump(
            {
                "goals": [
                    {"id": "near_gate", "x": 1.5, "y": 2.0, "yaw": 0.0},
                    {"id": "far_gate", "x": 4, "y": -1, "yaw": 1.57},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    goals = collect.load_goals(goals_path)

    assert [goal.id for goal in goals] == ["near_gate", "far_gate"]
    assert [(goal.x, goal.y, goal.yaw) for goal in goals] == [(1.5, 2.0, 0.0), (4.0, -1.0, 1.57)]


def test_goal_yaml_rejects_missing_fields(tmp_path):
    collect = load_collector_module()
    goals_path = tmp_path / "goals.yaml"
    goals_path.write_text(yaml.safe_dump({"goals": [{"id": "bad", "x": 1.0, "y": 2.0}]}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"goals\[0\]\.yaw"):
        collect.load_goals(goals_path)


def test_topic_buffer_slices_episode_windows_without_crossing_boundaries():
    collect = load_collector_module()
    buffer = collect.TopicBuffer()
    buffer.add("odom", collect.RecordSample(stamp=9.99, payload={"x": -1}))
    buffer.add("odom", collect.RecordSample(stamp=10.0, payload={"x": 0}))
    buffer.add("odom", collect.RecordSample(stamp=10.5, payload={"x": 1}))
    buffer.add("odom", collect.RecordSample(stamp=11.0, payload={"x": 2}))

    first = buffer.slice("odom", start=10.0, end=11.0)
    second = buffer.slice("odom", start=11.0, end=12.0)

    assert [sample.payload["x"] for sample in first] == [0, 1]
    assert [sample.payload["x"] for sample in second] == [2]


def test_local_costmap_throttle_accepts_at_two_hz_boundaries():
    collect = load_collector_module()
    throttle = collect.SampleThrottle(rate_hz=2.0)

    decisions = [throttle.accept(stamp) for stamp in [0.0, 0.20, 0.50, 0.99, 1.00]]

    assert decisions == [True, False, True, False, True]


def test_controller_max_steps_uses_profile_sampling_rate(tmp_path):
    collect = load_collector_module()
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({"simulation": {"sampling_rate": 4.0}}), encoding="utf-8")

    assert collect.controller_max_steps(profile, controller="nominal_cuda", timeout_s=3.2) == 13


def test_controller_max_steps_scales_for_seed_level_multi_goal_run(tmp_path):
    collect = load_collector_module()
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({"simulation": {"sampling_rate": 10.0}}), encoding="utf-8")

    assert collect.controller_max_steps(profile, controller="nominal_cuda", timeout_s=180.0, episodes=10) == 18000


def test_selected_goals_for_config_honors_max_goals_per_seed(tmp_path):
    collect = load_collector_module()
    goals_path = tmp_path / "goals.yaml"
    goals_path.write_text(
        yaml.safe_dump(
            {
                "goals": [
                    {"id": "g0", "x": 1.0, "y": 0.0, "yaw": 0.0},
                    {"id": "g1", "x": 2.0, "y": 0.0, "yaw": 0.0},
                    {"id": "g2", "x": 3.0, "y": 0.0, "yaw": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    goals = collect.selected_goals_for_config(
        collect.CollectorConfig(
            goals_path=goals_path,
            seeds=[17],
            output_dir=tmp_path / "dataset_run",
            max_goals_per_seed=2,
        )
    )

    assert [goal.id for goal in goals] == ["g0", "g1"]


def test_controller_command_uses_longer_internal_odom_timeout(tmp_path):
    collect = load_collector_module()
    command = collect.controller_command(
        profile=Path("/tmp/profile.yaml"),
        controller="nominal_cuda",
        native_run_dir=tmp_path / "native_run",
        max_steps=123,
        odom_timeout_s=30.0,
    )

    assert "--max-steps 123" in command
    assert "--odom-timeout 30.0" in command


def test_native_summary_reached_goal_marks_episode_success(tmp_path):
    collect = load_collector_module()
    native_run_dir = tmp_path / "native_run"
    native_run_dir.mkdir()
    (native_run_dir / "summary.json").write_text(
        json.dumps({"reached_goal": True, "final_distance": 0.39}),
        encoding="utf-8",
    )

    assert collect.native_goal_reached(native_run_dir) is True


def test_native_summary_ignores_stale_previous_episode(tmp_path):
    collect = load_collector_module()
    native_run_dir = tmp_path / "native_run"
    native_run_dir.mkdir()
    summary_path = native_run_dir / "summary.json"
    summary_path.write_text(json.dumps({"reached_goal": True}), encoding="utf-8")
    os.utime(summary_path, (100.0, 100.0))

    assert collect.native_goal_reached(native_run_dir, min_mtime=101.0) is False


def test_write_local_costmap_npz_preserves_required_layers(tmp_path):
    collect = load_collector_module()
    sample = collect.CostmapRecord(
        stamp=3.0,
        origin=(1.0, 2.0),
        resolution=0.25,
        width=2,
        height=2,
        reward_cost=[1.0, 2.0, 3.0, 4.0],
        height_layer=[0.1, 0.2, 0.3, 0.4],
        roughness=[0.0, 0.1, 0.2, 0.3],
        cost_map=[4.0, 3.0, 2.0, 1.0],
    )

    collect.write_local_costmap_npz([sample], tmp_path / "local_costmap.npz")

    data = np.load(tmp_path / "local_costmap.npz")
    np.testing.assert_allclose(data["stamp"], np.asarray([3.0]))
    np.testing.assert_allclose(data["origin"], np.asarray([[1.0, 2.0]]))
    assert data["resolution"].tolist() == [0.25]
    assert data["width"].tolist() == [2]
    assert data["height"].tolist() == [2]
    np.testing.assert_allclose(data["reward_cost"], np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(data["height_layer"], np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))
    np.testing.assert_allclose(data["roughness"], np.asarray([[0.0, 0.1, 0.2, 0.3]], dtype=np.float32))
    np.testing.assert_allclose(data["cost_map"], np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32))


def test_write_episode_artifacts_saves_trajectory_and_obstacle_plots(tmp_path):
    collect = load_collector_module()
    goal = collect.Goal(id="plot_goal", x=2.0, y=1.0, yaw=0.0)

    episode = collect.write_episode_artifacts(
        tmp_path,
        seed=17,
        episode_index=0,
        goal=goal,
        start_time=10.0,
        end_time=12.0,
        status="success",
        failure_reason=None,
        odom_samples=[
            collect.OdomRecord(stamp=10.0, x=0.0, y=0.0, yaw=0.0, vx=0.2, vy=0.0, wz=0.0),
            collect.OdomRecord(stamp=11.0, x=1.0, y=0.4, yaw=0.1, vx=0.2, vy=0.0, wz=0.0),
            collect.OdomRecord(stamp=12.0, x=2.0, y=1.0, yaw=0.2, vx=0.0, vy=0.0, wz=0.0),
        ],
        cmd_samples=[],
        frontend_path=[],
        tltrajectory=[],
        dynamic_obstacles=[
            collect.JsonRecord(
                stamp=10.5,
                payload={
                    "obstacles": [
                        {"x": 0.8, "y": 0.6, "radius": 0.25},
                        {"x": 1.4, "y": 0.8, "radius": 0.35},
                    ]
                },
            )
        ],
        local_costmaps=[],
    )

    assert (tmp_path / "trajectory.png").stat().st_size > 0
    assert (tmp_path / "obstacles.png").stat().st_size > 0
    assert episode["artifacts"]["trajectory_plot"] == "trajectory.png"
    assert episode["artifacts"]["obstacle_plot"] == "obstacles.png"


def test_failure_classifier_covers_default_failure_modes():
    collect = load_collector_module()
    thresholds = collect.FailureThresholds(
        goal_timeout_s=180.0,
        odom_timeout_s=2.0,
        no_progress_window_s=20.0,
        no_progress_min_delta_m=0.2,
        low_speed_window_s=10.0,
        low_speed_threshold_mps=0.03,
        goal_tolerance_m=0.3,
    )
    goal = collect.Goal(id="g0", x=10.0, y=0.0, yaw=0.0)

    assert collect.classify_failure(
        now=5.0,
        episode_start=0.0,
        goal=goal,
        odom_samples=[collect.OdomRecord(stamp=4.9, x=0.0, y=0.0, yaw=0.0, vx=0.1, vy=0.0, wz=0.0)],
        process_exits=[collect.ProcessExit(name="mppi", returncode=7)],
        thresholds=thresholds,
    ) == collect.Failure(reason="process_exit", detail="mppi exited with code 7")

    assert collect.classify_failure(
        now=5.0,
        episode_start=0.0,
        goal=goal,
        odom_samples=[collect.OdomRecord(stamp=2.9, x=0.0, y=0.0, yaw=0.0, vx=0.1, vy=0.0, wz=0.0)],
        process_exits=[],
        thresholds=thresholds,
    ) == collect.Failure(reason="odom_timeout", detail="last odom age 2.100s")

    assert collect.classify_failure(
        now=181.0,
        episode_start=0.0,
        goal=goal,
        odom_samples=[collect.OdomRecord(stamp=180.9, x=1.0, y=0.0, yaw=0.0, vx=0.1, vy=0.0, wz=0.0)],
        process_exits=[],
        thresholds=thresholds,
    ) == collect.Failure(reason="timeout", detail="goal timeout 180.000s")

    stagnant = [
        collect.OdomRecord(stamp=0.0, x=0.0, y=0.0, yaw=0.0, vx=0.2, vy=0.0, wz=0.0),
        collect.OdomRecord(stamp=10.0, x=0.05, y=0.0, yaw=0.0, vx=0.2, vy=0.0, wz=0.0),
        collect.OdomRecord(stamp=21.0, x=0.10, y=0.0, yaw=0.0, vx=0.2, vy=0.0, wz=0.0),
    ]
    assert collect.classify_failure(
        now=21.0,
        episode_start=0.0,
        goal=goal,
        odom_samples=stagnant,
        process_exits=[],
        thresholds=thresholds,
    ) == collect.Failure(reason="no_progress", detail="progress 0.100m in 20.000s")

    slow = [
        collect.OdomRecord(stamp=11.0, x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0.0, wz=0.0),
        collect.OdomRecord(stamp=16.0, x=0.4, y=0.0, yaw=0.0, vx=0.01, vy=0.0, wz=0.0),
        collect.OdomRecord(stamp=21.0, x=0.8, y=0.0, yaw=0.0, vx=0.01, vy=0.0, wz=0.0),
    ]
    assert collect.classify_failure(
        now=21.0,
        episode_start=0.0,
        goal=goal,
        odom_samples=slow,
        process_exits=[],
        thresholds=thresholds,
    ) == collect.Failure(reason="low_speed_stall", detail="average speed 0.010m/s in 10.000s")


def test_failure_classifier_ignores_missing_tltrajectory_when_frontend_path_is_active():
    collect = load_collector_module()
    thresholds = collect.FailureThresholds(
        goal_timeout_s=180.0,
        odom_timeout_s=2.0,
        no_progress_window_s=60.0,
        low_speed_window_s=60.0,
        tltrajectory_timeout_s=3.0,
        goal_tolerance_m=0.3,
    )
    goal = collect.Goal(id="g0", x=10.0, y=0.0, yaw=0.0)

    assert collect.classify_failure(
        now=8.0,
        episode_start=0.0,
        goal=goal,
        odom_samples=[
            collect.OdomRecord(stamp=7.9, x=2.0, y=0.0, yaw=0.0, vx=0.2, vy=0.0, wz=0.0),
        ],
        process_exits=[],
        thresholds=thresholds,
        frontend_path_samples=[
            collect.JsonRecord(stamp=1.0, payload={"point_count": 3}),
            collect.JsonRecord(stamp=7.5, payload={"point_count": 12}),
        ],
        tltrajectory_samples=[],
    ) is None


def test_fake_runtime_smoke_writes_seed_episode_tree_and_manifest(tmp_path):
    collect = load_collector_module()
    goals_path = tmp_path / "goals.yaml"
    goals_path.write_text(
        yaml.safe_dump(
            {
                "goals": [
                    {"id": "g0", "x": 1.0, "y": 0.0, "yaw": 0.0},
                    {"id": "g1", "x": 2.0, "y": 0.0, "yaw": 0.0},
                    {"id": "g2", "x": 3.0, "y": 0.0, "yaw": 0.0},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = collect.CollectorConfig(
        goals_path=goals_path,
        seeds=[17, 101],
        output_dir=tmp_path / "dataset_run",
        max_goals_per_seed=None,
        thresholds=collect.FailureThresholds(),
    )
    runtime = FakeRuntime(
        {
            17: [
                collect.EpisodeResult(status="success", failure_reason=None),
                collect.EpisodeResult(status="failed", failure_reason="no_progress"),
                collect.EpisodeResult(status="success", failure_reason=None),
            ],
            101: [
                collect.EpisodeResult(status="success", failure_reason=None),
                collect.EpisodeResult(status="success", failure_reason=None),
                collect.EpisodeResult(status="success", failure_reason=None),
            ],
        }
    )

    summary = collect.collect_dataset(config, runtime_factory=lambda _: runtime)

    assert runtime.started == [17, 101]
    assert runtime.stopped == [17, 101]
    manifest = [
        json.loads(line)
        for line in (tmp_path / "dataset_run" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["seed"], row["goal"]["id"], row["status"]) for row in manifest] == [
        (17, "g0", "success"),
        (17, "g1", "failed"),
        (101, "g0", "success"),
        (101, "g1", "success"),
        (101, "g2", "success"),
    ]
    assert summary["episodes"] == 5
    assert summary["failed"] == 1
    assert (tmp_path / "dataset_run" / "seed_17" / "scene").is_dir()
    assert (tmp_path / "dataset_run" / "seed_17" / "episodes" / "episode_000_g0" / "episode.json").exists()


def test_continue_on_failure_attempts_all_goals_in_seed(tmp_path):
    collect = load_collector_module()
    goals_path = tmp_path / "goals.yaml"
    goals_path.write_text(
        yaml.safe_dump(
            {
                "goals": [
                    {"id": "g0", "x": 1.0, "y": 0.0, "yaw": 0.0},
                    {"id": "g1", "x": 2.0, "y": 0.0, "yaw": 0.0},
                    {"id": "g2", "x": 3.0, "y": 0.0, "yaw": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = collect.CollectorConfig(
        goals_path=goals_path,
        seeds=[17],
        output_dir=tmp_path / "dataset_run",
        stop_on_failure=False,
        thresholds=collect.FailureThresholds(),
    )
    runtime = FakeRuntime(
        {
            17: [
                collect.EpisodeResult(status="success", failure_reason=None),
                collect.EpisodeResult(status="failed", failure_reason="no_progress"),
                collect.EpisodeResult(status="success", failure_reason=None),
            ],
        }
    )

    summary = collect.collect_dataset(config, runtime_factory=lambda _: runtime)
    manifest = [
        json.loads(line)
        for line in (tmp_path / "dataset_run" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [(row["goal"]["id"], row["status"]) for row in manifest] == [
        ("g0", "success"),
        ("g1", "failed"),
        ("g2", "success"),
    ]
    assert summary["episodes"] == 3
    assert summary["success"] == 2
    assert summary["failed"] == 1


def test_collect_dataset_prints_episode_status_after_each_episode(tmp_path, capsys):
    collect = load_collector_module()
    goals_path = tmp_path / "goals.yaml"
    goals_path.write_text(
        yaml.safe_dump(
            {
                "goals": [
                    {"id": "g0", "x": 1.0, "y": 0.0, "yaw": 0.0},
                    {"id": "g1", "x": 2.0, "y": 0.0, "yaw": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = collect.CollectorConfig(
        goals_path=goals_path,
        seeds=[17],
        output_dir=tmp_path / "dataset_run",
        stop_on_failure=False,
        thresholds=collect.FailureThresholds(),
    )
    runtime = FakeRuntime(
        {
            17: [
                collect.EpisodeResult(status="success", failure_reason=None),
                collect.EpisodeResult(status="failed", failure_reason="no_progress"),
            ],
        }
    )

    collect.collect_dataset(config, runtime_factory=lambda _: runtime)

    output_lines = [line for line in capsys.readouterr().out.splitlines() if "episode_complete" in line]
    assert output_lines == [
        "[dataset] episode_complete seed=17 episode=1/2 goal=g0 status=success total=1 success=1 failed=0",
        "[dataset] episode_complete seed=17 episode=2/2 goal=g1 status=failed reason=no_progress total=2 success=1 failed=1",
    ]


def test_goal_subscriber_ready_requires_mppi_node_name():
    collect = load_collector_module()

    assert not collect.goal_subscriber_ready(
        subscription_count=2,
        subscriber_names=["traversability_map", "rviz2"],
        required_name="fdm_mppi_mujoco_closed_loop",
    )
    assert collect.goal_subscriber_ready(
        subscription_count=3,
        subscriber_names=["traversability_map", "/fdm_mppi_mujoco_closed_loop"],
        required_name="fdm_mppi_mujoco_closed_loop",
    )
    assert collect.goal_subscriber_ready(
        subscription_count=3,
        subscriber_names=["/robot1/fdm_mppi_mujoco_closed_loop"],
        required_name="fdm_mppi_mujoco_closed_loop",
    )
    assert collect.goal_subscriber_ready(
        subscription_count=2,
        subscriber_names=[],
        required_name="",
    )


def test_choose_ros_domain_id_validates_explicit_domain():
    collect = load_collector_module()

    assert collect.choose_ros_domain_id(seed=17, configured_domain_id=42) == 42
    with pytest.raises(ValueError, match="ROS_DOMAIN_ID"):
        collect.choose_ros_domain_id(seed=17, configured_domain_id=233)


def test_collect_dataset_passes_absolute_seed_dir_to_runtime(tmp_path, monkeypatch):
    collect = load_collector_module()
    goals_path = tmp_path / "goals.yaml"
    goals_path.write_text(
        yaml.safe_dump({"goals": [{"id": "g0", "x": 1.0, "y": 0.0, "yaw": 0.0}]}),
        encoding="utf-8",
    )
    relative_output = Path("results/nav_dataset/test_relative_output")
    monkeypatch.chdir(SCRIPT_PATH.parents[1])
    runtime = AbsolutePathRuntime()

    collect.collect_dataset(
        collect.CollectorConfig(
            goals_path=goals_path,
            seeds=[17],
            output_dir=relative_output,
            thresholds=collect.FailureThresholds(),
        ),
        runtime_factory=lambda _: runtime,
    )

    assert runtime.seed_dirs
    assert all(path.is_absolute() for path in runtime.seed_dirs)


def test_seed_startup_failure_still_stops_runtime(tmp_path):
    collect = load_collector_module()
    goals_path = tmp_path / "goals.yaml"
    goals_path.write_text(
        yaml.safe_dump({"goals": [{"id": "g0", "x": 1.0, "y": 0.0, "yaw": 0.0}]}),
        encoding="utf-8",
    )
    config = collect.CollectorConfig(
        goals_path=goals_path,
        seeds=[17],
        output_dir=tmp_path / "dataset_run",
        thresholds=collect.FailureThresholds(),
    )
    runtime = FailingStartRuntime()

    with pytest.raises(RuntimeError, match="startup failed"):
        collect.collect_dataset(config, runtime_factory=lambda _: runtime)

    assert runtime.started == [17]
    assert runtime.stopped == [17]


class FakeRuntime:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.started = []
        self.stopped = []

    def start_seed(self, seed, seed_dir, config):
        self.started.append(seed)
        (seed_dir / "scene" / "fake_scene.txt").write_text(f"seed={seed}\n", encoding="utf-8")

    def run_episode(self, seed, episode_index, goal, episode_dir, thresholds):
        result = self.outcomes[seed][episode_index]
        episode_dir.mkdir(parents=True, exist_ok=True)
        collect = load_collector_module()
        collect.write_episode_artifacts(
            episode_dir,
            seed=seed,
            episode_index=episode_index,
            goal=goal,
            start_time=10.0 + episode_index,
            end_time=12.0 + episode_index,
            status=result.status,
            failure_reason=result.failure_reason,
            odom_samples=[
                collect.OdomRecord(stamp=10.0 + episode_index, x=0.0, y=0.0, yaw=0.0, vx=0.1, vy=0.0, wz=0.0),
                collect.OdomRecord(stamp=12.0 + episode_index, x=goal.x, y=goal.y, yaw=goal.yaw, vx=0.0, vy=0.0, wz=0.0),
            ],
            cmd_samples=[],
            frontend_path=[],
            tltrajectory=[],
            dynamic_obstacles=[],
            local_costmaps=[],
        )
        return result

    def stop_seed(self, seed):
        self.stopped.append(seed)


class FailingStartRuntime:
    def __init__(self):
        self.started = []
        self.stopped = []

    def start_seed(self, seed, seed_dir, config):
        del seed_dir, config
        self.started.append(seed)
        raise RuntimeError("startup failed")

    def run_episode(self, seed, episode_index, goal, episode_dir, thresholds):
        raise AssertionError("run_episode should not be called")

    def stop_seed(self, seed):
        self.stopped.append(seed)


class AbsolutePathRuntime(FakeRuntime):
    def __init__(self):
        collect = load_collector_module()
        super().__init__({17: [collect.EpisodeResult(status="success", failure_reason=None)]})
        self.seed_dirs = []

    def start_seed(self, seed, seed_dir, config):
        self.seed_dirs.append(Path(seed_dir))
        super().start_seed(seed, seed_dir, config)
