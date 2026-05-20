#!/usr/bin/env python3
"""Collect raw Geomapping navigation episodes across scene seeds and goals."""

import argparse
import csv
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


WORKSPACE_ROOT = Path("/home/mexxiie/prj")
GEOMAPPING_ROOT = WORKSPACE_ROOT / "Geomapping_ros2"
AUSIM_ROOT = WORKSPACE_ROOT / "ausim2"
DEFAULT_PROFILE = GEOMAPPING_ROOT / "src" / "mppi_controller" / "configs" / "mujoco_rviz_goal.yaml"
DEFAULT_OBSTACLE_CONFIG = GEOMAPPING_ROOT / "src" / "mppi_controller" / "configs" / "obstacle_scout_sparse.yaml"
DEFAULT_OUTPUT_ROOT = GEOMAPPING_ROOT / "results" / "nav_dataset"
CONTROLLER_EXECUTABLE = GEOMAPPING_ROOT / "install" / "mppi_controller" / "lib" / "mppi_controller" / "fdm_mppi"


@dataclass(frozen=True)
class Goal:
    id: str
    x: float
    y: float
    yaw: float

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "x": float(self.x), "y": float(self.y), "yaw": float(self.yaw)}


@dataclass(frozen=True)
class RecordSample:
    stamp: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class OdomRecord:
    stamp: float
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float


@dataclass(frozen=True)
class CmdRecord:
    stamp: float
    linear_x: float
    linear_y: float
    angular_z: float


@dataclass(frozen=True)
class JsonRecord:
    stamp: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class CostmapRecord:
    stamp: float
    origin: tuple[float, float]
    resolution: float
    width: int
    height: int
    reward_cost: list[float]
    height_layer: list[float]
    roughness: list[float]
    cost_map: list[float]


@dataclass(frozen=True)
class ProcessExit:
    name: str
    returncode: int | None


@dataclass(frozen=True)
class Failure:
    reason: str
    detail: str


@dataclass(frozen=True)
class FailureThresholds:
    goal_timeout_s: float = 180.0
    odom_timeout_s: float = 2.0
    no_progress_window_s: float = 20.0
    no_progress_min_delta_m: float = 0.2
    low_speed_window_s: float = 10.0
    low_speed_threshold_mps: float = 0.03
    goal_tolerance_m: float = 0.3


@dataclass(frozen=True)
class EpisodeResult:
    status: str
    failure_reason: str | None = None
    episode: dict[str, Any] | None = None


@dataclass
class CollectorConfig:
    goals_path: Path
    seeds: list[int]
    output_dir: Path
    max_goals_per_seed: int | None = None
    stop_on_failure: bool = True
    thresholds: FailureThresholds = FailureThresholds()
    profile: Path = DEFAULT_PROFILE
    controller: str = "nominal_cuda"
    headless: bool = True
    launch_rviz: bool = False
    obstacle_config: Path | None = DEFAULT_OBSTACLE_CONFIG
    odom_topic: str = "/scout1/odom"
    goal_topic: str = "/move_base_simple/goal"
    cmd_topic: str = "/joy/cmd_vel"
    frontend_path_topic: str = "/smooth_path"
    tltrajectory_topic: str = "/tltrajectory"
    dynamic_obstacles_topic: str = "/dyn_obstacle"
    local_costmap_topic: str = "/msg_local_reward"
    local_costmap_sample_hz: float = 2.0
    controller_odom_timeout_s: float = 30.0
    readiness_timeout_odom_s: float = 60.0
    readiness_timeout_goal_subscriber_s: float = 60.0
    reach_dwell_s: float = 0.5


class TopicBuffer:
    def __init__(self) -> None:
        self._samples: dict[str, list[Any]] = defaultdict(list)

    def add(self, topic: str, sample: Any) -> None:
        self._samples[str(topic)].append(sample)

    def slice(self, topic: str, start: float, end: float) -> list[Any]:
        samples = sorted(self._samples.get(str(topic), []), key=lambda item: float(item.stamp))
        return [sample for sample in samples if float(sample.stamp) >= float(start) and float(sample.stamp) < float(end)]

    def latest(self, topic: str) -> Any | None:
        samples = self._samples.get(str(topic), [])
        return samples[-1] if samples else None


class SampleThrottle:
    def __init__(self, *, rate_hz: float) -> None:
        self.period_s = 1.0 / max(float(rate_hz), 1e-9)
        self._last_stamp: float | None = None

    def accept(self, stamp: float) -> bool:
        stamp = float(stamp)
        if self._last_stamp is None or stamp - self._last_stamp >= self.period_s - 1e-9:
            self._last_stamp = stamp
            return True
        return False


def load_goals(path: str | Path) -> list[Goal]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_goals = data.get("goals")
    if not isinstance(raw_goals, list):
        raise ValueError("goals YAML must contain a list field: goals")
    goals: list[Goal] = []
    for index, raw_goal in enumerate(raw_goals):
        if not isinstance(raw_goal, dict):
            raise ValueError(f"goals[{index}] must be a mapping")
        for field in ("id", "x", "y", "yaw"):
            if field not in raw_goal:
                raise ValueError(f"goals[{index}].{field} is required")
        goal_id = str(raw_goal["id"]).strip()
        if not goal_id:
            raise ValueError(f"goals[{index}].id must be non-empty")
        goals.append(
            Goal(
                id=goal_id,
                x=float(raw_goal["x"]),
                y=float(raw_goal["y"]),
                yaw=float(raw_goal["yaw"]),
            )
        )
    return goals


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one integer seed")
    return seeds


def classify_failure(
    *,
    now: float,
    episode_start: float,
    goal: Goal,
    odom_samples: list[OdomRecord],
    process_exits: list[ProcessExit],
    thresholds: FailureThresholds,
) -> Failure | None:
    if process_exits:
        exit_info = process_exits[0]
        return Failure("process_exit", f"{exit_info.name} exited with code {exit_info.returncode}")

    samples = sorted(odom_samples, key=lambda item: float(item.stamp))
    if not samples:
        if float(now) - float(episode_start) >= float(thresholds.odom_timeout_s):
            return Failure("odom_timeout", f"no odom for {thresholds.odom_timeout_s:.3f}s")
        return None

    last_age = float(now) - float(samples[-1].stamp)
    if last_age > float(thresholds.odom_timeout_s):
        return Failure("odom_timeout", f"last odom age {last_age:.3f}s")

    if goal_reached(samples[-1], goal, thresholds.goal_tolerance_m):
        return None

    elapsed = float(now) - float(episode_start)
    if elapsed >= float(thresholds.goal_timeout_s):
        return Failure("timeout", f"goal timeout {thresholds.goal_timeout_s:.3f}s")

    progress = _window_progress(samples, goal, now=float(now), window_s=float(thresholds.no_progress_window_s))
    if progress is not None and progress < float(thresholds.no_progress_min_delta_m):
        return Failure("no_progress", f"progress {progress:.3f}m in {thresholds.no_progress_window_s:.3f}s")

    avg_speed = _window_average_speed(samples, now=float(now), window_s=float(thresholds.low_speed_window_s))
    if avg_speed is not None and avg_speed < float(thresholds.low_speed_threshold_mps):
        return Failure("low_speed_stall", f"average speed {avg_speed:.3f}m/s in {thresholds.low_speed_window_s:.3f}s")

    return None


def goal_reached(sample: OdomRecord, goal: Goal, tolerance_m: float) -> bool:
    return _goal_distance(sample, goal) <= float(tolerance_m)


def write_episode_artifacts(
    episode_dir: str | Path,
    *,
    seed: int,
    episode_index: int,
    goal: Goal,
    start_time: float,
    end_time: float,
    status: str,
    failure_reason: str | None,
    odom_samples: list[OdomRecord],
    cmd_samples: list[CmdRecord],
    frontend_path: list[JsonRecord],
    tltrajectory: list[JsonRecord],
    dynamic_obstacles: list[JsonRecord],
    local_costmaps: list[CostmapRecord],
) -> dict[str, Any]:
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    odom_samples = sorted(odom_samples, key=lambda item: item.stamp)
    cmd_samples = sorted(cmd_samples, key=lambda item: item.stamp)
    _write_odom_csv(odom_samples, episode_dir / "odom.csv", start_time=start_time)
    _write_cmd_csv(cmd_samples, episode_dir / "cmd.csv", start_time=start_time)
    _write_jsonl(frontend_path, episode_dir / "frontend_path.jsonl", start_time=start_time)
    _write_jsonl(tltrajectory, episode_dir / "tltrajectory.jsonl", start_time=start_time)
    _write_jsonl(dynamic_obstacles, episode_dir / "dynamic_obstacles.jsonl", start_time=start_time)
    write_local_costmap_npz(local_costmaps, episode_dir / "local_costmap.npz")
    write_episode_plots(
        episode_dir,
        odom_samples=odom_samples,
        goal=goal,
        dynamic_obstacles=dynamic_obstacles,
    )

    metrics = compute_episode_metrics(odom_samples, goal)
    episode = {
        "seed": int(seed),
        "episode_index": int(episode_index),
        "goal": goal.as_dict(),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "duration_s": max(0.0, float(end_time) - float(start_time)),
        "status": str(status),
        "failure_reason": failure_reason,
        **metrics,
        "artifacts": {
            "odom": "odom.csv",
            "cmd": "cmd.csv",
            "frontend_path": "frontend_path.jsonl",
            "tltrajectory": "tltrajectory.jsonl",
            "dynamic_obstacles": "dynamic_obstacles.jsonl",
            "local_costmap": "local_costmap.npz",
            "trajectory_plot": "trajectory.png",
            "obstacle_plot": "obstacles.png",
        },
    }
    (episode_dir / "episode.json").write_text(json.dumps(episode, indent=2), encoding="utf-8")
    return episode


def write_local_costmap_npz(samples: list[CostmapRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        np.savez_compressed(
            path,
            stamp=np.asarray([], dtype=np.float64),
            origin=np.empty((0, 2), dtype=np.float32),
            resolution=np.asarray([], dtype=np.float32),
            width=np.asarray([], dtype=np.int32),
            height=np.asarray([], dtype=np.int32),
            reward_cost=np.empty((0, 0), dtype=np.float32),
            height_layer=np.empty((0, 0), dtype=np.float32),
            height_data=np.empty((0, 0), dtype=np.float32),
            roughness=np.empty((0, 0), dtype=np.float32),
            cost_map=np.empty((0, 0), dtype=np.float32),
        )
        return
    height_layers = _stack_layer(samples, "height_layer")
    np.savez_compressed(
        path,
        stamp=np.asarray([sample.stamp for sample in samples], dtype=np.float64),
        origin=np.asarray([sample.origin for sample in samples], dtype=np.float32),
        resolution=np.asarray([sample.resolution for sample in samples], dtype=np.float32),
        width=np.asarray([sample.width for sample in samples], dtype=np.int32),
        height=np.asarray([sample.height for sample in samples], dtype=np.int32),
        reward_cost=_stack_layer(samples, "reward_cost"),
        height_layer=height_layers,
        height_data=height_layers,
        roughness=_stack_layer(samples, "roughness"),
        cost_map=_stack_layer(samples, "cost_map"),
    )


def write_episode_plots(
    episode_dir: str | Path,
    *,
    odom_samples: list[OdomRecord],
    goal: Goal,
    dynamic_obstacles: list[JsonRecord],
) -> None:
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    obstacles = _latest_obstacles(dynamic_obstacles)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    xs = [float(sample.x) for sample in odom_samples]
    ys = [float(sample.y) for sample in odom_samples]
    obstacle_xs = [float(item.get("x", 0.0)) for item in obstacles]
    obstacle_ys = [float(item.get("y", 0.0)) for item in obstacles]
    extent_x = xs + obstacle_xs + [float(goal.x)]
    extent_y = ys + obstacle_ys + [float(goal.y)]
    if not extent_x:
        extent_x = [0.0]
    if not extent_y:
        extent_y = [0.0]
    margin = 1.5
    x_min = min(extent_x) - margin
    x_max = max(extent_x) + margin
    y_min = min(extent_y) - margin
    y_max = max(extent_y) + margin

    def setup_axis(axis: Any, *, title: str) -> None:
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(title)

    trajectory_figure, trajectory_axis = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    _draw_obstacles(trajectory_axis, obstacles, Circle)
    if xs and ys:
        trajectory_axis.plot(xs, ys, color="#155eef", linewidth=2.0, label="trajectory")
        trajectory_axis.scatter(xs[0], ys[0], s=36, color="#111827", marker="o", label="start")
        trajectory_axis.scatter(xs[-1], ys[-1], s=40, color="#b42318", marker="x", label="end")
    trajectory_axis.scatter(float(goal.x), float(goal.y), s=58, color="#16a34a", marker="*", label="goal")
    setup_axis(trajectory_axis, title="Trajectory")
    trajectory_axis.legend(loc="best")
    trajectory_figure.tight_layout()
    trajectory_figure.savefig(episode_dir / "trajectory.png", bbox_inches="tight")
    plt.close(trajectory_figure)

    obstacle_figure, obstacle_axis = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    _draw_obstacles(obstacle_axis, obstacles, Circle)
    if xs and ys:
        obstacle_axis.plot(xs, ys, color="#667085", linewidth=1.2, alpha=0.45, label="trajectory")
    obstacle_axis.scatter(float(goal.x), float(goal.y), s=58, color="#16a34a", marker="*", label="goal")
    setup_axis(obstacle_axis, title="Obstacles")
    obstacle_axis.legend(loc="best")
    obstacle_figure.tight_layout()
    obstacle_figure.savefig(episode_dir / "obstacles.png", bbox_inches="tight")
    plt.close(obstacle_figure)


def selected_goals_for_config(config: CollectorConfig) -> list[Goal]:
    goals = load_goals(config.goals_path)
    if config.max_goals_per_seed is not None:
        goals = goals[: max(int(config.max_goals_per_seed), 0)]
    return goals


def collect_dataset(
    config: CollectorConfig,
    *,
    runtime_factory: Callable[[CollectorConfig], Any] | None = None,
) -> dict[str, Any]:
    goals = selected_goals_for_config(config)
    if not goals:
        raise ValueError("no goals selected for collection")

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(config, goals, output_dir / "config.yaml")
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    runtime = runtime_factory(config) if runtime_factory is not None else RealGeomappingRuntime(config)
    summary = {
        "output_dir": str(output_dir),
        "seeds": [int(seed) for seed in config.seeds],
        "goals": [goal.as_dict() for goal in goals],
        "episodes": 0,
        "success": 0,
        "failed": 0,
        "seeds_completed": 0,
    }
    for seed in config.seeds:
        seed_dir = output_dir / f"seed_{int(seed)}"
        (seed_dir / "scene").mkdir(parents=True, exist_ok=True)
        (seed_dir / "episodes").mkdir(parents=True, exist_ok=True)
        try:
            runtime.start_seed(int(seed), seed_dir, config)
            for episode_index, goal in enumerate(goals):
                episode_dir = seed_dir / "episodes" / f"episode_{episode_index:03d}_{safe_tag(goal.id)}"
                result = runtime.run_episode(int(seed), episode_index, goal, episode_dir, config.thresholds)
                episode = result.episode or _load_episode_json(episode_dir)
                manifest_entry = _manifest_entry(
                    output_dir=output_dir,
                    episode_dir=episode_dir,
                    seed=int(seed),
                    episode_index=episode_index,
                    goal=goal,
                    result=result,
                    episode=episode,
                )
                append_jsonl(manifest_path, manifest_entry)
                summary["episodes"] += 1
                if result.status == "success":
                    summary["success"] += 1
                else:
                    summary["failed"] += 1
                    if config.stop_on_failure:
                        break
            summary["seeds_completed"] += 1
        finally:
            runtime.stop_seed(int(seed))

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class RealGeomappingRuntime:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self.env: dict[str, str] | None = None
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.threads: list[threading.Thread] = []
        self.recorder: RosDatasetRecorder | None = None
        self.seed_dir: Path | None = None
        self.native_run_dir: Path | None = None

    def start_seed(self, seed: int, seed_dir: Path, config: CollectorConfig) -> None:
        self.config = config
        self.seed_dir = seed_dir
        log_path = seed_dir / "seed.log"
        env = process_env()
        prepared_obstacle_config = prepare_obstacle_config(
            scene_dir=seed_dir / "scene",
            scene_seed=seed,
            obstacle_config=config.obstacle_config,
        )
        if prepared_obstacle_config is not None:
            env["AUSIM_DYNAMIC_OBSTACLE_CONFIG_OVERRIDE"] = str(prepared_obstacle_config)
        self.env = env

        ausim_cmd = f"{shlex.quote(str(AUSIM_ROOT / 'em_run.sh'))} {'--headless' if config.headless else ''}".strip()
        adapter_cmd = (
            "ros2 launch ausim_geomapping_adapter ausim_scout_mppi_frontend.launch.py "
            f"launch_rviz:={'true' if config.launch_rviz else 'false'} "
            "launch_mppi:=false "
            f"mppi_profile:={shlex.quote(str(Path(config.profile).resolve()))} "
            f"mppi_controller:={shlex.quote(str(config.controller))}"
        )
        native_run_dir = seed_dir / "native_run"
        self.native_run_dir = native_run_dir
        seed_goal_count = max(1, len(selected_goals_for_config(config)))
        max_steps = controller_max_steps(
            config.profile,
            controller=config.controller,
            timeout_s=config.thresholds.goal_timeout_s,
            episodes=seed_goal_count,
        )
        controller_cmd = controller_command(
            profile=config.profile,
            controller=config.controller,
            native_run_dir=native_run_dir,
            max_steps=max_steps,
            odom_timeout_s=config.controller_odom_timeout_s,
        )
        for name, command in (("ausim2", ausim_cmd), ("adapter", adapter_cmd), ("mppi", controller_cmd)):
            process, thread = start_process(label=name, command=command, log_path=log_path, env=env)
            self.processes[name] = process
            self.threads.append(thread)

        self.recorder = RosDatasetRecorder(config)
        self._wait_ready()

    def run_episode(
        self,
        seed: int,
        episode_index: int,
        goal: Goal,
        episode_dir: Path,
        thresholds: FailureThresholds,
    ) -> EpisodeResult:
        if self.recorder is None:
            raise RuntimeError("seed runtime has not been started")
        recorder = self.recorder
        episode_start_wall = time.time()
        start_time = recorder.now()
        for _ in range(8):
            recorder.publish_goal(goal)
            recorder.spin_once(0.05)
            time.sleep(0.10)

        reach_started_at: float | None = None
        failure: Failure | None = None
        while True:
            recorder.spin_once(0.05)
            now = recorder.now()
            process_exits = self._process_exits()
            odom_window = recorder.odom_samples(start_time, now + 1e-6)
            if odom_window and goal_reached(odom_window[-1], goal, thresholds.goal_tolerance_m):
                if reach_started_at is None:
                    reach_started_at = now
                elif now - reach_started_at >= float(self.config.reach_dwell_s):
                    break
            else:
                reach_started_at = None
            if self.native_run_dir is not None and native_goal_reached(self.native_run_dir, min_mtime=episode_start_wall):
                break
            failure = classify_failure(
                now=now,
                episode_start=start_time,
                goal=goal,
                odom_samples=odom_window,
                process_exits=process_exits,
                thresholds=thresholds,
            )
            if failure is not None:
                break
            time.sleep(0.02)

        end_time = recorder.now()
        status = "success" if failure is None else "failed"
        episode = recorder.write_episode_window(
            episode_dir,
            seed=seed,
            episode_index=episode_index,
            goal=goal,
            start_time=start_time,
            end_time=end_time,
            status=status,
            failure_reason=None if failure is None else failure.reason,
        )
        return EpisodeResult(status=status, failure_reason=None if failure is None else failure.reason, episode=episode)

    def stop_seed(self, seed: int) -> None:
        del seed
        for name in ("mppi", "adapter", "ausim2"):
            process = self.processes.get(name)
            if process is not None:
                terminate_process(process)
        for thread in self.threads:
            thread.join(timeout=2.0)
        self.processes.clear()
        self.threads.clear()
        if self.recorder is not None:
            self.recorder.shutdown()
            self.recorder = None

    def _wait_ready(self) -> None:
        assert self.recorder is not None
        odom_deadline = time.monotonic() + float(self.config.readiness_timeout_odom_s)
        while time.monotonic() < odom_deadline:
            self.recorder.spin_once(0.1)
            if self.recorder.has_odom():
                break
            exits = self._process_exits()
            if exits:
                raise RuntimeError(exits[0].detail if hasattr(exits[0], "detail") else f"{exits[0].name} exited")
        if not self.recorder.has_odom():
            raise RuntimeError("odom readiness timeout")

        goal_deadline = time.monotonic() + float(self.config.readiness_timeout_goal_subscriber_s)
        while time.monotonic() < goal_deadline:
            self.recorder.spin_once(0.1)
            if self.recorder.goal_subscription_count() >= 2:
                return
            exits = self._process_exits()
            if exits:
                raise RuntimeError(f"{exits[0].name} exited with code {exits[0].returncode}")
        raise RuntimeError("goal subscriber readiness timeout")

    def _process_exits(self) -> list[ProcessExit]:
        exits = []
        for name, process in self.processes.items():
            code = process.poll()
            if code is not None:
                exits.append(ProcessExit(name=name, returncode=code))
        return exits


class RosDatasetRecorder:
    def __init__(self, config: CollectorConfig) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Odometry
        from nav_msgs.msg import Path as NavPath
        from rclpy.node import Node

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("geomapping_dataset_collect")
        self.config = config
        self.buffer = TopicBuffer()
        self._costmap_throttle = SampleThrottle(rate_hz=config.local_costmap_sample_hz)
        self._pose_stamped_type = PoseStamped
        self.goal_pub = self.node.create_publisher(PoseStamped, config.goal_topic, 10)
        self.node.create_subscription(Odometry, config.odom_topic, self._on_odom, 100)
        self.node.create_subscription(Twist, config.cmd_topic, self._on_cmd, 100)
        self.node.create_subscription(NavPath, config.frontend_path_topic, self._on_frontend_path, 10)
        self._subscribe_tltrajectory()
        self._subscribe_dynamic_obstacles()
        self._subscribe_local_costmap()

    def now(self) -> float:
        return time.monotonic()

    def spin_once(self, timeout: float = 0.1) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=float(timeout))

    def shutdown(self) -> None:
        try:
            self.node.destroy_node()
        finally:
            if self._rclpy.ok():
                self._rclpy.shutdown()

    def has_odom(self) -> bool:
        return self.buffer.latest("odom") is not None

    def goal_subscription_count(self) -> int:
        return int(self.goal_pub.get_subscription_count())

    def publish_goal(self, goal: Goal) -> None:
        message = self._pose_stamped_type()
        message.header.frame_id = "map"
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.pose.position.x = float(goal.x)
        message.pose.position.y = float(goal.y)
        qx, qy, qz, qw = yaw_to_quaternion(goal.yaw)
        message.pose.orientation.x = qx
        message.pose.orientation.y = qy
        message.pose.orientation.z = qz
        message.pose.orientation.w = qw
        self.goal_pub.publish(message)

    def odom_samples(self, start_time: float, end_time: float) -> list[OdomRecord]:
        return self.buffer.slice("odom", start_time, end_time)

    def write_episode_window(
        self,
        episode_dir: Path,
        *,
        seed: int,
        episode_index: int,
        goal: Goal,
        start_time: float,
        end_time: float,
        status: str,
        failure_reason: str | None,
    ) -> dict[str, Any]:
        return write_episode_artifacts(
            episode_dir,
            seed=seed,
            episode_index=episode_index,
            goal=goal,
            start_time=start_time,
            end_time=end_time,
            status=status,
            failure_reason=failure_reason,
            odom_samples=self.buffer.slice("odom", start_time, end_time),
            cmd_samples=self.buffer.slice("cmd", start_time, end_time),
            frontend_path=self.buffer.slice("frontend_path", start_time, end_time),
            tltrajectory=self.buffer.slice("tltrajectory", start_time, end_time),
            dynamic_obstacles=self.buffer.slice("dynamic_obstacles", start_time, end_time),
            local_costmaps=self.buffer.slice("local_costmap", start_time, end_time),
        )

    def _on_odom(self, message: Any) -> None:
        pose = message.pose.pose
        twist = message.twist.twist
        self.buffer.add(
            "odom",
            OdomRecord(
                stamp=self.now(),
                x=float(pose.position.x),
                y=float(pose.position.y),
                yaw=yaw_from_quaternion(pose.orientation),
                vx=float(twist.linear.x),
                vy=float(twist.linear.y),
                wz=float(twist.angular.z),
            ),
        )

    def _on_cmd(self, message: Any) -> None:
        self.buffer.add(
            "cmd",
            CmdRecord(
                stamp=self.now(),
                linear_x=float(message.linear.x),
                linear_y=float(message.linear.y),
                angular_z=float(message.angular.z),
            ),
        )

    def _on_frontend_path(self, message: Any) -> None:
        stamp = self.now()
        points = _nav_path_points(message)
        self.buffer.add("frontend_path", JsonRecord(stamp=stamp, payload=_path_payload(points)))

    def _on_tltrajectory(self, message: Any) -> None:
        stamp = self.now()
        points = [[float(point.x), float(point.y)] for point in list(getattr(message, "pos_pts", []))]
        payload = _path_payload(points)
        payload["t_pts"] = [float(value) for value in list(getattr(message, "t_pts", []))]
        self.buffer.add("tltrajectory", JsonRecord(stamp=stamp, payload=payload))

    def _on_dynamic_obstacles(self, message: Any) -> None:
        self.buffer.add("dynamic_obstacles", JsonRecord(stamp=self.now(), payload=_dynamic_obstacles_payload(message)))

    def _on_local_costmap(self, message: Any) -> None:
        stamp = self.now()
        if not self._costmap_throttle.accept(stamp):
            return
        info = message.occupancy.info
        self.buffer.add(
            "local_costmap",
            CostmapRecord(
                stamp=stamp,
                origin=(float(info.origin.position.x), float(info.origin.position.y)),
                resolution=float(info.resolution),
                width=int(info.width),
                height=int(info.height),
                reward_cost=[float(value) for value in list(getattr(message, "reward_cost", []))],
                height_layer=[float(value) for value in list(getattr(message, "height", []))],
                roughness=[float(value) for value in list(getattr(message, "roughness", []))],
                cost_map=[float(value) for value in list(getattr(message, "cost_map", []))],
            ),
        )

    def _subscribe_tltrajectory(self) -> None:
        try:
            from traversability_mapping.msg import Polynome
        except Exception as exc:
            self.node.get_logger().warning(f"Could not subscribe to {self.config.tltrajectory_topic}: {exc}")
            return
        self.node.create_subscription(Polynome, self.config.tltrajectory_topic, self._on_tltrajectory, 10)

    def _subscribe_dynamic_obstacles(self) -> None:
        try:
            from ausim_msg.msg import BoundingBox3DArray
        except Exception as exc:
            self.node.get_logger().warning(f"Could not subscribe to {self.config.dynamic_obstacles_topic}: {exc}")
            return
        self.node.create_subscription(
            BoundingBox3DArray,
            self.config.dynamic_obstacles_topic,
            self._on_dynamic_obstacles,
            10,
        )

    def _subscribe_local_costmap(self) -> None:
        try:
            from elevation_msgs.msg import OccupancyElevation
        except Exception as exc:
            self.node.get_logger().warning(f"Could not subscribe to {self.config.local_costmap_topic}: {exc}")
            return
        self.node.create_subscription(
            OccupancyElevation,
            self.config.local_costmap_topic,
            self._on_local_costmap,
            10,
        )


def prepare_obstacle_config(*, scene_dir: Path, scene_seed: int | None, obstacle_config: Path | None) -> Path | None:
    if obstacle_config is None and scene_seed is None:
        return None
    source = Path(obstacle_config).expanduser().resolve() if obstacle_config is not None else DEFAULT_OBSTACLE_CONFIG
    if not source.exists():
        raise FileNotFoundError(f"obstacle config not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if scene_seed is not None:
        data["random_seed"] = int(scene_seed)
    scene_dir.mkdir(parents=True, exist_ok=True)
    target = scene_dir / "obstacle_config.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def controller_max_steps(profile: str | Path, *, controller: str | None, timeout_s: float, episodes: int = 1) -> int:
    sampling_rate = 10.0
    profile_path = Path(profile).expanduser()
    try:
        package_path = GEOMAPPING_ROOT / "src" / "mppi_controller"
        if str(package_path) not in sys.path:
            sys.path.insert(0, str(package_path))
        from mppi_controller.experiment import build_experiment_config

        config, _metadata = build_experiment_config(profile_path, controller_name=controller)
        sampling_rate = float(config.get("simulation", {}).get("sampling_rate", sampling_rate))
    except Exception:
        if profile_path.exists():
            data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                sampling_rate = float(data.get("simulation", {}).get("sampling_rate", sampling_rate))
    total_timeout_s = float(timeout_s) * max(1, int(episodes))
    return max(1, int(math.ceil(total_timeout_s * max(sampling_rate, 1e-6))))


def controller_command(
    *,
    profile: str | Path,
    controller: str,
    native_run_dir: str | Path,
    max_steps: int,
    odom_timeout_s: float,
) -> str:
    return (
        f"{shlex.quote(str(CONTROLLER_EXECUTABLE))} mujoco-closed-loop "
        f"--profile {shlex.quote(str(Path(profile).resolve()))} "
        f"--controller {shlex.quote(str(controller))} "
        f"--results-dir {shlex.quote(str(Path(native_run_dir)))} "
        f"--max-steps {int(max_steps)} "
        f"--odom-timeout {float(odom_timeout_s)}"
    )


def native_goal_reached(native_run_dir: str | Path, *, min_mtime: float | None = None) -> bool:
    summary_path = Path(native_run_dir) / "summary.json"
    if not summary_path.exists():
        return False
    if min_mtime is not None and summary_path.stat().st_mtime < float(min_mtime):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(summary.get("reached_goal") or summary.get("success"))


def process_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("ROS_LOG_DIR", "/tmp/ros_logs")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
    env.setdefault("PYTHONUNBUFFERED", "1")
    Path(env["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def start_process(
    *,
    label: str,
    command: str,
    log_path: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[str], threading.Thread]:
    process = subprocess.Popen(
        source_and_exec(command),
        cwd=str(WORKSPACE_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    def forward_output() -> None:
        assert process.stdout is not None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            for line in process.stdout:
                stream.write(f"[{label}] {line}")
                stream.flush()

    thread = threading.Thread(target=forward_output, name=f"{label}-log", daemon=True)
    thread.start()
    return process, thread


def source_and_exec(command: str) -> list[str]:
    ausim_ros_setup = AUSIM_ROOT / "build" / "ros_ws" / "install" / "setup.bash"
    setup_parts = ["source /opt/ros/humble/setup.bash"]
    if ausim_ros_setup.exists():
        setup_parts.append(f"source {shlex.quote(str(ausim_ros_setup))}")
    setup_parts.append(f"source {shlex.quote(str(GEOMAPPING_ROOT / 'install' / 'setup.bash'))}")
    setup_parts.append(f"exec {command}")
    return ["bash", "-lc", " && ".join(setup_parts)]


def terminate_process(process: subprocess.Popen[str], *, timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def compute_episode_metrics(odom_samples: list[OdomRecord], goal: Goal) -> dict[str, Any]:
    if not odom_samples:
        return {
            "start_pose": None,
            "end_pose": None,
            "initial_distance_m": None,
            "final_distance_m": None,
            "min_distance_m": None,
            "path_length_m": 0.0,
            "progress_m": 0.0,
            "progress_ratio": 0.0,
            "odom_samples": 0,
        }
    distances = [_goal_distance(sample, goal) for sample in odom_samples]
    path_length = 0.0
    for previous, current in zip(odom_samples[:-1], odom_samples[1:]):
        path_length += math.hypot(float(current.x) - float(previous.x), float(current.y) - float(previous.y))
    initial = float(distances[0])
    final = float(distances[-1])
    progress = initial - final
    return {
        "start_pose": _pose_dict(odom_samples[0]),
        "end_pose": _pose_dict(odom_samples[-1]),
        "initial_distance_m": initial,
        "final_distance_m": final,
        "min_distance_m": float(min(distances)),
        "path_length_m": float(path_length),
        "progress_m": float(progress),
        "progress_ratio": 0.0 if initial <= 1e-9 else float(progress / initial),
        "odom_samples": len(odom_samples),
    }


def append_jsonl(path: str | Path, entry: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")


def safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return tag.strip("_") or "goal"


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = float(yaw) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_from_quaternion(quaternion: Any) -> float:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", required=True, help="YAML file with goals: [{id, x, y, yaw}].")
    parser.add_argument("--seeds", required=True, help="Comma-separated scene seeds, e.g. 17,101,211.")
    parser.add_argument("--output", default=None, help="Output run directory. Defaults to results/nav_dataset/<timestamp>.")
    parser.add_argument("--max-goals-per-seed", type=int, default=None, help="Limit goals per seed.")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Keep attempting later goals in the same seed after a failed episode.",
    )
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE), help="MPPI closed-loop profile.")
    parser.add_argument("--controller", default="nominal_cuda", help="Controller name from the profile.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="Run ausim2 headless.")
    parser.add_argument("--launch-rviz", action=argparse.BooleanOptionalAction, default=False, help="Launch RViz.")
    parser.add_argument("--obstacle-config", default=str(DEFAULT_OBSTACLE_CONFIG), help="Dynamic obstacle config template.")
    parser.add_argument("--odom-topic", default="/scout1/odom")
    parser.add_argument("--goal-topic", default="/move_base_simple/goal")
    parser.add_argument("--cmd-topic", default="/joy/cmd_vel")
    parser.add_argument("--frontend-path-topic", default="/smooth_path")
    parser.add_argument("--tltrajectory-topic", default="/tltrajectory")
    parser.add_argument("--dynamic-obstacles-topic", default="/dyn_obstacle")
    parser.add_argument("--local-costmap-topic", default="/msg_local_reward")
    parser.add_argument("--local-costmap-sample-hz", type=float, default=2.0)
    parser.add_argument(
        "--controller-odom-timeout-s",
        type=float,
        default=30.0,
        help="Internal MPPI odom timeout. Kept above the outer timeout so seed-level multi-goal runs survive artifact writes between goals.",
    )
    parser.add_argument("--goal-timeout-s", type=float, default=180.0)
    parser.add_argument("--odom-timeout-s", type=float, default=2.0)
    parser.add_argument("--no-progress-window-s", type=float, default=20.0)
    parser.add_argument("--no-progress-min-delta-m", type=float, default=0.2)
    parser.add_argument("--low-speed-window-s", type=float, default=10.0)
    parser.add_argument("--low-speed-threshold-mps", type=float, default=0.03)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.3)
    parser.add_argument("--reach-dwell-s", type=float, default=0.5)
    parser.add_argument("--readiness-timeout-odom-s", type=float, default=60.0)
    parser.add_argument("--readiness-timeout-goal-subscriber-s", type=float, default=60.0)
    return parser


def config_from_args(args: argparse.Namespace) -> CollectorConfig:
    output_dir = Path(args.output).expanduser() if args.output else DEFAULT_OUTPUT_ROOT / timestamp()
    obstacle_config = None if str(args.obstacle_config).lower() in {"none", ""} else Path(args.obstacle_config).expanduser()
    return CollectorConfig(
        goals_path=Path(args.goals).expanduser(),
        seeds=parse_seeds(args.seeds),
        output_dir=output_dir,
        max_goals_per_seed=args.max_goals_per_seed,
        stop_on_failure=not bool(args.continue_on_failure),
        thresholds=FailureThresholds(
            goal_timeout_s=float(args.goal_timeout_s),
            odom_timeout_s=float(args.odom_timeout_s),
            no_progress_window_s=float(args.no_progress_window_s),
            no_progress_min_delta_m=float(args.no_progress_min_delta_m),
            low_speed_window_s=float(args.low_speed_window_s),
            low_speed_threshold_mps=float(args.low_speed_threshold_mps),
            goal_tolerance_m=float(args.goal_tolerance_m),
        ),
        profile=Path(args.profile).expanduser(),
        controller=str(args.controller),
        headless=bool(args.headless),
        launch_rviz=bool(args.launch_rviz),
        obstacle_config=obstacle_config,
        odom_topic=str(args.odom_topic),
        goal_topic=str(args.goal_topic),
        cmd_topic=str(args.cmd_topic),
        frontend_path_topic=str(args.frontend_path_topic),
        tltrajectory_topic=str(args.tltrajectory_topic),
        dynamic_obstacles_topic=str(args.dynamic_obstacles_topic),
        local_costmap_topic=str(args.local_costmap_topic),
        local_costmap_sample_hz=float(args.local_costmap_sample_hz),
        controller_odom_timeout_s=float(args.controller_odom_timeout_s),
        readiness_timeout_odom_s=float(args.readiness_timeout_odom_s),
        readiness_timeout_goal_subscriber_s=float(args.readiness_timeout_goal_subscriber_s),
        reach_dwell_s=float(args.reach_dwell_s),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    summary = collect_dataset(config)
    print(json.dumps(summary, indent=2))
    return 0


def _write_odom_csv(samples: list[OdomRecord], path: Path, *, start_time: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stamp", "rel_t", "x", "y", "yaw", "vx", "vy", "wz"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "stamp": f"{sample.stamp:.6f}",
                    "rel_t": f"{sample.stamp - start_time:.6f}",
                    "x": f"{sample.x:.6f}",
                    "y": f"{sample.y:.6f}",
                    "yaw": f"{sample.yaw:.6f}",
                    "vx": f"{sample.vx:.6f}",
                    "vy": f"{sample.vy:.6f}",
                    "wz": f"{sample.wz:.6f}",
                }
            )


def _write_cmd_csv(samples: list[CmdRecord], path: Path, *, start_time: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stamp", "rel_t", "linear_x", "linear_y", "angular_z"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "stamp": f"{sample.stamp:.6f}",
                    "rel_t": f"{sample.stamp - start_time:.6f}",
                    "linear_x": f"{sample.linear_x:.6f}",
                    "linear_y": f"{sample.linear_y:.6f}",
                    "angular_z": f"{sample.angular_z:.6f}",
                }
            )


def _write_jsonl(samples: list[JsonRecord], path: Path, *, start_time: float) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for sample in sorted(samples, key=lambda item: item.stamp):
            payload = {"stamp": float(sample.stamp), "rel_t": float(sample.stamp - start_time), **sample.payload}
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _stack_layer(samples: list[CostmapRecord], attr: str) -> np.ndarray:
    arrays = [np.asarray(getattr(sample, attr), dtype=np.float32).reshape(-1) for sample in samples]
    lengths = {array.size for array in arrays}
    if len(lengths) == 1:
        return np.asarray(arrays, dtype=np.float32)
    return np.asarray(arrays, dtype=object)


def _latest_obstacles(records: list[JsonRecord]) -> list[dict[str, Any]]:
    for record in sorted(records, key=lambda item: float(item.stamp), reverse=True):
        obstacles = record.payload.get("obstacles", [])
        if isinstance(obstacles, list) and obstacles:
            return [item for item in obstacles if isinstance(item, dict)]
    return []


def _draw_obstacles(axis: Any, obstacles: list[dict[str, Any]], circle_type: Any) -> None:
    for index, obstacle in enumerate(obstacles):
        x = float(obstacle.get("x", 0.0))
        y = float(obstacle.get("y", 0.0))
        radius = max(float(obstacle.get("radius", 0.2)), 0.02)
        axis.add_patch(
            circle_type(
                (x, y),
                radius=radius,
                facecolor="#f97316",
                edgecolor="#9a3412",
                alpha=0.30,
                linewidth=1.2,
                label="obstacle" if index == 0 else None,
            )
        )
        axis.scatter(x, y, s=12, color="#9a3412")


def _window_progress(samples: list[OdomRecord], goal: Goal, *, now: float, window_s: float) -> float | None:
    if float(now) - float(samples[0].stamp) < float(window_s):
        return None
    start = float(now) - float(window_s)
    anchor = None
    for sample in samples:
        if float(sample.stamp) <= start:
            anchor = sample
        else:
            break
    window_samples = [sample for sample in samples if float(sample.stamp) >= start]
    if anchor is not None:
        window_samples.insert(0, anchor)
    if len(window_samples) < 2:
        return None
    initial_distance = _goal_distance(window_samples[0], goal)
    min_distance = min(_goal_distance(sample, goal) for sample in window_samples)
    return float(initial_distance - min_distance)


def _window_average_speed(samples: list[OdomRecord], *, now: float, window_s: float) -> float | None:
    start = float(now) - float(window_s)
    window_samples = [sample for sample in samples if float(sample.stamp) >= start]
    if len(window_samples) < 2:
        return None
    if float(window_samples[-1].stamp) - float(window_samples[0].stamp) < float(window_s) - 1e-6:
        return None
    speeds = [math.hypot(float(sample.vx), float(sample.vy)) for sample in window_samples]
    return float(sum(speeds) / len(speeds))


def _goal_distance(sample: OdomRecord, goal: Goal) -> float:
    return float(math.hypot(float(goal.x) - float(sample.x), float(goal.y) - float(sample.y)))


def _pose_dict(sample: OdomRecord) -> dict[str, float]:
    return {
        "stamp": float(sample.stamp),
        "x": float(sample.x),
        "y": float(sample.y),
        "yaw": float(sample.yaw),
        "vx": float(sample.vx),
        "vy": float(sample.vy),
        "wz": float(sample.wz),
    }


def _path_payload(points: list[list[float]]) -> dict[str, Any]:
    length_m = 0.0
    for previous, current in zip(points[:-1], points[1:]):
        length_m += math.hypot(float(current[0]) - float(previous[0]), float(current[1]) - float(previous[1]))
    return {"point_count": len(points), "length_m": float(length_m), "points": points}


def _nav_path_points(message: Any) -> list[list[float]]:
    points: list[list[float]] = []
    for pose_stamped in list(getattr(message, "poses", [])):
        position = pose_stamped.pose.position
        points.append([float(position.x), float(position.y)])
    return points


def _dynamic_obstacles_payload(message: Any) -> dict[str, Any]:
    obstacles = []
    for box in list(getattr(message, "boxes", [])):
        center = getattr(box, "center", None)
        position = getattr(center, "position", None)
        size = getattr(box, "size", None)
        sx = abs(float(getattr(size, "x", 0.0)))
        sy = abs(float(getattr(size, "y", 0.0)))
        obstacles.append(
            {
                "x": float(getattr(position, "x", 0.0)),
                "y": float(getattr(position, "y", 0.0)),
                "z": float(getattr(position, "z", 0.0)),
                "size_x": sx,
                "size_y": sy,
                "size_z": abs(float(getattr(size, "z", 0.0))),
                "radius": 0.5 * math.hypot(sx, sy),
            }
        )
    return {"count": len(obstacles), "obstacles": obstacles}


def _load_episode_json(episode_dir: Path) -> dict[str, Any] | None:
    path = episode_dir / "episode.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_entry(
    *,
    output_dir: Path,
    episode_dir: Path,
    seed: int,
    episode_index: int,
    goal: Goal,
    result: EpisodeResult,
    episode: dict[str, Any] | None,
) -> dict[str, Any]:
    entry = {
        "seed": int(seed),
        "episode_index": int(episode_index),
        "goal": goal.as_dict(),
        "episode_dir": str(episode_dir.relative_to(output_dir)),
        "status": result.status,
        "failure_reason": result.failure_reason,
    }
    if episode is not None:
        for key in (
            "duration_s",
            "start_pose",
            "end_pose",
            "initial_distance_m",
            "final_distance_m",
            "min_distance_m",
            "path_length_m",
            "progress_m",
            "progress_ratio",
            "odom_samples",
        ):
            if key in episode:
                entry[key] = episode[key]
    return entry


def _write_run_config(config: CollectorConfig, goals: list[Goal], path: Path) -> None:
    data = {
        "goals_path": str(config.goals_path),
        "goals": [goal.as_dict() for goal in goals],
        "seeds": [int(seed) for seed in config.seeds],
        "output_dir": str(config.output_dir),
        "max_goals_per_seed": config.max_goals_per_seed,
        "stop_on_failure": bool(config.stop_on_failure),
        "thresholds": asdict(config.thresholds),
        "profile": str(config.profile),
        "controller": str(config.controller),
        "headless": bool(config.headless),
        "launch_rviz": bool(config.launch_rviz),
        "obstacle_config": None if config.obstacle_config is None else str(config.obstacle_config),
        "topics": {
            "odom": config.odom_topic,
            "goal": config.goal_topic,
            "cmd": config.cmd_topic,
            "frontend_path": config.frontend_path_topic,
            "tltrajectory": config.tltrajectory_topic,
            "dynamic_obstacles": config.dynamic_obstacles_topic,
            "local_costmap": config.local_costmap_topic,
        },
        "local_costmap_sample_hz": float(config.local_costmap_sample_hz),
        "controller_odom_timeout_s": float(config.controller_odom_timeout_s),
        "reach_dwell_s": float(config.reach_dwell_s),
        "readiness_timeout_odom_s": float(config.readiness_timeout_odom_s),
        "readiness_timeout_goal_subscriber_s": float(config.readiness_timeout_goal_subscriber_s),
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
