#!/usr/bin/env python3
"""Run one ausim2 + Geomapping navigation trial and package artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


WORKSPACE_ROOT = Path("/home/mexxiie/prj")
GEOMAPPING_ROOT = WORKSPACE_ROOT / "Geomapping_ros2"
AUSIM_ROOT = WORKSPACE_ROOT / "ausim2"
DEFAULT_PROFILE = GEOMAPPING_ROOT / "src" / "mppi_controller" / "configs" / "mujoco_rviz_goal.yaml"
DEFAULT_OBSTACLE_CONFIG = GEOMAPPING_ROOT / "src" / "mppi_controller" / "configs" / "obstacle_scout_sparse.yaml"
RESULTS_ROOT = GEOMAPPING_ROOT / "results" / "mppi_tuning"
CONTROLLER_EXECUTABLE = GEOMAPPING_ROOT / "install" / "mppi_controller" / "lib" / "mppi_controller" / "fdm_mppi"


@dataclass
class OdomSample:
    t: float
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float


@dataclass
class PathSample:
    t: float
    pose_count: int
    length_m: float


class TrialMonitor:
    def __init__(self, *, goal_topic: str, odom_topic: str) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from nav_msgs.msg import Path as NavPath
        from rclpy.node import Node

        self._rclpy = rclpy
        rclpy.init(args=None)
        self.node = Node("geomapping_nav_trial_monitor")
        self.goal_topic = goal_topic
        self.odom_topic = odom_topic
        self.publisher = self.node.create_publisher(PoseStamped, self.goal_topic, 10)
        self._odom_sub = self.node.create_subscription(Odometry, self.odom_topic, self._on_odom, 50)
        self._smooth_path_sub = self.node.create_subscription(NavPath, "/smooth_path", self._on_smooth_path, 10)
        self.start_time = time.monotonic()
        self.last_odom_time: float | None = None
        self.samples: list[OdomSample] = []
        self.smooth_path_samples: list[PathSample] = []

    def _on_odom(self, message: Any) -> None:
        pose = message.pose.pose
        twist = message.twist.twist
        self.last_odom_time = time.monotonic()
        self.samples.append(
            OdomSample(
                t=self.last_odom_time - self.start_time,
                x=float(pose.position.x),
                y=float(pose.position.y),
                yaw=_yaw_from_quaternion(pose.orientation),
                vx=float(twist.linear.x),
                vy=float(twist.linear.y),
                wz=float(twist.angular.z),
            )
        )

    def _on_smooth_path(self, message: Any) -> None:
        poses = list(message.poses)
        length_m = 0.0
        for prev, curr in zip(poses[:-1], poses[1:]):
            dx = float(curr.pose.position.x) - float(prev.pose.position.x)
            dy = float(curr.pose.position.y) - float(prev.pose.position.y)
            length_m += math.hypot(dx, dy)
        self.smooth_path_samples.append(
            PathSample(
                t=time.monotonic() - self.start_time,
                pose_count=len(poses),
                length_m=length_m,
            )
        )

    def spin_once(self, timeout: float = 0.1) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=float(timeout))

    def has_odom(self) -> bool:
        return self.last_odom_time is not None

    def subscription_count(self) -> int:
        return int(self.publisher.get_subscription_count())

    def publish_goal(self, x: float, y: float, yaw: float) -> None:
        from geometry_msgs.msg import PoseStamped

        message = PoseStamped()
        message.header.frame_id = "map"
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.pose.position.x = float(x)
        message.pose.position.y = float(y)
        qx, qy, qz, qw = _yaw_to_quaternion(yaw)
        message.pose.orientation.x = qx
        message.pose.orientation.y = qy
        message.pose.orientation.z = qz
        message.pose.orientation.w = qw
        self.publisher.publish(message)

    def shutdown(self) -> None:
        try:
            self.node.destroy_node()
        finally:
            if self._rclpy.ok():
                self._rclpy.shutdown()


def _yaw_from_quaternion(quaternion: Any) -> float:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = float(yaw) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _safe_tag(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "trial"


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("ROS_LOG_DIR", "/tmp/ros_logs")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
    env.setdefault("PYTHONUNBUFFERED", "1")
    Path(env["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _source_and_exec(command: str) -> list[str]:
    ausim_ros_setup = AUSIM_ROOT / "build" / "ros_ws" / "install" / "setup.bash"
    ausim_source = f"source {ausim_ros_setup} && " if ausim_ros_setup.exists() else ""
    return [
        "bash",
        "-lc",
        (
            "source /opt/ros/humble/setup.bash && "
            f"{ausim_source}"
            f"source {GEOMAPPING_ROOT / 'install' / 'setup.bash'} && "
            f"exec {command}"
        ),
    ]


def _start_process(*, label: str, command: str, log_path: Path, env: dict[str, str]) -> tuple[subprocess.Popen[str], threading.Thread]:
    process = subprocess.Popen(
        _source_and_exec(command),
        cwd=str(WORKSPACE_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    def _forward() -> None:
        assert process.stdout is not None
        with log_path.open("a", encoding="utf-8") as stream:
            for line in process.stdout:
                stream.write(f"[{label}] {line}")
                stream.flush()

    thread = threading.Thread(target=_forward, name=f"{label}-log", daemon=True)
    thread.start()
    return process, thread


def _terminate_process(process: subprocess.Popen[str], *, sig: int = signal.SIGTERM, timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
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


def _compute_metrics(
    samples: list[OdomSample],
    smooth_path_samples: list[PathSample],
    *,
    goal_x: float,
    goal_y: float,
    goal_tolerance_m: float,
    timeout_s: float,
    launch_variant: str,
    mppi_profile: str,
    output_dir: Path,
    started: bool,
    error: str | None,
    native_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    distances = [math.hypot(goal_x - sample.x, goal_y - sample.y) for sample in samples]
    trajectory_length = 0.0
    speed_values: list[float] = []
    accel_values: list[float] = []
    jerk_values: list[float] = []
    yaw_rates: list[float] = []
    accel_vectors: list[tuple[float, float]] = []
    accel_wz: list[float] = []
    for sample in samples:
        speed_values.append(math.hypot(sample.vx, sample.vy))
        yaw_rates.append(sample.wz)
    for prev, curr in zip(samples[:-1], samples[1:]):
        dt = max(curr.t - prev.t, 1e-6)
        trajectory_length += math.hypot(curr.x - prev.x, curr.y - prev.y)
        ax = (curr.vx - prev.vx) / dt
        ay = (curr.vy - prev.vy) / dt
        awz = (curr.wz - prev.wz) / dt
        accel_vectors.append((ax, ay))
        accel_wz.append(awz)
        accel_values.append(math.sqrt(ax * ax + ay * ay + awz * awz))
    for (ax0, ay0), awz0, (ax1, ay1), awz1, prev, curr in zip(
        accel_vectors[:-1],
        accel_wz[:-1],
        accel_vectors[1:],
        accel_wz[1:],
        samples[1:-1],
        samples[2:],
    ):
        dt = max(curr.t - prev.t, 1e-6)
        jx = (ax1 - ax0) / dt
        jy = (ay1 - ay0) / dt
        jwz = (awz1 - awz0) / dt
        jerk_values.append(math.sqrt(jx * jx + jy * jy + jwz * jwz))

    reached = bool(native_summary.get("reached_goal")) if native_summary else False
    if not reached and distances:
        reached = min(distances) <= float(goal_tolerance_m)
    timed_out = bool(error == "timeout") and not bool(native_summary and native_summary.get("reached_goal"))
    duration_s = float(samples[-1].t) if samples else 0.0
    final_distance = float(distances[-1]) if distances else None
    min_distance = float(min(distances)) if distances else None
    return {
        "reached": reached,
        "distance_to_goal_m": final_distance,
        "min_distance_to_goal_m": min_distance,
        "duration_s": duration_s,
        "timed_out": timed_out,
        "num_samples": len(samples),
        "trajectory_length_m": trajectory_length,
        "speed": _stats(speed_values),
        "accel": _stats(accel_values),
        "smoothness": {
            "jerk_rms": _rms(jerk_values),
            "yaw_rate_rms": _rms(yaw_rates),
        },
        "smooth_path": {
            "samples": len(smooth_path_samples),
            "last_pose_count": smooth_path_samples[-1].pose_count if smooth_path_samples else 0,
            "last_length_m": smooth_path_samples[-1].length_m if smooth_path_samples else 0.0,
            "max_pose_count": max((sample.pose_count for sample in smooth_path_samples), default=0),
            "max_length_m": max((sample.length_m for sample in smooth_path_samples), default=0.0),
        },
        "output_dir": str(output_dir),
        "goal": [float(goal_x), float(goal_y)],
        "goal_yaw": None,
        "goal_tolerance_m": float(goal_tolerance_m),
        "timeout_s": float(timeout_s),
        "launch_variant": launch_variant,
        "mppi_profile": mppi_profile,
        "started": bool(started),
        "error": error,
        "native_summary_path": None if native_summary is None else str(output_dir / "native_run" / "summary.json"),
        "native_summary": native_summary,
    }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0}
    values_sorted = sorted(float(v) for v in values)
    return {
        "mean": float(sum(values_sorted) / len(values_sorted)),
        "max": float(values_sorted[-1]),
        "p95": float(values_sorted[min(len(values_sorted) - 1, int(round(0.95 * (len(values_sorted) - 1))))]),
    }


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(math.sqrt(sum(float(v) * float(v) for v in values) / len(values)))


def _write_odom(samples: list[OdomSample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("t,x,y,yaw,vx,vy,wz\n")
        for sample in samples:
            stream.write(
                f"{sample.t:.6f},{sample.x:.6f},{sample.y:.6f},{sample.yaw:.6f},"
                f"{sample.vx:.6f},{sample.vy:.6f},{sample.wz:.6f}\n"
            )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _write_trajectory_plots(
    samples: list[OdomSample],
    *,
    obstacles: list[dict[str, Any]],
    goal_x: float,
    goal_y: float,
    goal_tolerance_m: float,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    xs = [sample.x for sample in samples]
    ys = [sample.y for sample in samples]
    extent_x = xs + [float(goal_x)] + [float(item.get("x", 0.0)) for item in obstacles]
    extent_y = ys + [float(goal_y)] + [float(item.get("y", 0.0)) for item in obstacles]
    if not extent_x:
        extent_x = [0.0]
    if not extent_y:
        extent_y = [0.0]
    margin = 1.5
    x_min = min(extent_x) - margin
    x_max = max(extent_x) + margin
    y_min = min(extent_y) - margin
    y_max = max(extent_y) + margin

    def _draw(ax: Any, *, annotate: bool) -> None:
        for index, obstacle in enumerate(obstacles):
            color = str(obstacle.get("color", "#808080"))
            label = "obstacles" if index == 0 else None
            x = float(obstacle.get("x", 0.0))
            y = float(obstacle.get("y", 0.0))
            geom_type = str(obstacle.get("geom_type", ""))
            if geom_type == "box":
                half_x = float(obstacle.get("half_x", obstacle.get("radius", 0.0)))
                half_y = float(obstacle.get("half_y", obstacle.get("radius", 0.0)))
                patch = Rectangle(
                    (x - half_x, y - half_y),
                    2.0 * half_x,
                    2.0 * half_y,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.30,
                    linewidth=1.2,
                    label=label,
                )
            else:
                radius = float(obstacle.get("radius", 0.0))
                patch = Circle(
                    (x, y),
                    radius=radius,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.28,
                    linewidth=1.2,
                    label=label,
                )
            ax.add_patch(patch)

        if samples:
            ax.plot(xs, ys, color="#155eef", linewidth=2.2, label="trajectory")
            ax.scatter(xs[0], ys[0], s=42, color="#111827", marker="o", label="start")
            ax.scatter(xs[-1], ys[-1], s=42, color="#b42318", marker="x", label="end")
        ax.scatter(float(goal_x), float(goal_y), s=56, color="#16a34a", marker="*", label="goal")
        ax.add_patch(
            Circle(
                (float(goal_x), float(goal_y)),
                radius=float(goal_tolerance_m),
                fill=False,
                edgecolor="#16a34a",
                linestyle="--",
                linewidth=1.4,
                label="goal tolerance",
            )
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if annotate:
            path_length = sum(
                math.hypot(curr.x - prev.x, curr.y - prev.y) for prev, curr in zip(samples[:-1], samples[1:])
            )
            final_distance = math.hypot(goal_x - xs[-1], goal_y - ys[-1]) if samples else math.hypot(goal_x, goal_y)
            ax.set_title(
                f"Trajectory with obstacles | length={path_length:.2f} m | final distance={final_distance:.2f} m"
            )
            ax.legend(loc="best")

    figure, axis = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    _draw(axis, annotate=False)
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory.png", bbox_inches="tight")
    plt.close(figure)

    overlay_figure, overlay_axis = plt.subplots(figsize=(8.8, 5.4), dpi=150)
    _draw(overlay_axis, annotate=True)
    overlay_figure.tight_layout()
    overlay_figure.savefig(output_dir / "trajectory_overlay.png", bbox_inches="tight")
    plt.close(overlay_figure)


def _latest_dynamic_scene(start_time: float) -> Path | None:
    candidates = sorted(
        AUSIM_ROOT.glob("assets/*/scene.dynamic_obstacles.xml"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.stat().st_mtime >= start_time - 5.0:
            return candidate
    return candidates[0] if candidates else None


def _scene_to_obstacles(scene_path: Path) -> list[dict[str, Any]]:
    tree = ET.parse(scene_path)
    root = tree.getroot()
    obstacles: list[dict[str, Any]] = []
    for body in root.findall(".//body"):
        name = str(body.attrib.get("name", ""))
        if not name.startswith("dynamic_obs_"):
            continue
        pos = [float(value) for value in str(body.attrib.get("pos", "0 0 0")).split()]
        geom = body.find("geom")
        if geom is None:
            continue
        geom_type = str(geom.attrib.get("type", "unknown"))
        size = [float(value) for value in str(geom.attrib.get("size", "0 0 0")).split()]
        color = [float(value) for value in str(geom.attrib.get("rgba", "1 1 1 1")).split()]
        radius = float(size[0]) if size else 0.0
        half_x = float(size[0]) if len(size) >= 1 else 0.0
        half_y = float(size[1]) if len(size) >= 2 else half_x
        obstacles.append(
            {
                "name": name,
                "geom_type": geom_type,
                "x": float(pos[0]) if len(pos) >= 1 else 0.0,
                "y": float(pos[1]) if len(pos) >= 2 else 0.0,
                "radius": radius if geom_type == "cylinder" else math.hypot(half_x, half_y),
                "half_x": half_x,
                "half_y": half_y,
                "yaw": 0.0,
                "color": _rgba_to_hex(color),
            }
        )
    return obstacles


def _rgba_to_hex(values: list[float]) -> str:
    rgb = [max(0, min(255, int(round(channel * 255.0)))) for channel in values[:3]]
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_obstacle_config(
    *,
    output_dir: Path,
    scene_seed: int | None,
    obstacle_config: str | None,
) -> Path | None:
    if obstacle_config is None and scene_seed is None:
        return None

    source_path = Path(obstacle_config).expanduser().resolve() if obstacle_config else DEFAULT_OBSTACLE_CONFIG.resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"obstacle config not found: {source_path}")

    config = _load_yaml(source_path)
    if scene_seed is not None:
        config["random_seed"] = int(scene_seed)

    target_path = output_dir / "obstacle_config.yaml"
    target_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, required=True, help="Goal x in map frame.")
    parser.add_argument("--y", type=float, required=True, help="Goal y in map frame.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Goal yaw in rad.")
    parser.add_argument("--tag", default=None, help="Optional suffix for the result directory name.")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE), help="Closed-loop profile path.")
    parser.add_argument("--controller", default="nominal_cuda", help="Controller name from the profile.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="Run ausim2 headless.")
    parser.add_argument("--launch-rviz", action=argparse.BooleanOptionalAction, default=False, help="Launch RViz.")
    parser.add_argument("--goal-tolerance-m", type=float, default=0.3, help="Success tolerance for outer packaging.")
    parser.add_argument("--reach-dwell-s", type=float, default=0.5, help="Continuous in-tolerance time before stopping.")
    parser.add_argument("--timeout-s", type=float, default=180.0, help="Hard timeout for the whole trial.")
    parser.add_argument("--readiness-timeout-odom-s", type=float, default=60.0, help="Timeout waiting for first odom.")
    parser.add_argument("--readiness-timeout-goal-subscriber-s", type=float, default=60.0, help="Timeout waiting for goal subscribers.")
    parser.add_argument("--odom-topic", default="/scout1/odom", help="Odometry topic.")
    parser.add_argument("--goal-topic", default="/move_base_simple/goal", help="Goal topic.")
    parser.add_argument("--cmd-topic", default="/joy/cmd_vel", help="Command topic remapped into MPPI.")
    parser.add_argument("--scene-seed", type=int, default=None, help="Override dynamic obstacle random seed for this run.")
    parser.add_argument(
        "--obstacle-config",
        default=str(DEFAULT_OBSTACLE_CONFIG),
        help="Dynamic obstacle config used to override the ausim2 registry config.",
    )
    args = parser.parse_args(argv)

    profile_path = Path(args.profile).resolve()
    if not profile_path.exists():
        raise FileNotFoundError(f"profile not found: {profile_path}")

    sys.path.insert(0, str(GEOMAPPING_ROOT / "src" / "mppi_controller"))
    from mppi_controller.experiment import build_experiment_config

    config, _metadata = build_experiment_config(profile_path, controller_name=args.controller)
    sampling_rate = float(config["simulation"]["sampling_rate"])
    max_steps = max(1, int(math.ceil(float(args.timeout_s) * sampling_rate)))

    tag = args.tag or f"x{args.x:g}_y{args.y:g}_frontend"
    output_dir = RESULTS_ROOT / f"{_timestamp()}_{_safe_tag(tag)}"
    native_run_dir = output_dir / "native_run"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "trial.log"
    prepared_obstacle_config = _prepare_obstacle_config(
        output_dir=output_dir,
        scene_seed=args.scene_seed,
        obstacle_config=args.obstacle_config,
    )

    wrapper_config = {
        "timestamp": output_dir.name.split("_", 2)[0] + "_" + output_dir.name.split("_", 2)[1],
        "tag": _safe_tag(tag),
        "goal": {"x": float(args.x), "y": float(args.y), "yaw": None if args.yaw is None else float(args.yaw)},
        "goal_tolerance_m": float(args.goal_tolerance_m),
        "timeout_s": float(args.timeout_s),
        "launch_variant": "frontend",
        "mppi_profile": str(profile_path),
        "controller": str(args.controller),
        "headless": bool(args.headless),
        "launch_rviz": bool(args.launch_rviz),
        "reach_dwell_s": float(args.reach_dwell_s),
        "readiness_timeout_odom_s": float(args.readiness_timeout_odom_s),
        "readiness_timeout_goal_subscriber_s": float(args.readiness_timeout_goal_subscriber_s),
        "output_dir": str(output_dir),
        "native_run_dir": str(native_run_dir),
        "odom_topic": str(args.odom_topic),
        "goal_topic": str(args.goal_topic),
        "cmd_topic": str(args.cmd_topic),
        "max_steps": int(max_steps),
        "scene_seed": None if args.scene_seed is None else int(args.scene_seed),
        "obstacle_config": None if prepared_obstacle_config is None else str(prepared_obstacle_config),
    }
    (output_dir / "config.yaml").write_text(yaml.safe_dump(wrapper_config, sort_keys=False), encoding="utf-8")

    env = _env()
    if prepared_obstacle_config is not None:
        env["AUSIM_DYNAMIC_OBSTACLE_CONFIG_OVERRIDE"] = str(prepared_obstacle_config)
    ausim_cmd = f"{AUSIM_ROOT / 'em_run.sh'} {'--headless' if args.headless else ''}".strip()
    adapter_cmd = (
        "ros2 launch ausim_geomapping_adapter ausim_scout_mppi_frontend.launch.py "
        f"launch_rviz:={'true' if args.launch_rviz else 'false'} "
        "launch_mppi:=false "
        f"mppi_profile:={profile_path} "
        f"mppi_controller:={args.controller}"
    )
    controller_cmd = (
        f"{CONTROLLER_EXECUTABLE} mujoco-closed-loop "
        f"--profile {profile_path} "
        f"--controller {args.controller} "
        f"--results-dir {native_run_dir} "
        f"--max-steps {max_steps} "
        "--odom-timeout 2.0"
    )

    start_wall = time.time()
    ausim_proc = adapter_proc = controller_proc = None
    ausim_thread = adapter_thread = controller_thread = None
    monitor: TrialMonitor | None = None
    error: str | None = None
    started = False

    try:
        ausim_proc, ausim_thread = _start_process(label="ausim2", command=ausim_cmd, log_path=log_path, env=env)
        adapter_proc, adapter_thread = _start_process(label="adapter[frontend]", command=adapter_cmd, log_path=log_path, env=env)
        controller_proc, controller_thread = _start_process(label="mppi", command=controller_cmd, log_path=log_path, env=env)

        monitor = TrialMonitor(goal_topic=args.goal_topic, odom_topic=args.odom_topic)

        odom_deadline = time.monotonic() + float(args.readiness_timeout_odom_s)
        while time.monotonic() < odom_deadline:
            monitor.spin_once()
            if monitor.has_odom():
                break
            if any(proc is not None and proc.poll() is not None for proc in (ausim_proc, adapter_proc, controller_proc)):
                error = "startup process exited early"
                break
        if error is None and not monitor.has_odom():
            error = "odom readiness timeout"

        if error is None:
            goal_deadline = time.monotonic() + float(args.readiness_timeout_goal_subscriber_s)
            while time.monotonic() < goal_deadline:
                monitor.spin_once()
                if monitor.subscription_count() >= 2:
                    break
                if any(proc is not None and proc.poll() is not None for proc in (ausim_proc, adapter_proc, controller_proc)):
                    error = "goal subscriber exited early"
                    break
            if error is None and monitor.subscription_count() < 2:
                error = "goal subscriber readiness timeout"

        if error is None:
            started = True
            for _ in range(8):
                monitor.publish_goal(args.x, args.y, args.yaw)
                time.sleep(0.15)
                monitor.spin_once()

        reach_started_at: float | None = None
        hard_deadline = time.monotonic() + float(args.timeout_s)
        while error is None and time.monotonic() < hard_deadline:
            monitor.spin_once()
            if controller_proc is not None and controller_proc.poll() is not None:
                break
            if adapter_proc is not None and adapter_proc.poll() is not None:
                error = "adapter exited early"
                break
            if ausim_proc is not None and ausim_proc.poll() is not None:
                error = "ausim2 exited early"
                break
            native_summary = _load_json(native_run_dir / "summary.json")
            if native_summary is not None and bool(native_summary.get("reached_goal")):
                break
            if monitor.samples:
                latest = monitor.samples[-1]
                distance = math.hypot(args.x - latest.x, args.y - latest.y)
                if distance <= float(args.goal_tolerance_m):
                    if reach_started_at is None:
                        reach_started_at = time.monotonic()
                    elif time.monotonic() - reach_started_at >= float(args.reach_dwell_s):
                        break
                else:
                    reach_started_at = None
            time.sleep(0.02)

        if error is None and time.monotonic() >= hard_deadline:
            error = "timeout"

        if error is None and controller_proc is not None and controller_proc.poll() is not None:
            if controller_proc.returncode not in (0, None):
                error = f"controller exited with code {controller_proc.returncode}"

        time.sleep(1.0)
    finally:
        if controller_proc is not None:
            _terminate_process(controller_proc)
        if adapter_proc is not None:
            _terminate_process(adapter_proc)
        if ausim_proc is not None:
            _terminate_process(ausim_proc)
        for thread in (controller_thread, adapter_thread, ausim_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        if monitor is not None:
            monitor.shutdown()

    samples = monitor.samples if monitor is not None else []
    _write_odom(samples, output_dir / "odom.csv")

    native_summary = _load_json(native_run_dir / "summary.json")
    metrics = _compute_metrics(
        samples,
        monitor.smooth_path_samples if monitor is not None else [],
        goal_x=float(args.x),
        goal_y=float(args.y),
        goal_tolerance_m=float(args.goal_tolerance_m),
        timeout_s=float(args.timeout_s),
        launch_variant="frontend",
        mppi_profile=str(profile_path),
        output_dir=output_dir,
        started=started,
        error=error,
        native_summary=native_summary,
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _copy_if_exists(native_run_dir / "summary.json", output_dir / "summary.json")

    obstacles: list[dict[str, Any]] = []
    scene_path = _latest_dynamic_scene(start_wall)
    if scene_path is not None:
        copied_scene = output_dir / "scene.dynamic_obstacles.xml"
        shutil.copy2(scene_path, copied_scene)
        obstacles = _scene_to_obstacles(copied_scene)
        (output_dir / "obstacles.json").write_text(json.dumps(obstacles, indent=2), encoding="utf-8")

    _write_trajectory_plots(
        samples,
        obstacles=obstacles,
        goal_x=float(args.x),
        goal_y=float(args.y),
        goal_tolerance_m=float(args.goal_tolerance_m),
        output_dir=output_dir,
    )

    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
