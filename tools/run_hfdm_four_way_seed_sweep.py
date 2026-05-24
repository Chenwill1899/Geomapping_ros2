#!/usr/bin/env python3
"""Run 5-seed four-way MPPI/H-FDM trajectory comparison and plot overlays."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "mppi_tuning"
DEFAULT_GOAL = (18.0, 5.0)
DEFAULT_SEEDS = [424242, 424243, 424244, 424245, 424246]


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    profile: Path
    controller: str
    use_frontend: bool
    color: str
    linestyle: str


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _latest_export() -> Path:
    exports = sorted(
        (ROOT / "results" / "hfdm_training").glob("*/export/fdm_ts.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not exports:
        raise FileNotFoundError("no H-FDM export found under results/hfdm_training/*/export/fdm_ts.pt")
    return exports[0].parent


def _write_profiles(output_dir: Path, model_dir: Path) -> tuple[Path, Path]:
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    hfdm_src = ROOT / "src" / "mppi_controller" / "configs" / "mujoco_rviz_goal_hfdm_h25.yaml"
    base_config = ROOT / "src" / "mppi_controller" / "configs" / "mujoco_external_path_base.yaml"
    frontend_cfg = yaml.safe_load(hfdm_src.read_text(encoding="utf-8"))
    frontend_cfg.setdefault("experiment", {})["base_config"] = str(base_config)
    _set_hfdm_model_dir(frontend_cfg, model_dir)
    frontend_path = profile_dir / "hfdm_frontend_latest.yaml"
    frontend_path.write_text(yaml.safe_dump(frontend_cfg, sort_keys=False), encoding="utf-8")

    no_frontend_cfg = yaml.safe_load(hfdm_src.read_text(encoding="utf-8"))
    no_frontend_cfg.setdefault("experiment", {})["base_config"] = str(base_config)
    _set_hfdm_model_dir(no_frontend_cfg, model_dir)
    no_frontend_cfg.setdefault("external_path", {})["enabled"] = False
    no_frontend_cfg.setdefault("global_path", {})["enabled"] = False
    no_frontend_cfg.setdefault("final_controller", {})["disable_when_local_costmap"] = True
    mppi = no_frontend_cfg.setdefault("mppi", {})
    mppi["path_tracking_weight"] = 0.0
    mppi["path_progress_weight"] = 0.0
    no_frontend_path = profile_dir / "hfdm_no_frontend_latest.yaml"
    no_frontend_path.write_text(yaml.safe_dump(no_frontend_cfg, sort_keys=False), encoding="utf-8")
    return frontend_path, no_frontend_path


def _set_hfdm_model_dir(cfg: dict[str, Any], model_dir: Path) -> None:
    for controller in cfg.get("controllers", []):
        fdm = controller.get("fdm")
        if isinstance(fdm, dict):
            fdm["model_dir"] = str(model_dir.resolve())
            fdm["model_file"] = "fdm_ts.pt"
            fdm["metadata_file"] = "fdm_metadata.json"


def _conditions(hfdm_frontend: Path, hfdm_no_frontend: Path) -> list[Condition]:
    return [
        Condition(
            key="frontend_no_fdm",
            label="frontend + nominal MPPI",
            profile=ROOT / "src" / "mppi_controller" / "configs" / "mujoco_rviz_goal.yaml",
            controller="nominal_cuda",
            use_frontend=True,
            color="#2563eb",
            linestyle="-",
        ),
        Condition(
            key="frontend_hfdm",
            label="frontend + H-FDM MPPI",
            profile=hfdm_frontend,
            controller="learned_hfdm_h25",
            use_frontend=True,
            color="#dc2626",
            linestyle="-",
        ),
        Condition(
            key="no_frontend_no_fdm",
            label="no frontend + nominal MPPI",
            profile=ROOT / "src" / "mppi_controller" / "configs" / "mujoco_rviz_goal_no_frontend.yaml",
            controller="nominal_cuda",
            use_frontend=False,
            color="#0f766e",
            linestyle="--",
        ),
        Condition(
            key="no_frontend_hfdm",
            label="no frontend + H-FDM MPPI",
            profile=hfdm_no_frontend,
            controller="learned_hfdm_h25",
            use_frontend=False,
            color="#9333ea",
            linestyle="--",
        ),
    ]


def _run_trial(condition: Condition, *, seed: int, x: float, y: float, timeout_s: float, goal_tolerance_m: float) -> Path:
    tag = f"hfdm150_seed{seed}_{condition.key}"
    before = set(RESULTS_ROOT.glob(f"*_{tag}"))
    cmd = [
        sys.executable,
        "tools/geomapping_nav_trial.py",
        "--x",
        str(x),
        "--y",
        str(y),
        "--yaw",
        "0.0",
        "--tag",
        tag,
        "--profile",
        str(condition.profile),
        "--controller",
        condition.controller,
        "--scene-seed",
        str(seed),
        "--timeout-s",
        str(timeout_s),
        "--goal-tolerance-m",
        str(goal_tolerance_m),
        "--headless",
        "--no-launch-rviz",
    ]
    cmd.append("--use-frontend" if condition.use_frontend else "--no-use-frontend")
    print(f"[four_way] seed={seed} condition={condition.key} start", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    after = set(RESULTS_ROOT.glob(f"*_{tag}"))
    created = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    if not created:
        created = sorted(after, key=lambda path: path.stat().st_mtime, reverse=True)
    if not created:
        raise RuntimeError(f"trial finished but no result directory matched tag {tag}")
    print(f"[four_way] seed={seed} condition={condition.key} output={created[0]}", flush=True)
    return created[0]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_odom(path: Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip().split(",")
        columns = {name: index for index, name in enumerate(header)}
        for line in stream:
            if not line.strip():
                continue
            values = line.rstrip("\n").split(",")
            rows.append((float(values[columns["t"]]), float(values[columns["x"]]), float(values[columns["y"]])))
    return rows


def _summarize_run(run_dir: Path, condition: Condition) -> dict[str, Any]:
    metrics = _read_json(run_dir / "metrics.json") or {}
    native = _read_json(run_dir / "native_run" / "summary.json") or {}
    return {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "label": condition.label,
        "launch_variant": "frontend" if condition.use_frontend else "no_frontend",
        "controller": condition.controller,
        "profile": str(condition.profile),
        "reached_goal": bool(native.get("reached_goal", metrics.get("reached", False))),
        "timed_out": bool(metrics.get("timed_out", False)),
        "error": metrics.get("error"),
        "arrival_time_s": native.get("arrival_time"),
        "wrapper_duration_s": metrics.get("duration_s"),
        "final_distance_m": native.get("final_distance", metrics.get("distance_to_goal_m")),
        "min_distance_to_goal_m": metrics.get("min_distance_to_goal_m"),
        "path_length_m": native.get("path_length", metrics.get("trajectory_length_m")),
        "mean_mppi_time_ms": native.get("mean_mppi_time_ms"),
        "max_mppi_time_ms": native.get("max_mppi_time_ms"),
        "odom_samples": len(_read_odom(run_dir / "odom.csv")),
    }


def _load_obstacles(seed_runs: dict[str, Path]) -> list[dict[str, Any]]:
    for run_dir in seed_runs.values():
        obstacles = _read_json(run_dir / "obstacles.json")
        if isinstance(obstacles, list):
            return obstacles
    return []


def _draw_obstacles(ax: Any, obstacles: list[dict[str, Any]]) -> None:
    from matplotlib.patches import Circle, Rectangle

    for index, obstacle in enumerate(obstacles):
        x = float(obstacle.get("x", 0.0))
        y = float(obstacle.get("y", 0.0))
        color = str(obstacle.get("color", "#737373"))
        label = "obstacle" if index == 0 else None
        if str(obstacle.get("geom_type")) == "box":
            hx = float(obstacle.get("half_x", obstacle.get("radius", 0.0)))
            hy = float(obstacle.get("half_y", obstacle.get("radius", 0.0)))
            ax.add_patch(Rectangle((x - hx, y - hy), 2 * hx, 2 * hy, facecolor=color, edgecolor=color, alpha=0.20, label=label))
        else:
            radius = float(obstacle.get("radius", 0.0))
            ax.add_patch(Circle((x, y), radius=radius, facecolor=color, edgecolor=color, alpha=0.20, label=label))


def _plot_seed_overlay(
    *,
    seed: int,
    seed_runs: dict[str, Path],
    conditions: list[Condition],
    obstacles: list[dict[str, Any]],
    x: float,
    y: float,
    goal_tolerance_m: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=160)
    _draw_obstacles(ax, obstacles)
    all_x = [0.0, x]
    all_y = [0.0, y]
    for condition in conditions:
        run_dir = seed_runs.get(condition.key)
        if run_dir is None:
            continue
        odom = _read_odom(run_dir / "odom.csv")
        if not odom:
            continue
        xs = [row[1] for row in odom]
        ys = [row[2] for row in odom]
        all_x.extend(xs)
        all_y.extend(ys)
        ax.plot(xs, ys, color=condition.color, linestyle=condition.linestyle, linewidth=2.0, label=condition.label)
        ax.scatter(xs[-1], ys[-1], color=condition.color, s=28, marker="x")
    for obstacle in obstacles:
        all_x.append(float(obstacle.get("x", 0.0)))
        all_y.append(float(obstacle.get("y", 0.0)))
    ax.scatter([0.0], [0.0], color="#111827", s=42, marker="o", label="start")
    ax.scatter([x], [y], color="#16a34a", s=70, marker="*", label="goal")
    ax.add_patch(Circle((x, y), radius=goal_tolerance_m, fill=False, edgecolor="#16a34a", linestyle=":", linewidth=1.4))
    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Four-way MPPI/H-FDM trajectory comparison | scene seed {seed}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_grid(
    *,
    seeds: list[int],
    runs_by_seed: dict[int, dict[str, Path]],
    conditions: list[Condition],
    x: float,
    y: float,
    goal_tolerance_m: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(len(seeds), 1, figsize=(9.4, 3.1 * len(seeds)), dpi=150, squeeze=False)
    for row, seed in enumerate(seeds):
        ax = axes[row][0]
        seed_runs = runs_by_seed.get(seed, {})
        obstacles = _load_obstacles(seed_runs)
        _draw_obstacles(ax, obstacles)
        all_x = [0.0, x]
        all_y = [0.0, y]
        for condition in conditions:
            run_dir = seed_runs.get(condition.key)
            if run_dir is None:
                continue
            odom = _read_odom(run_dir / "odom.csv")
            if not odom:
                continue
            xs = [item[1] for item in odom]
            ys = [item[2] for item in odom]
            all_x.extend(xs)
            all_y.extend(ys)
            ax.plot(xs, ys, color=condition.color, linestyle=condition.linestyle, linewidth=1.8, label=condition.label)
        for obstacle in obstacles:
            all_x.append(float(obstacle.get("x", 0.0)))
            all_y.append(float(obstacle.get("y", 0.0)))
        ax.scatter([0.0], [0.0], color="#111827", s=30, marker="o")
        ax.scatter([x], [y], color="#16a34a", s=48, marker="*")
        ax.add_patch(Circle((x, y), radius=goal_tolerance_m, fill=False, edgecolor="#16a34a", linestyle=":", linewidth=1.0))
        ax.set_xlim(min(all_x) - 1.2, max(all_x) + 1.2)
        ax.set_ylim(min(all_y) - 1.2, max(all_y) + 1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.30)
        ax.set_title(f"scene seed {seed}", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if row == 0:
            ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--x", type=float, default=DEFAULT_GOAL[0])
    parser.add_argument("--y", type=float, default=DEFAULT_GOAL[1])
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.5)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    model_dir = (args.model_dir or _latest_export()).resolve()
    if not (model_dir / "fdm_ts.pt").exists():
        raise FileNotFoundError(f"missing {model_dir / 'fdm_ts.pt'}")
    if not (model_dir / "fdm_metadata.json").exists():
        raise FileNotFoundError(f"missing {model_dir / 'fdm_metadata.json'}")

    output_dir = args.output_dir or (RESULTS_ROOT / f"{_timestamp()}_hfdm150_four_way_5seeds")
    output_dir.mkdir(parents=True, exist_ok=False)
    hfdm_frontend, hfdm_no_frontend = _write_profiles(output_dir, model_dir)
    conditions = _conditions(hfdm_frontend, hfdm_no_frontend)
    seeds = list(args.seeds)

    runs_by_seed: dict[int, dict[str, Path]] = {}
    summary: dict[str, Any] = {
        "model_dir": str(model_dir),
        "goal": [float(args.x), float(args.y)],
        "goal_tolerance_m": float(args.goal_tolerance_m),
        "timeout_s": float(args.timeout_s),
        "seeds": seeds,
        "conditions": {condition.key: condition.label for condition in conditions},
        "runs": {},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for seed in seeds:
        runs_by_seed[seed] = {}
        summary["runs"][str(seed)] = {}
        for condition in conditions:
            run_dir = _run_trial(
                condition,
                seed=seed,
                x=float(args.x),
                y=float(args.y),
                timeout_s=float(args.timeout_s),
                goal_tolerance_m=float(args.goal_tolerance_m),
            )
            runs_by_seed[seed][condition.key] = run_dir
            summary["runs"][str(seed)][condition.key] = _summarize_run(run_dir, condition)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        overlay_path = output_dir / f"seed_{seed}_trajectory_overlay_four_way.png"
        _plot_seed_overlay(
            seed=seed,
            seed_runs=runs_by_seed[seed],
            conditions=conditions,
            obstacles=_load_obstacles(runs_by_seed[seed]),
            x=float(args.x),
            y=float(args.y),
            goal_tolerance_m=float(args.goal_tolerance_m),
            output_path=overlay_path,
        )
        summary["runs"][str(seed)]["overlay_png"] = str(overlay_path.relative_to(ROOT))
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    grid_path = output_dir / "trajectory_overlay_four_way_all_5seeds.png"
    _plot_grid(
        seeds=seeds,
        runs_by_seed=runs_by_seed,
        conditions=conditions,
        x=float(args.x),
        y=float(args.y),
        goal_tolerance_m=float(args.goal_tolerance_m),
        output_path=grid_path,
    )
    summary["overlay_grid_png"] = str(grid_path.relative_to(ROOT))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": str(output_dir / "summary.json"), "overlay_grid_png": str(grid_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
