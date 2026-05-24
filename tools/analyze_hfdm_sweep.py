#!/usr/bin/env python3
"""Offline diagnostics for the H-FDM four-way MPPI sweep.

The script is intentionally read-only for run artifacts. It consumes the
four-way sweep summary, computes trajectory/control oscillation metrics, and
writes compact JSON/CSV/Markdown reports.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results" / "mppi_tuning" / "20260524_112625_hfdm150_four_way_5seeds" / "summary.json"
KEY_CONFIG_FIELDS = [
    "external_path.enabled",
    "global_path.enabled",
    "final_controller.disable_when_local_costmap",
    "local_costmap.cost_weight",
    "command_filter.enabled",
    "command_filter.alpha",
    "command_filter.lateral_deadband",
    "command_filter.yaw_deadband",
    "mppi.control_weight",
    "mppi.smooth_weight",
    "mppi.accel_weight",
    "mppi.lateral_weight",
    "mppi.yaw_rate_weight",
    "mppi.jerk_weight",
    "mppi.update_smoothing_alpha",
    "mppi.goal_progress_weight",
    "mppi.heading_to_goal_weight",
    "mppi.path_tracking_weight",
    "mppi.path_progress_weight",
    "mppi.learned_risk_weight",
    "mppi.learned_risk_threshold",
]
TUNED_NO_FRONTEND_SETTINGS = {
    "smooth_weight": 0.6,
    "accel_weight": 0.12,
    "lateral_weight": 0.12,
    "yaw_rate_weight": 0.12,
    "jerk_weight": 0.2,
    "update_smoothing_alpha": [0.12, 0.45, 0.45],
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _unwrap_delta(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _mean(values: Iterable[float]) -> float | None:
    data = list(values)
    if not data:
        return None
    return float(sum(data) / len(data))


def _path_length(points: list[tuple[float, float]]) -> float:
    return float(sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(points[:-1], points[1:])))


def _line_deviations(points: list[tuple[float, float]], goal: tuple[float, float]) -> list[float]:
    if not points:
        return []
    x0, y0 = points[0]
    x1, y1 = goal
    dx = x1 - x0
    dy = y1 - y0
    denom = math.hypot(dx, dy)
    if denom <= 1e-9:
        return [0.0 for _point in points]
    return [abs(dy * x - dx * y + x1 * y0 - y1 * x0) / denom for x, y in points]


def _total_heading_change(points: list[tuple[float, float]]) -> float:
    headings: list[float] = []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        dx = x1 - x0
        dy = y1 - y0
        if math.hypot(dx, dy) > 1e-4:
            headings.append(math.atan2(dy, dx))
    return float(sum(abs(_unwrap_delta(curr - prev)) for prev, curr in zip(headings[:-1], headings[1:])))


def _sign_switches(values: Iterable[float], *, eps: float = 1e-5) -> int:
    switches = 0
    previous: int | None = None
    for value in values:
        if abs(value) <= eps:
            continue
        sign = 1 if value > 0.0 else -1
        if previous is not None and sign != previous:
            switches += 1
        previous = sign
    return switches


def _trajectory_rows(run_dir: Path) -> tuple[str, list[dict[str, str]]]:
    native_rows = _read_csv(run_dir / "native_run" / "trajectory.csv")
    if native_rows:
        return "native_run/trajectory.csv", native_rows
    return "odom.csv", _read_csv(run_dir / "odom.csv")


def _planning_goal_stats(run_dir: Path, goal: tuple[float, float]) -> dict[str, Any]:
    rows = _read_csv(run_dir / "native_run" / "planning_goals.csv")
    unique = {
        (round(_float(row, "x"), 3), round(_float(row, "y"), 3), round(_float(row, "theta"), 3))
        for row in rows
    }
    final_goal = (round(float(goal[0]), 3), round(float(goal[1]), 3))
    is_constant = bool(unique) and all(item[:2] == final_goal for item in unique)
    return {
        "planning_goal_samples": len(rows),
        "planning_goal_unique_count": len(unique),
        "planning_goal_is_constant_final_goal": is_constant,
    }


def analyze_run(run_dir: Path, *, goal: tuple[float, float]) -> dict[str, Any]:
    """Compute shape/control diagnostics for one trial directory."""

    run_dir = Path(run_dir)
    source, rows = _trajectory_rows(run_dir)
    points = [(_float(row, "x"), _float(row, "y")) for row in rows]
    straight_distance = math.hypot(float(goal[0]) - points[0][0], float(goal[1]) - points[0][1]) if points else 0.0
    path_length = _path_length(points)
    deviations = _line_deviations(points, goal)
    vy_values = [_float(row, "vy") for row in rows if "vy" in row]
    wz_values = [_float(row, "wz") for row in rows if "wz" in row]

    control_rows = _read_csv(run_dir / "native_run" / "controls.csv")
    control_vy = [_float(row, "vy_cmd") for row in control_rows if "vy_cmd" in row]
    control_wz = [_float(row, "wz_cmd") for row in control_rows if "wz_cmd" in row]

    native_summary_path = run_dir / "native_run" / "summary.json"
    native_summary = _read_json(native_summary_path) if native_summary_path.exists() else {}

    metrics: dict[str, Any] = {
        "run_dir": str(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir),
        "trajectory_source": source,
        "trajectory_samples": len(rows),
        "reached_goal": bool(native_summary.get("reached_goal", False)),
        "arrival_time_s": native_summary.get("arrival_time"),
        "final_distance_m": native_summary.get("final_distance"),
        "path_length_m": path_length,
        "native_summary_path_length_m": native_summary.get("path_length"),
        "straight_distance_m": straight_distance,
        "straight_ratio": path_length / straight_distance if straight_distance > 1e-9 else None,
        "max_line_deviation_m": max(deviations) if deviations else None,
        "mean_line_deviation_m": _mean(deviations),
        "total_heading_change_rad": _total_heading_change(points),
        "odom_wz_sign_switches": _sign_switches(wz_values),
        "mean_abs_vy": _mean(abs(value) for value in vy_values),
        "mean_abs_wz": _mean(abs(value) for value in wz_values),
        "control_samples": len(control_rows),
        "control_wz_sign_switches": _sign_switches(control_wz),
        "mean_abs_cmd_vy": _mean(abs(value) for value in control_vy),
        "mean_abs_cmd_wz": _mean(abs(value) for value in control_wz),
        "control_smoothness": native_summary.get("control_smoothness"),
        "control_jerk": native_summary.get("control_jerk"),
        "lateral_usage": native_summary.get("lateral_usage"),
    }
    metrics.update(_planning_goal_stats(run_dir, goal))
    return metrics


def _resolve_run_dir(path: str) -> Path:
    run_dir = Path(path)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    return run_dir


def _get_nested(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def build_tuned_no_frontend_profile(source: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal stability-tuned no-frontend H-FDM profile candidate."""

    tuned = copy.deepcopy(source)
    tuned.setdefault("external_path", {})["enabled"] = False
    tuned.setdefault("global_path", {})["enabled"] = False
    tuned.setdefault("final_controller", {})["disable_when_local_costmap"] = True

    command_filter = tuned.setdefault("command_filter", {})
    command_filter["enabled"] = True
    command_filter["alpha"] = 0.18
    command_filter["lateral_deadband"] = 0.02
    command_filter["yaw_deadband"] = 0.02

    mppi = tuned.setdefault("mppi", {})
    mppi.update(TUNED_NO_FRONTEND_SETTINGS)
    mppi["path_tracking_weight"] = 0.0
    mppi["path_progress_weight"] = 0.0
    mppi["path_tracking_tolerance"] = 0.20
    return tuned


def profile_key_settings(summary: dict[str, Any]) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for seed_runs in summary.get("runs", {}).values():
        if not isinstance(seed_runs, dict):
            continue
        for condition, run_info in seed_runs.items():
            if not isinstance(run_info, dict) or "profile" not in run_info:
                continue
            profile = _resolve_run_dir(str(run_info["profile"]))
            key = (condition, str(profile))
            if key in seen:
                continue
            seen.add(key)
            cfg = _load_yaml(profile)
            row: dict[str, Any] = {"condition": condition, "profile": str(profile)}
            for field in KEY_CONFIG_FIELDS:
                row[field] = _get_nested(cfg, field)
            settings.append(row)
    return settings


def analyze_summary(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    goal_values = summary.get("goal") or [0.0, 0.0]
    goal = (float(goal_values[0]), float(goal_values[1]))
    rows: list[dict[str, Any]] = []
    for seed, seed_runs in summary.get("runs", {}).items():
        if not isinstance(seed_runs, dict):
            continue
        for condition, run_info in seed_runs.items():
            if not isinstance(run_info, dict) or "run_dir" not in run_info:
                continue
            run_dir = _resolve_run_dir(str(run_info["run_dir"]))
            metrics = analyze_run(run_dir, goal=goal)
            metrics.update(
                {
                    "seed": int(seed),
                    "condition": condition,
                    "label": run_info.get("label", condition),
                    "launch_variant": run_info.get("launch_variant"),
                    "controller": run_info.get("controller"),
                    "profile": run_info.get("profile"),
                    "summary_path_length_m": run_info.get("path_length_m"),
                }
            )
            rows.append(metrics)
    return {
        "summary_path": str(summary_path),
        "goal": list(goal),
        "conditions": summary.get("conditions", {}),
        "run_metrics": rows,
        "aggregate_metrics": aggregate_metrics(rows),
        "profile_key_settings": profile_key_settings(summary),
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    numeric_fields = [
        "arrival_time_s",
        "path_length_m",
        "straight_ratio",
        "max_line_deviation_m",
        "mean_line_deviation_m",
        "total_heading_change_rad",
        "odom_wz_sign_switches",
        "control_wz_sign_switches",
        "mean_abs_vy",
        "mean_abs_wz",
        "mean_abs_cmd_vy",
        "mean_abs_cmd_wz",
        "control_smoothness",
        "control_jerk",
        "lateral_usage",
    ]
    output: list[dict[str, Any]] = []
    for condition, items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "condition": condition,
            "label": items[0].get("label", condition),
            "runs": len(items),
            "reached": sum(1 for item in items if item.get("reached_goal")),
        }
        for field in numeric_fields:
            values = [float(item[field]) for item in items if item.get(field) is not None]
            row[f"mean_{field}"] = _mean(values)
        output.append(row)
    return output


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _write_tuned_profile_candidate(result: dict[str, Any], output_dir: Path) -> Path | None:
    profile_path: Path | None = None
    for row in result.get("profile_key_settings", []):
        if row.get("condition") == "no_frontend_hfdm":
            profile_path = _resolve_run_dir(str(row.get("profile")))
            break
    if profile_path is None or not profile_path.exists():
        return None

    source = _load_yaml(profile_path)
    tuned = build_tuned_no_frontend_profile(source)
    target = output_dir / "hfdm_no_frontend_tuned_stability.yaml"
    target.write_text(yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    return target


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tuned_profile = _write_tuned_profile_candidate(result, output_dir)
    if tuned_profile is not None:
        result["tuned_profile_candidate"] = str(tuned_profile)
    (output_dir / "diagnostics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "run_metrics.csv", result["run_metrics"])
    _write_csv(output_dir / "aggregate_metrics.csv", result["aggregate_metrics"])
    _write_csv(output_dir / "profile_key_settings.csv", result["profile_key_settings"])

    lines = [
        "# H-FDM Sweep Offline Diagnostics",
        "",
        f"- Source summary: `{result['summary_path']}`",
        f"- Goal: `{result['goal']}`",
        f"- Runs analyzed: `{len(result['run_metrics'])}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Condition | Reached | Path ratio | Max line dev | Heading change | Control wz switches | Mean abs vy | Mean abs wz | Arrival |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["aggregate_metrics"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["condition"]),
                    f"{row['reached']}/{row['runs']}",
                    _fmt(row.get("mean_straight_ratio")),
                    _fmt(row.get("mean_max_line_deviation_m")),
                    _fmt(row.get("mean_total_heading_change_rad")),
                    _fmt(row.get("mean_control_wz_sign_switches"), 1),
                    _fmt(row.get("mean_mean_abs_vy")),
                    _fmt(row.get("mean_mean_abs_wz")),
                    _fmt(row.get("mean_arrival_time_s"), 2),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Key Profile Settings",
            "",
            "| Condition | external path | global path | command filter | smooth | accel | lateral | yaw rate | jerk | update alpha | path weights | learned risk |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in result["profile_key_settings"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["condition"]),
                    _fmt(row.get("external_path.enabled")),
                    _fmt(row.get("global_path.enabled")),
                    _fmt(row.get("command_filter.enabled")),
                    _fmt(row.get("mppi.smooth_weight")),
                    _fmt(row.get("mppi.accel_weight")),
                    _fmt(row.get("mppi.lateral_weight")),
                    _fmt(row.get("mppi.yaw_rate_weight")),
                    _fmt(row.get("mppi.jerk_weight")),
                    _fmt(row.get("mppi.update_smoothing_alpha")),
                    f"{_fmt(row.get('mppi.path_tracking_weight'))}/{_fmt(row.get('mppi.path_progress_weight'))}",
                    f"{_fmt(row.get('mppi.learned_risk_weight'))}@{_fmt(row.get('mppi.learned_risk_threshold'))}",
                ]
            )
            + " |"
        )
    if tuned_profile is not None:
        lines.extend(
            [
                "",
                "## Tuned Profile Candidate",
                "",
                f"- File: `{tuned_profile}`",
                "- Scope: keeps H-FDM rollout, learned risk, no-frontend mode, and zero path tracking/progress weights.",
                "- Stability changes: `smooth_weight=0.6`, `accel_weight=0.12`, `lateral_weight=0.12`, `yaw_rate_weight=0.12`, `jerk_weight=0.2`, `update_smoothing_alpha=[0.12, 0.45, 0.45]`, command filter enabled with `alpha=0.18`.",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Four-way sweep summary.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for diagnostics reports")
    args = parser.parse_args(argv)

    summary_path = args.summary.resolve()
    output_dir = args.output_dir or (summary_path.parent / "diagnostics")
    result = analyze_summary(summary_path)
    write_report(result, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "runs": len(result["run_metrics"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
