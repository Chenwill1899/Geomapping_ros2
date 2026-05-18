#!/usr/bin/env python3
"""Aggregate Geomapping navigation trial directories into JSON and CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import yaml


RESULTS_ROOT = Path("/home/mexxiie/prj/Geomapping_ros2/results/mppi_tuning")


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _safe_tag(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "batch"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _seed_from_name(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values))


def _min(values: list[float]) -> float | None:
    if not values:
        return None
    return float(min(values))


def _resolve_run_dirs(run_dirs: list[str], patterns: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for value in run_dirs:
        path = Path(value).resolve()
        if path not in seen:
            resolved.append(path)
            seen.add(path)
    for pattern in patterns:
        for path in sorted(RESULTS_ROOT.glob(pattern)):
            path = path.resolve()
            if path not in seen:
                resolved.append(path)
                seen.add(path)
    return resolved


def _collect_run(path: Path) -> dict[str, Any]:
    summary = _load_json(path / "summary.json") or {}
    metrics = _load_json(path / "metrics.json") or {}
    config = _load_yaml(path / "config.yaml")
    native = metrics.get("native_summary") if isinstance(metrics, dict) else None
    source = native if isinstance(native, dict) and native else summary
    return {
        "run_dir": str(path),
        "name": path.name,
        "seed": _seed_from_name(path),
        "reached_goal": bool(source.get("reached_goal", False)),
        "success": bool(source.get("success", False)),
        "arrival_time": source.get("arrival_time"),
        "path_length": source.get("path_length"),
        "control_jerk": source.get("control_jerk"),
        "final_distance": source.get("final_distance"),
        "lateral_usage": source.get("lateral_usage"),
        "goal": None if not isinstance(config, dict) else config.get("goal"),
        "scene_seed": None if not isinstance(config, dict) else config.get("scene_seed"),
        "error": None if not isinstance(metrics, dict) else metrics.get("error"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", default=[], help="One trial directory under results/mppi_tuning.")
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern relative to results/mppi_tuning, for example '*goal18_5_batch_seed*'.",
    )
    parser.add_argument("--tag", default=None, help="Optional suffix for the aggregate result directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional aggregate output directory. Defaults to results/mppi_tuning/<timestamp>_<tag>.",
    )
    args = parser.parse_args(argv)

    run_dirs = _resolve_run_dirs(run_dirs=list(args.run_dir), patterns=list(args.glob))
    if not run_dirs:
        raise SystemExit("no run directories matched")

    runs = [_collect_run(path) for path in run_dirs]
    arrival_times = [float(item["arrival_time"]) for item in runs if item["arrival_time"] is not None]
    path_lengths = [float(item["path_length"]) for item in runs if item["path_length"] is not None]
    control_jerks = [float(item["control_jerk"]) for item in runs if item["control_jerk"] is not None]
    final_distances = [float(item["final_distance"]) for item in runs if item["final_distance"] is not None]
    lateral_usages = [float(item["lateral_usage"]) for item in runs if item["lateral_usage"] is not None]
    success_count = sum(1 for item in runs if item["reached_goal"])

    output_dir = Path(args.output_dir).resolve() if args.output_dir else RESULTS_ROOT / f"{_timestamp()}_{_safe_tag(args.tag or 'batch_summary')}"
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "output_dir": str(output_dir),
        "count": len(runs),
        "success_count": success_count,
        "success_rate": float(success_count / len(runs)),
        "seed_list": [item["seed"] for item in runs],
        "arrival_time": {"mean": _mean(arrival_times), "max": _max(arrival_times), "min": _min(arrival_times)},
        "path_length": {"mean": _mean(path_lengths), "max": _max(path_lengths), "min": _min(path_lengths)},
        "control_jerk": {"mean": _mean(control_jerks), "max": _max(control_jerks), "min": _min(control_jerks)},
        "final_distance": {"mean": _mean(final_distances), "max": _max(final_distances), "min": _min(final_distances)},
        "lateral_usage": {"mean": _mean(lateral_usages), "max": _max(lateral_usages), "min": _min(lateral_usages)},
        "runs": runs,
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = output_dir / "batch_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "name",
                "seed",
                "reached_goal",
                "success",
                "arrival_time",
                "path_length",
                "control_jerk",
                "final_distance",
                "lateral_usage",
                "goal",
                "scene_seed",
                "error",
                "run_dir",
            ],
        )
        writer.writeheader()
        for row in runs:
            writer.writerow(row)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
