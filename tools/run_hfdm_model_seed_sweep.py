#!/usr/bin/env python3
"""Run H-FDM frontend/no-frontend trials for a specific exported model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import run_hfdm_four_way_seed_sweep as four_way


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=four_way.DEFAULT_SEEDS)
    parser.add_argument("--x", type=float, default=four_way.DEFAULT_GOAL[0])
    parser.add_argument("--y", type=float, default=four_way.DEFAULT_GOAL[1])
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.5)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not (model_dir / "fdm_ts.pt").exists():
        raise FileNotFoundError(f"missing {model_dir / 'fdm_ts.pt'}")
    if not (model_dir / "fdm_metadata.json").exists():
        raise FileNotFoundError(f"missing {model_dir / 'fdm_metadata.json'}")

    output_dir = args.output_dir or (
        four_way.RESULTS_ROOT / f"{_timestamp()}_hfdm_model_two_way_5seeds"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hfdm_frontend, hfdm_no_frontend = four_way._write_profiles(output_dir, model_dir)
    conditions = [
        condition
        for condition in four_way._conditions(hfdm_frontend, hfdm_no_frontend)
        if condition.key in {"frontend_hfdm", "no_frontend_hfdm"}
    ]

    seeds = list(args.seeds)
    runs_by_seed: dict[int, dict[str, Path]] = {}
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary: dict[str, object] = {
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
        summary.setdefault("runs", {}).setdefault(str(seed), {})
        for condition in conditions:
            existing = summary["runs"][str(seed)].get(condition.key)
            run_dir = None
            if isinstance(existing, dict) and existing.get("run_dir"):
                candidate = four_way.ROOT / str(existing["run_dir"])
                if (candidate / "native_run" / "summary.json").exists():
                    run_dir = candidate
                    print(f"[hfdm_model] seed={seed} condition={condition.key} skip existing={run_dir}", flush=True)
            if run_dir is None:
                run_dir = four_way._run_trial(
                    condition,
                    seed=seed,
                    x=float(args.x),
                    y=float(args.y),
                    timeout_s=float(args.timeout_s),
                    goal_tolerance_m=float(args.goal_tolerance_m),
                )
            runs_by_seed[seed][condition.key] = run_dir
            summary["runs"][str(seed)][condition.key] = four_way._summarize_run(run_dir, condition)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        overlay_path = output_dir / f"seed_{seed}_trajectory_overlay_hfdm_two_way.png"
        four_way._plot_seed_overlay(
            seed=seed,
            seed_runs=runs_by_seed[seed],
            conditions=conditions,
            obstacles=four_way._load_obstacles(runs_by_seed[seed]),
            x=float(args.x),
            y=float(args.y),
            goal_tolerance_m=float(args.goal_tolerance_m),
            output_path=overlay_path,
        )
        summary["runs"][str(seed)]["overlay_png"] = str(overlay_path.resolve().relative_to(four_way.ROOT))
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    grid_path = output_dir / "trajectory_overlay_hfdm_two_way_all_5seeds.png"
    four_way._plot_grid(
        seeds=seeds,
        runs_by_seed=runs_by_seed,
        conditions=conditions,
        x=float(args.x),
        y=float(args.y),
        goal_tolerance_m=float(args.goal_tolerance_m),
        output_path=grid_path,
    )
    summary["overlay_grid_png"] = str(grid_path.resolve().relative_to(four_way.ROOT))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": str(output_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
