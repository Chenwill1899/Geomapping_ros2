#!/usr/bin/env python3
"""Run no-frontend H-FDM tuning candidates on selected seeds."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import run_hfdm_four_way_seed_sweep as four_way


DEFAULT_MODEL_DIR = (
    four_way.ROOT
    / "results"
    / "hfdm_training"
    / "geomapping_data1_150_h25_20260523_221211"
    / "export_cuda_trace"
)


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    mppi: dict[str, Any]
    command_filter: dict[str, Any] | None = None


CANDIDATES = [
    Candidate(
        key="nf_micro_cost",
        label="no frontend H-FDM micro cost regularization",
        mppi={
            "smooth_weight": 0.10,
            "accel_weight": 0.015,
            "lateral_weight": 0.025,
            "yaw_rate_weight": 0.030,
            "jerk_weight": 0.035,
            "update_smoothing_alpha": [0.02, 0.08, 0.08],
        },
    ),
    Candidate(
        key="nf_yaw_lateral",
        label="no frontend H-FDM yaw/lateral regularization",
        mppi={
            "smooth_weight": 0.06,
            "accel_weight": 0.0,
            "lateral_weight": 0.070,
            "yaw_rate_weight": 0.070,
            "jerk_weight": 0.020,
            "update_smoothing_alpha": [0.0, 0.06, 0.08],
        },
    ),
    Candidate(
        key="nf_deadband_light",
        label="no frontend H-FDM light deadband/filter",
        mppi={
            "smooth_weight": 0.05,
            "accel_weight": 0.0,
            "lateral_weight": 0.020,
            "yaw_rate_weight": 0.020,
            "jerk_weight": 0.020,
            "update_smoothing_alpha": [0.0, 0.0, 0.0],
        },
        command_filter={
            "enabled": True,
            "alpha": 0.06,
            "max_ax": 1.0,
            "max_ay": 0.45,
            "max_awz": 1.2,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.015,
            "yaw_deadband": 0.015,
        },
    ),
    Candidate(
        key="nf_filter_soft",
        label="no frontend H-FDM soft execution filter",
        mppi={
            "smooth_weight": 0.0,
            "accel_weight": 0.0,
            "lateral_weight": 0.0,
            "yaw_rate_weight": 0.0,
            "jerk_weight": 0.0,
            "update_smoothing_alpha": [0.0, 0.0, 0.0],
        },
        command_filter={
            "enabled": True,
            "alpha": 0.03,
            "max_ax": 2.0,
            "max_ay": 1.0,
            "max_awz": 2.5,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.010,
            "yaw_deadband": 0.010,
        },
    ),
    Candidate(
        key="nf_yaw_light_filter",
        label="no frontend H-FDM light yaw/lateral with soft filter",
        mppi={
            "smooth_weight": 0.03,
            "accel_weight": 0.0,
            "lateral_weight": 0.040,
            "yaw_rate_weight": 0.040,
            "jerk_weight": 0.010,
            "update_smoothing_alpha": [0.0, 0.03, 0.04],
        },
        command_filter={
            "enabled": True,
            "alpha": 0.02,
            "max_ax": 2.0,
            "max_ay": 1.0,
            "max_awz": 2.5,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.008,
            "yaw_deadband": 0.008,
        },
    ),
    Candidate(
        key="nf_direct_light_filter",
        label="no frontend H-FDM direct path with light filter",
        mppi={
            "smooth_weight": 0.02,
            "accel_weight": 0.0,
            "lateral_weight": 0.030,
            "yaw_rate_weight": 0.030,
            "jerk_weight": 0.008,
            "update_smoothing_alpha": [0.0, 0.015, 0.025],
            "goal_progress_weight": 84.0,
            "heading_to_goal_weight": 0.20,
        },
        command_filter={
            "enabled": True,
            "alpha": 0.01,
            "max_ax": 2.4,
            "max_ay": 1.2,
            "max_awz": 3.0,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.004,
            "yaw_deadband": 0.004,
        },
    ),
    Candidate(
        key="nf_silk_light_filter",
        label="no frontend H-FDM silk path with light filter",
        mppi={
            "smooth_weight": 0.025,
            "accel_weight": 0.0,
            "lateral_weight": 0.035,
            "yaw_rate_weight": 0.035,
            "jerk_weight": 0.018,
            "update_smoothing_alpha": [0.0, 0.02, 0.03],
            "goal_progress_weight": 78.0,
            "heading_to_goal_weight": 0.18,
        },
        command_filter={
            "enabled": True,
            "alpha": 0.015,
            "max_ax": 2.2,
            "max_ay": 1.1,
            "max_awz": 2.8,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.006,
            "yaw_deadband": 0.006,
        },
    ),
    Candidate(
        key="nf_crisp_silk_filter",
        label="no frontend H-FDM crisp silk filter",
        mppi={
            "smooth_weight": 0.02,
            "accel_weight": 0.0,
            "lateral_weight": 0.035,
            "yaw_rate_weight": 0.035,
            "jerk_weight": 0.012,
            "update_smoothing_alpha": [0.0, 0.015, 0.03],
            "goal_progress_weight": 82.0,
            "heading_to_goal_weight": 0.18,
        },
        command_filter={
            "enabled": True,
            "alpha": 0.012,
            "max_ax": 2.4,
            "max_ay": 1.2,
            "max_awz": 3.0,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.005,
            "yaw_deadband": 0.005,
        },
    ),
    Candidate(
        key="nf_green_lowlag_filter",
        label="no frontend H-FDM green low-lag filter",
        mppi={
            "smooth_weight": 0.03,
            "accel_weight": 0.0,
            "lateral_weight": 0.040,
            "yaw_rate_weight": 0.040,
            "jerk_weight": 0.010,
            "update_smoothing_alpha": [0.0, 0.02, 0.03],
        },
        command_filter={
            "enabled": True,
            "alpha": 0.01,
            "max_ax": 2.4,
            "max_ay": 1.2,
            "max_awz": 3.0,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.004,
            "yaw_deadband": 0.004,
        },
    ),
    Candidate(
        key="nf_green_lowlag_jerk",
        label="no frontend H-FDM green low-lag jerk filter",
        mppi={
            "smooth_weight": 0.025,
            "accel_weight": 0.0,
            "lateral_weight": 0.040,
            "yaw_rate_weight": 0.040,
            "jerk_weight": 0.015,
            "update_smoothing_alpha": [0.0, 0.02, 0.035],
        },
        command_filter={
            "enabled": True,
            "alpha": 0.01,
            "max_ax": 2.4,
            "max_ay": 1.2,
            "max_awz": 3.0,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.004,
            "yaw_deadband": 0.004,
        },
    ),
    Candidate(
        key="nf_green_tight_noise",
        label="no frontend H-FDM green tight sampling noise",
        mppi={
            "std_normal": [0.45, 0.16, 0.22],
            "smooth_weight": 0.03,
            "accel_weight": 0.0,
            "lateral_weight": 0.040,
            "yaw_rate_weight": 0.040,
            "jerk_weight": 0.010,
            "update_smoothing_alpha": [0.0, 0.03, 0.04],
        },
        command_filter={
            "enabled": True,
            "alpha": 0.02,
            "max_ax": 2.0,
            "max_ay": 1.0,
            "max_awz": 2.5,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.008,
            "yaw_deadband": 0.008,
        },
    ),
    Candidate(
        key="nf_green_tight_noise_jerk",
        label="no frontend H-FDM green tight noise with jerk",
        mppi={
            "std_normal": [0.45, 0.16, 0.22],
            "smooth_weight": 0.025,
            "accel_weight": 0.0,
            "lateral_weight": 0.040,
            "yaw_rate_weight": 0.040,
            "jerk_weight": 0.015,
            "update_smoothing_alpha": [0.0, 0.03, 0.04],
        },
        command_filter={
            "enabled": True,
            "alpha": 0.02,
            "max_ax": 2.0,
            "max_ay": 1.0,
            "max_awz": 2.5,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.008,
            "yaw_deadband": 0.008,
        },
    ),
    Candidate(
        key="nf_green_mild_noise_jerk",
        label="no frontend H-FDM green mild noise with jerk",
        mppi={
            "std_normal": [0.50, 0.19, 0.25],
            "smooth_weight": 0.028,
            "accel_weight": 0.0,
            "lateral_weight": 0.040,
            "yaw_rate_weight": 0.040,
            "jerk_weight": 0.012,
            "update_smoothing_alpha": [0.0, 0.03, 0.04],
        },
        command_filter={
            "enabled": True,
            "alpha": 0.02,
            "max_ax": 2.0,
            "max_ay": 1.0,
            "max_awz": 2.5,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.008,
            "yaw_deadband": 0.008,
        },
    ),
    Candidate(
        key="nf_green_soft_cost",
        label="no frontend H-FDM green soft cost",
        mppi={
            "smooth_weight": 0.025,
            "accel_weight": 0.0,
            "lateral_weight": 0.035,
            "yaw_rate_weight": 0.035,
            "jerk_weight": 0.012,
            "update_smoothing_alpha": [0.0, 0.025, 0.035],
        },
        command_filter={
            "enabled": True,
            "alpha": 0.02,
            "max_ax": 2.0,
            "max_ay": 1.0,
            "max_awz": 2.5,
            "lateral_scale": 1.0,
            "lateral_deadband": 0.008,
            "yaw_deadband": 0.008,
        },
    ),
]


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _set_hfdm_model_dir(cfg: dict[str, Any], model_dir: Path) -> None:
    for controller in cfg.get("controllers", []):
        fdm = controller.get("fdm")
        if isinstance(fdm, dict):
            fdm["model_dir"] = str(model_dir.resolve())
            fdm["model_file"] = "fdm_ts.pt"
            fdm["metadata_file"] = "fdm_metadata.json"


def _candidate_profile(base: dict[str, Any], candidate: Candidate, model_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("experiment", {})["base_config"] = str(
        four_way.ROOT / "src" / "mppi_controller" / "configs" / "mujoco_external_path_base.yaml"
    )
    cfg.setdefault("external_path", {})["enabled"] = False
    cfg.setdefault("global_path", {})["enabled"] = False
    cfg.setdefault("final_controller", {})["disable_when_local_costmap"] = True
    cfg.setdefault("mppi", {}).update(candidate.mppi)
    cfg["mppi"]["path_tracking_weight"] = 0.0
    cfg["mppi"]["path_progress_weight"] = 0.0
    cfg["mppi"]["path_tracking_tolerance"] = 0.20
    if candidate.command_filter is not None:
        cfg.setdefault("command_filter", {}).update(candidate.command_filter)
    else:
        cfg.setdefault("command_filter", {})["enabled"] = False
    _set_hfdm_model_dir(cfg, model_dir)
    return cfg


def _select_candidates(keys: list[str] | None) -> list[Candidate]:
    if not keys:
        return CANDIDATES
    candidates_by_key = {candidate.key: candidate for candidate in CANDIDATES}
    unknown = sorted(set(keys) - set(candidates_by_key))
    if unknown:
        raise SystemExit(f"Unknown candidate(s): {', '.join(unknown)}")
    return [candidates_by_key[key] for key in keys]


def _write_profiles(output_dir: Path, model_dir: Path, candidates: list[Candidate]) -> dict[str, Path]:
    base_path = (
        four_way.ROOT
        / "results"
        / "mppi_tuning"
        / "20260524_112625_hfdm150_four_way_5seeds"
        / "profiles"
        / "hfdm_no_frontend_latest.yaml"
    )
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, Path] = {}
    for candidate in candidates:
        path = profile_dir / f"{candidate.key}.yaml"
        path.write_text(
            yaml.safe_dump(_candidate_profile(base, candidate, model_dir), sort_keys=False),
            encoding="utf-8",
        )
        profiles[candidate.key] = path
    return profiles


def _conditions(profiles: dict[str, Path], candidates: list[Candidate]) -> list[four_way.Condition]:
    colors = {
        "nf_micro_cost": "#7c3aed",
        "nf_yaw_lateral": "#dc2626",
        "nf_deadband_light": "#0891b2",
        "nf_filter_soft": "#ea580c",
        "nf_yaw_light_filter": "#16a34a",
        "nf_direct_light_filter": "#0f766e",
        "nf_silk_light_filter": "#2563eb",
        "nf_crisp_silk_filter": "#9333ea",
        "nf_green_lowlag_filter": "#22c55e",
        "nf_green_lowlag_jerk": "#65a30d",
        "nf_green_tight_noise": "#15803d",
        "nf_green_tight_noise_jerk": "#84cc16",
        "nf_green_mild_noise_jerk": "#4d7c0f",
        "nf_green_soft_cost": "#059669",
    }
    return [
        four_way.Condition(
            key=candidate.key,
            label=candidate.label,
            profile=profiles[candidate.key],
            controller="learned_hfdm_h25",
            use_frontend=False,
            color=colors[candidate.key],
            linestyle="--",
        )
        for candidate in candidates
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=[424242, 424243])
    parser.add_argument("--x", type=float, default=four_way.DEFAULT_GOAL[0])
    parser.add_argument("--y", type=float, default=four_way.DEFAULT_GOAL[1])
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.5)
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Candidate keys to run. Defaults to all candidates.",
    )
    args = parser.parse_args()

    candidates = _select_candidates(args.candidates)
    model_dir = args.model_dir.resolve()
    output_dir = (args.output_dir or four_way.RESULTS_ROOT / f"{_timestamp()}_hfdm_no_frontend_tuning").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = _write_profiles(output_dir, model_dir, candidates)
    conditions = _conditions(profiles, candidates)

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary: dict[str, Any] = {
            "model_dir": str(model_dir),
            "goal": [float(args.x), float(args.y)],
            "goal_tolerance_m": float(args.goal_tolerance_m),
            "timeout_s": float(args.timeout_s),
            "seeds": list(args.seeds),
            "conditions": {condition.key: condition.label for condition in conditions},
            "runs": {},
        }
    summary.setdefault("conditions", {}).update({condition.key: condition.label for condition in conditions})
    summary["seeds"] = list(args.seeds)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    runs_by_seed: dict[int, dict[str, Path]] = {}
    for seed in args.seeds:
        runs_by_seed[seed] = {}
        summary.setdefault("runs", {}).setdefault(str(seed), {})
        for condition in conditions:
            existing = summary["runs"][str(seed)].get(condition.key)
            run_dir = None
            if isinstance(existing, dict) and existing.get("run_dir"):
                candidate = four_way.ROOT / str(existing["run_dir"])
                if (candidate / "native_run" / "summary.json").exists():
                    run_dir = candidate
                    print(f"[hfdm_tune] seed={seed} condition={condition.key} skip existing={run_dir}", flush=True)
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
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        overlay_path = output_dir / f"seed_{seed}_trajectory_overlay_hfdm_no_frontend_tuning.png"
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
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
