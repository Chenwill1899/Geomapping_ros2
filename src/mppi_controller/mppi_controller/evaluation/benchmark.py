#!/usr/bin/env python3
"""Run paired Stage 5 closed-loop nominal vs learned-FDM MPPI benchmarks."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from mppi_controller.config import load_config
from mppi_controller.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller


RUN_METRICS = (
    "success",
    "failed",
    "final_distance",
    "steps",
    "arrival_time",
    "path_length",
    "min_obstacle_clearance",
    "mean_terrain_risk",
    "max_terrain_risk",
    "cumulative_terrain_risk",
    "terrain_risk_excess",
    "terrain_risk_excess_integral",
    "terrain_risk_exposure_ratio",
    "mean_cmd_real_error",
    "mean_residual_norm",
    "control_smoothness",
    "control_jerk",
    "mean_mppi_time_ms",
    "max_mppi_time_ms",
)

NUMERIC_METRICS = (
    "final_distance",
    "steps",
    "arrival_time",
    "path_length",
    "min_obstacle_clearance",
    "mean_terrain_risk",
    "max_terrain_risk",
    "cumulative_terrain_risk",
    "terrain_risk_excess",
    "terrain_risk_excess_integral",
    "terrain_risk_exposure_ratio",
    "mean_cmd_real_error",
    "mean_residual_norm",
    "control_smoothness",
    "control_jerk",
    "mean_mppi_time_ms",
    "max_mppi_time_ms",
)

VALID_CONTROLLERS = ("nominal", "learned")
LEARNED_MPPI_OVERRIDE_KEYS = (
    "goal_xy_weight",
    "obstacle_weight",
    "obstacle_soft_weight",
    "smooth_weight",
    "accel_weight",
    "lateral_weight",
    "yaw_rate_weight",
)
MPPI_OVERRIDE_KEYS = (
    "terrain_risk_weight",
    "terrain_risk_power",
    "terrain_risk_threshold",
    "terrain_risk_mode",
)
MPPI_STRING_OVERRIDE_KEYS = {"terrain_risk_mode"}


def run_benchmark(
    *,
    config_path: str | Path,
    scenario_name: str,
    output_dir: str | Path,
    episodes: int,
    base_seed: int,
    backend: str | None = "torch",
    controllers: Sequence[str] = VALID_CONTROLLERS,
    fdm_model_dir: str | Path = "results/fdm_baselines/stage4_mlp_seed123_hardened",
    fdm_checkpoint: str | Path = "best_model.pt",
    fdm_normalization: str | Path = "normalization.npz",
    fdm_device: str | None = None,
    mppi_overrides: dict[str, float | str] | None = None,
    learned_mppi_overrides: dict[str, float] | None = None,
    command: str | None = None,
    argv: Sequence[str] | None = None,
    runner_cls=OmniMppiSimulationRunner,
) -> dict:
    config_path = Path(config_path)
    base_config = load_config(config_path)
    if backend is None:
        backend = str(base_config.get("mppi", {}).get("backend", "numpy"))
    backend = str(backend).lower()
    if backend not in {"numpy", "cuda", "torch"}:
        raise ValueError("Stage 5 benchmark supports only numpy, cuda, or torch backends")
    fdm_device = fdm_device or ("cuda" if backend == "cuda" else "cpu")
    controllers = tuple(_validate_controllers(controllers))
    mppi_overrides = _validate_mppi_overrides(mppi_overrides or {})
    learned_mppi_overrides = _validate_learned_mppi_overrides(learned_mppi_overrides or {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(base_seed) + episode_id for episode_id in range(int(episodes))]

    runs = []
    for episode_id, seed in enumerate(seeds):
        for controller in controllers:
            run_config = prepare_run_config(
                base_config,
                output_dir=output_dir,
                scenario_name=scenario_name,
                controller=controller,
                episode_id=episode_id,
                seed=seed,
                backend=backend,
                fdm_model_dir=fdm_model_dir,
                fdm_checkpoint=fdm_checkpoint,
                fdm_normalization=fdm_normalization,
                fdm_device=fdm_device,
                mppi_overrides=mppi_overrides,
                learned_mppi_overrides=learned_mppi_overrides,
            )
            runner = runner_cls(
                run_config,
                controller_factory=lambda *, config, runner, seed=seed: create_omni_controller(
                    config,
                    seed=seed,
                ),
            )
            summary = runner.run()
            runs.append(
                _run_record(
                    summary_path=Path(summary.results_path) / "summary.json",
                    scenario=scenario_name,
                    controller=controller,
                    episode_id=episode_id,
                    seed=seed,
                    results_path=Path(summary.results_path),
                )
            )

    benchmark_summary = {
        "metadata": {
            "command": command,
            "argv": [str(item) for item in argv] if argv is not None else None,
            "config": str(config_path),
            "scenario_name": str(scenario_name),
            "output_dir": str(output_dir),
            "backend": backend,
            "controllers": list(controllers),
            "episodes": int(episodes),
            "base_seed": int(base_seed),
            "seeds": seeds,
            "fdm_model_dir": str(fdm_model_dir),
            "fdm_checkpoint": str(fdm_checkpoint),
            "fdm_normalization": str(fdm_normalization),
            "fdm_device": str(fdm_device),
            "mppi_overrides": dict(mppi_overrides),
            "learned_mppi_overrides": dict(learned_mppi_overrides),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            **current_git_metadata(),
        },
        "runs": runs,
        "aggregates": aggregate_runs(runs),
        "paired_deltas": compute_paired_deltas(runs),
    }
    (output_dir / "stage5_benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2),
        encoding="utf-8",
    )
    return benchmark_summary


def prepare_run_config(
    base_config: dict,
    *,
    output_dir: Path,
    scenario_name: str,
    controller: str,
    episode_id: int,
    seed: int,
    backend: str,
    fdm_model_dir: str | Path,
    fdm_checkpoint: str | Path,
    fdm_normalization: str | Path,
    fdm_device: str,
    mppi_overrides: dict[str, float | str] | None = None,
    learned_mppi_overrides: dict[str, float] | None = None,
) -> dict:
    if controller not in VALID_CONTROLLERS:
        raise ValueError(f"Unsupported controller '{controller}'. Expected one of {VALID_CONTROLLERS}")
    config = copy.deepcopy(base_config)
    config.setdefault("simulation", {})["world_mode"] = "oracle"
    config.setdefault("mppi", {})["backend"] = str(backend).lower()
    if str(backend).lower() == "torch":
        config.setdefault("mppi", {})["device"] = str(fdm_device)
    _apply_mppi_overrides(config, mppi_overrides or {})
    _apply_episode_seed(config, seed)
    safe_scenario = _safe_name(scenario_name)
    config["results"] = {
        **config.get("results", {}),
        "root": str(output_dir / "runs"),
        "run_name": f"{safe_scenario}_episode_{int(episode_id):04d}_{controller}",
        "timestamp_suffix": False,
        "overwrite": True,
        "enable_plots": False,
        "enable_animation": False,
    }
    if controller == "learned":
        _apply_learned_mppi_overrides(config, learned_mppi_overrides or {})
        config["fdm"] = {
            **config.get("fdm", {}),
            "enabled": True,
            "model_dir": str(fdm_model_dir),
            "checkpoint": str(fdm_checkpoint),
            "normalization": str(fdm_normalization),
            "device": str(fdm_device),
            "mode": "sequence",
        }
    else:
        config.pop("fdm", None)
    return config


def aggregate_runs(runs: Sequence[dict]) -> dict:
    aggregates: dict[str, dict] = {}
    for controller in sorted({str(run["controller"]) for run in runs}):
        subset = [run for run in runs if run.get("controller") == controller]
        success_count = sum(1 for run in subset if bool(run.get("success", False)))
        failed_count = sum(1 for run in subset if bool(run.get("failed", False)))
        stats = {
            "count": len(subset),
            "success_count": success_count,
            "failed_count": failed_count,
            "success_rate": float(success_count / len(subset)) if subset else 0.0,
        }
        for metric in NUMERIC_METRICS:
            values = [_finite_float(run.get(metric)) for run in subset]
            values = [value for value in values if value is not None]
            stats[f"{metric}_count"] = len(values)
            stats[f"{metric}_mean"] = float(np.mean(values)) if values else None
            stats[f"{metric}_std"] = float(np.std(values)) if values else None
        aggregates[controller] = stats
    return aggregates


def compute_paired_deltas(runs: Sequence[dict]) -> dict:
    nominal = _index_runs(runs, "nominal")
    learned = _index_runs(runs, "learned")
    pairs = []
    for key in sorted(set(nominal) & set(learned)):
        nominal_run = nominal[key]
        learned_run = learned[key]
        scenario, episode_id, seed = key
        pair = {
            "scenario": scenario,
            "episode_id": episode_id,
            "seed": seed,
            "nominal_results_path": nominal_run.get("results_path"),
            "learned_results_path": learned_run.get("results_path"),
            "success_delta": float(bool(learned_run.get("success", False)))
            - float(bool(nominal_run.get("success", False))),
        }
        for metric in NUMERIC_METRICS:
            nominal_value = _finite_float(nominal_run.get(metric))
            learned_value = _finite_float(learned_run.get(metric))
            pair[f"{metric}_delta"] = (
                float(learned_value - nominal_value)
                if nominal_value is not None and learned_value is not None
                else None
            )
        pairs.append(pair)

    aggregate = {"count": len(pairs)}
    for metric in ("success", *NUMERIC_METRICS):
        key = f"{metric}_delta"
        values = [_finite_float(pair.get(key)) for pair in pairs]
        values = [value for value in values if value is not None]
        aggregate[f"{key}_mean"] = float(np.mean(values)) if values else None
        aggregate[f"{key}_std"] = float(np.std(values)) if values else None
    return {"pairs": pairs, "aggregate": aggregate}


def current_git_metadata() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "git_sha": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": _git_dirty(repo_root),
    }


def shell_join(argv: Sequence[str]) -> str:
    return shlex.join([str(item) for item in argv])


def _run_record(
    *,
    summary_path: Path,
    scenario: str,
    controller: str,
    episode_id: int,
    seed: int,
    results_path: Path,
) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = {
        "scenario": str(scenario),
        "controller": str(controller),
        "episode_id": int(episode_id),
        "seed": int(seed),
        "results_path": str(results_path),
    }
    for metric in RUN_METRICS:
        record[metric] = _json_scalar(summary.get(metric))
    return record


def _apply_episode_seed(config: dict, seed: int) -> None:
    if bool(config.get("scenario", {}).get("random_start_goal_enabled", False)):
        config.setdefault("scenario", {})["random_seed"] = int(seed)
    if "oracle_residual" in config:
        config.setdefault("oracle_residual", {})["seed"] = int(seed)


def _index_runs(runs: Sequence[dict], controller: str) -> dict:
    indexed = {}
    for run in runs:
        if run.get("controller") != controller:
            continue
        key = (str(run["scenario"]), int(run["episode_id"]), int(run["seed"]))
        indexed[key] = run
    return indexed


def _finite_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_controllers(controllers: Iterable[str]) -> list[str]:
    parsed = [str(controller).strip().lower() for controller in controllers if str(controller).strip()]
    if not parsed:
        raise ValueError("At least one controller must be specified")
    invalid = [controller for controller in parsed if controller not in VALID_CONTROLLERS]
    if invalid:
        raise ValueError(f"Unsupported controllers: {invalid}. Expected a subset of {VALID_CONTROLLERS}")
    return parsed


def _safe_name(name: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in str(name)]
    safe = "".join(chars).strip("_")
    return safe or "scenario"


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    value = result.stdout.strip()
    return value or None


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(result.stdout.strip())


def parse_controllers(value: str) -> tuple[str, ...]:
    return tuple(_validate_controllers(value.split(",")))


def parse_learned_mppi_overrides(entries: Sequence[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for entry in entries or []:
        if "=" not in str(entry):
            raise ValueError(f"Expected learned MPPI override in KEY=VALUE form, got: {entry}")
        key, value = str(entry).split("=", 1)
        key = key.strip()
        if key not in LEARNED_MPPI_OVERRIDE_KEYS:
            raise ValueError(
                f"Unsupported learned MPPI override '{key}'. Expected one of {LEARNED_MPPI_OVERRIDE_KEYS}"
            )
        try:
            overrides[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Override '{entry}' must use a numeric value") from exc
    return overrides


def parse_mppi_overrides(entries: Sequence[str] | None) -> dict[str, float | str]:
    overrides: dict[str, float | str] = {}
    for entry in entries or []:
        if "=" not in str(entry):
            raise ValueError(f"Expected shared MPPI override in KEY=VALUE form, got: {entry}")
        key, value = str(entry).split("=", 1)
        key = key.strip()
        if key not in MPPI_OVERRIDE_KEYS:
            raise ValueError(f"Unsupported shared MPPI override '{key}'. Expected one of {MPPI_OVERRIDE_KEYS}")
        if key in MPPI_STRING_OVERRIDE_KEYS:
            overrides[key] = value.strip()
        else:
            try:
                overrides[key] = float(value)
            except ValueError as exc:
                raise ValueError(f"Override '{entry}' must use a numeric value") from exc
    return overrides


def _validate_learned_mppi_overrides(overrides: dict[str, float]) -> dict[str, float]:
    parsed = {}
    for key, value in overrides.items():
        if key not in LEARNED_MPPI_OVERRIDE_KEYS:
            raise ValueError(
                f"Unsupported learned MPPI override '{key}'. Expected one of {LEARNED_MPPI_OVERRIDE_KEYS}"
            )
        parsed[key] = float(value)
    return parsed


def _validate_mppi_overrides(overrides: dict[str, float | str]) -> dict[str, float | str]:
    parsed: dict[str, float | str] = {}
    for key, value in overrides.items():
        if key not in MPPI_OVERRIDE_KEYS:
            raise ValueError(f"Unsupported shared MPPI override '{key}'. Expected one of {MPPI_OVERRIDE_KEYS}")
        if key in MPPI_STRING_OVERRIDE_KEYS:
            parsed[key] = str(value).strip().lower()
        else:
            parsed[key] = float(value)
    return parsed


def _apply_mppi_overrides(config: dict, overrides: dict[str, float | str]) -> None:
    validated = _validate_mppi_overrides(overrides)
    if not validated:
        return
    mppi = config.setdefault("mppi", {})
    for key, value in validated.items():
        mppi[key] = value


def _apply_learned_mppi_overrides(config: dict, overrides: dict[str, float]) -> None:
    validated = _validate_learned_mppi_overrides(overrides)
    if not validated:
        return
    mppi = config.setdefault("mppi", {})
    for key, value in validated.items():
        if key == "goal_xy_weight":
            weights = list(mppi.get("weights", [0.0, 0.0, 0.0]))
            while len(weights) < 3:
                weights.append(0.0)
            weights[0] = float(value)
            mppi["weights"] = weights
        else:
            mppi[key] = float(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/b2_omni_oracle.yaml")
    parser.add_argument("--scenario-name", default="standard")
    parser.add_argument("--output", default="results/stage5_benchmark/standard_seed123")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--backend", choices=["numpy", "cuda", "torch"], default="torch")
    parser.add_argument("--controllers", default="nominal,learned")
    parser.add_argument("--fdm-model-dir", default="results/fdm_baselines/stage4_mlp_seed123_hardened")
    parser.add_argument("--fdm-checkpoint", default="best_model.pt")
    parser.add_argument("--fdm-normalization", default="normalization.npz")
    parser.add_argument("--fdm-device", default=None)
    parser.add_argument(
        "--mppi-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply a shared MPPI override to nominal and learned controllers; repeat for multiple keys.",
    )
    parser.add_argument(
        "--learned-mppi-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply a learned-controller-only MPPI cost override; repeat for multiple keys.",
    )
    args = parser.parse_args(argv)

    summary = run_benchmark(
        config_path=args.config,
        scenario_name=args.scenario_name,
        output_dir=args.output,
        episodes=args.episodes,
        base_seed=args.base_seed,
        backend=args.backend,
        controllers=parse_controllers(args.controllers),
        fdm_model_dir=args.fdm_model_dir,
        fdm_checkpoint=args.fdm_checkpoint,
        fdm_normalization=args.fdm_normalization,
        fdm_device=args.fdm_device or ("cuda" if args.backend == "cuda" else "cpu"),
        mppi_overrides=parse_mppi_overrides(args.mppi_override),
        learned_mppi_overrides=parse_learned_mppi_overrides(args.learned_mppi_override),
        command=shell_join([sys.executable, *sys.argv]),
        argv=[sys.executable, *sys.argv],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
