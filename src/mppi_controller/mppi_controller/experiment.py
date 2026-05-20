"""Profile-driven experiment runner for the FDM-MPPI workbench."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from mppi_controller.config import load_config, validate_config
from mppi_controller.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller


SIMULATION_SCENARIO_KEYS = {
    "sampling_rate",
    "time_horizon",
    "map_size",
    "map_origin",
    "goal",
    "initial_state",
    "minimum_distance",
    "goal_termination_distance",
    "max_steps",
    "without_heading",
    "print_out",
    "world_mode",
    "disable_goal_termination",
}
SCENARIO_GROUP_KEYS = {
    "obstacles",
    "terrain",
    "oracle_residual",
    "execution",
    "visualization",
}
TOP_LEVEL_CONFIG_GROUPS = {
    "simulation",
    "mppi",
    "robot",
    "cbf",
    "obstacles",
    "results",
    "execution",
    "terrain",
    "oracle_residual",
    "visualization",
    "mujoco",
    "elevation_map",
    "local_costmap",
    "dynamic_obstacles",
    "goal_topic",
    "external_path",
    "global_path",
    "local_goal",
    "final_controller",
    "command_filter",
    "reactive_avoidance",
    "rviz",
    "scenario",
}
LEARNED_METHODS = {"learned", "learned_fdm", "fdm", "residual_fdm"}
DEFAULT_MODEL_SEARCH_ROOT = Path("results/fdm_baselines")
ARTIFACT_FILES = {
    "summary_json": "summary.json",
    "config_yaml": "config.yaml",
    "trajectory_csv": "trajectory.csv",
    "controls_csv": "controls.csv",
    "raw_controls_csv": "raw_controls.csv",
    "residuals_csv": "residuals.csv",
    "terrain_csv": "terrain.csv",
    "time_results_csv": "time_results.csv",
    "trajectory_png": "trajectory.png",
    "oracle_diagnostics_png": "oracle_diagnostics.png",
    "animation_gif": "animation.gif",
}


class ExperimentConfigError(RuntimeError):
    """Raised when an experiment profile is valid YAML but cannot run."""


def load_experiment_profile(profile_path: str | Path) -> dict[str, Any]:
    path = Path(profile_path)
    with path.open("r", encoding="utf-8") as stream:
        profile = yaml.safe_load(stream) or {}
    if not isinstance(profile, dict):
        raise ValueError("experiment profile must be a YAML mapping")
    return profile


def build_experiment_config(
    profile_path: str | Path,
    *,
    controller_name: str | None = None,
    seed: int | None = None,
    backend: str | None = None,
    output_root: str | Path | None = None,
    results_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    normalization: str | Path | None = None,
    device: str | None = None,
    residual_gain: float | None = None,
    enable_plots: bool | None = None,
    enable_animation: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_path = Path(profile_path)
    profile = load_experiment_profile(profile_path)
    experiment = _mapping(profile.get("experiment", {}), "experiment")
    experiment_name = str(experiment.get("name", profile_path.stem))
    base_config_path = _resolve_path(profile_path, experiment.get("base_config", "configs/smoke.yaml"))
    config = load_config(base_config_path)

    _apply_profile_config_groups(config, profile)
    _sync_goal_relief_center(config, profile)
    controller = _select_controller(profile, controller_name)
    resolved_seed = int(seed if seed is not None else experiment.get("seed", 123))
    resolved_controller_name = str(controller.get("name", controller_name or "default"))
    method = str(controller.get("method", "nominal")).lower()
    _apply_controller(
        config,
        controller,
        method=method,
        backend=backend,
        model_dir=model_dir,
        checkpoint=checkpoint,
        normalization=normalization,
        device=device,
        residual_gain=residual_gain,
    )
    _apply_output_and_visualization(
        config,
        profile,
        experiment=experiment,
        experiment_name=experiment_name,
        controller_name=resolved_controller_name,
        seed=resolved_seed,
        output_root=output_root,
        results_dir=results_dir,
        enable_plots=enable_plots,
        enable_animation=enable_animation,
    )
    validate_config(config)
    metadata = {
        "profile_path": str(profile_path),
        "base_config_path": str(base_config_path),
        "experiment_name": experiment_name,
        "controller_name": resolved_controller_name,
        "method": method,
        "seed": resolved_seed,
    }
    return config, metadata


def run_experiment_profile(
    profile_path: str | Path,
    *,
    controller_name: str | None = None,
    seed: int | None = None,
    backend: str | None = None,
    output_root: str | Path | None = None,
    results_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    normalization: str | Path | None = None,
    device: str | None = None,
    residual_gain: float | None = None,
    enable_plots: bool | None = None,
    enable_animation: bool | None = None,
    runner_cls=OmniMppiSimulationRunner,
) -> dict[str, Any]:
    config, metadata = build_experiment_config(
        profile_path,
        controller_name=controller_name,
        seed=seed,
        backend=backend,
        output_root=output_root,
        results_dir=results_dir,
        model_dir=model_dir,
        checkpoint=checkpoint,
        normalization=normalization,
        device=device,
        residual_gain=residual_gain,
        enable_plots=enable_plots,
        enable_animation=enable_animation,
    )
    validate_learned_fdm_artifacts(config, metadata)
    runner = runner_cls(
        config,
        controller_factory=lambda *, config, runner: create_omni_controller(config, seed=metadata["seed"]),
    )
    summary = runner.run()
    artifacts = collect_run_artifacts(summary.results_path)
    result = {
        **metadata,
        "backend": str(config["mppi"].get("backend", "numpy")).lower(),
        "results_path": str(summary.results_path),
        "steps": int(summary.steps),
        "reached_goal": bool(summary.reached_goal),
        "failed": bool(summary.failed),
        "run_time": float(summary.run_time),
        "artifacts": artifacts,
    }
    summary_path = Path(summary.results_path) / "summary.json"
    if summary_path.exists():
        result["metrics"] = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = Path(summary.results_path) / "experiment_summary.json"
    result["artifacts"]["experiment_summary"] = str(manifest_path)
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def collect_run_artifacts(results_path: str | Path) -> dict[str, str | None]:
    results_path = Path(results_path)
    artifacts: dict[str, str | None] = {}
    for key, filename in ARTIFACT_FILES.items():
        path = results_path / filename
        artifacts[key] = str(path) if path.exists() else None
    return artifacts


def validate_learned_fdm_artifacts(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    fdm = _mapping(config.get("fdm", {}), "fdm")
    if not bool(fdm.get("enabled", False)):
        return
    model_dir = Path(str(fdm.get("model_dir", "")))
    checkpoint = _resolve_artifact(model_dir, fdm.get("checkpoint", "best_model.pt"))
    normalization = _resolve_artifact(model_dir, fdm.get("normalization", "normalization.npz"))
    missing = [path for path in (checkpoint, normalization) if not path.exists()]
    if not missing:
        return
    raise ExperimentConfigError(_missing_fdm_artifact_message(missing, model_dir, checkpoint, normalization, metadata))


def _apply_profile_config_groups(config: dict[str, Any], profile: Mapping[str, Any]) -> None:
    for group in TOP_LEVEL_CONFIG_GROUPS:
        if group in profile:
            _deep_update(config.setdefault(group, {}), _mapping(profile[group], group))

    scenario = _mapping(profile.get("scenario", {}), "scenario")
    scenario_config = config.setdefault("scenario", {})
    for key, value in scenario.items():
        if key in SIMULATION_SCENARIO_KEYS:
            config.setdefault("simulation", {})[key] = deepcopy(value)
        elif key in SCENARIO_GROUP_KEYS:
            _deep_update(config.setdefault(key, {}), _mapping(value, f"scenario.{key}"))
        else:
            scenario_config[key] = deepcopy(value)


def _sync_goal_relief_center(config: dict[str, Any], profile: Mapping[str, Any]) -> None:
    terrain = config.get("terrain")
    if not isinstance(terrain, dict):
        return
    relief = terrain.get("goal_relief")
    if not isinstance(relief, dict) or not bool(relief.get("enabled", False)):
        return
    explicit_center = _profile_goal_relief_center(profile)
    if explicit_center is not None and not _is_auto_center(explicit_center):
        return
    if explicit_center is None and not _profile_overrides_goal(profile) and not _is_auto_center(relief.get("center")):
        return
    goal = config.get("simulation", {}).get("goal")
    if not isinstance(goal, list) or len(goal) < 2:
        return
    relief["center"] = [float(goal[0]), float(goal[1])]


def _profile_goal_relief_center(profile: Mapping[str, Any]) -> Any:
    for terrain in (
        _optional_mapping(_optional_mapping(profile.get("scenario")).get("terrain")),
        _optional_mapping(profile.get("terrain")),
    ):
        relief = _optional_mapping(terrain.get("goal_relief"))
        if "center" in relief:
            return relief["center"]
    return None


def _profile_overrides_goal(profile: Mapping[str, Any]) -> bool:
    return "goal" in _optional_mapping(profile.get("scenario")) or "goal" in _optional_mapping(profile.get("simulation"))


def _is_auto_center(value: Any) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def _optional_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _select_controller(profile: Mapping[str, Any], controller_name: str | None) -> dict[str, Any]:
    if "controllers" in profile:
        controllers = profile["controllers"]
        if not isinstance(controllers, list) or not controllers:
            raise ValueError("controllers must be a non-empty list")
        selected_name = controller_name or profile.get("default_controller") or controllers[0].get("name")
        for controller in controllers:
            controller_map = _mapping(controller, "controllers[]")
            if str(controller_map.get("name")) == str(selected_name):
                return deepcopy(controller_map)
        raise ValueError(f"controller '{selected_name}' is not defined in the experiment profile")
    if "controller" in profile:
        controller = deepcopy(_mapping(profile["controller"], "controller"))
        if controller_name is not None and str(controller.get("name", controller_name)) != str(controller_name):
            raise ValueError(f"controller '{controller_name}' is not defined in the experiment profile")
        controller.setdefault("name", controller_name or "default")
        return controller
    if controller_name is not None:
        return {"name": controller_name, "method": "nominal"}
    return {"name": "nominal", "method": "nominal"}


def _apply_controller(
    config: dict[str, Any],
    controller: Mapping[str, Any],
    *,
    method: str,
    backend: str | None,
    model_dir: str | Path | None,
    checkpoint: str | Path | None,
    normalization: str | Path | None,
    device: str | None,
    residual_gain: float | None,
) -> None:
    mppi = config.setdefault("mppi", {})
    if "mppi" in controller:
        _deep_update(mppi, _mapping(controller["mppi"], "controller.mppi"))
    selected_backend = backend if backend is not None else controller.get("backend")
    if selected_backend is not None:
        mppi["backend"] = str(selected_backend).lower()

    if method in LEARNED_METHODS:
        fdm = config.setdefault("fdm", {})
        fdm["enabled"] = True
        learned = controller.get("learned_fdm", controller.get("fdm", {}))
        if learned:
            _deep_update(fdm, _mapping(learned, "controller.learned_fdm"))
        if model_dir is not None:
            fdm["model_dir"] = str(model_dir)
        if checkpoint is not None:
            fdm["checkpoint"] = str(checkpoint)
        if normalization is not None:
            fdm["normalization"] = str(normalization)
        if device is not None:
            fdm["device"] = str(device)
        if residual_gain is not None:
            fdm["residual_gain"] = float(residual_gain)
    else:
        config.setdefault("fdm", {})["enabled"] = False


def _apply_output_and_visualization(
    config: dict[str, Any],
    profile: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any],
    experiment_name: str,
    controller_name: str,
    seed: int,
    output_root: str | Path | None,
    results_dir: str | Path | None,
    enable_plots: bool | None,
    enable_animation: bool | None,
) -> None:
    results = config.setdefault("results", {})
    results["overwrite"] = bool(experiment.get("overwrite", True))
    resolved_results_dir = results_dir if results_dir is not None else experiment.get("results_dir")
    if resolved_results_dir is not None:
        _apply_explicit_results_dir(results, resolved_results_dir)
    else:
        resolved_output_root = output_root if output_root is not None else experiment.get("output_root", results.get("root"))
        if resolved_output_root is not None:
            results["root"] = str(resolved_output_root)
        run_name_template = str(experiment.get("run_name", "{experiment}_{controller}_seed{seed}"))
        results["run_name"] = run_name_template.format(
            experiment=experiment_name,
            controller=controller_name,
            seed=seed,
        )
        results["timestamp_suffix"] = bool(experiment.get("timestamp_suffix", False))

    visualization = _mapping(profile.get("visualization", {}), "visualization")
    if "plots" in visualization:
        results["enable_plots"] = bool(visualization["plots"])
    if "animation" in visualization:
        results["enable_animation"] = bool(visualization["animation"])
    if enable_plots is not None:
        results["enable_plots"] = bool(enable_plots)
    if enable_animation is not None:
        results["enable_animation"] = bool(enable_animation)
    visualization_config = config.setdefault("visualization", {})
    for key, value in visualization.items():
        if key not in {"plots", "animation"}:
            visualization_config[key] = deepcopy(value)


def _apply_explicit_results_dir(results: dict[str, Any], results_dir: str | Path) -> None:
    path = Path(str(results_dir))
    if not path.name or path.name in {".", ".."}:
        raise ValueError(f"results_dir must point to a named run directory: {results_dir}")
    results["root"] = str(path.parent)
    results["run_name"] = path.name
    results["timestamp_suffix"] = False


def _resolve_path(profile_path: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return profile_path.parent / path


def _resolve_artifact(model_dir: Path, artifact_path: Any) -> Path:
    path = Path(str(artifact_path))
    if path.is_absolute():
        return path
    return model_dir / path


def _missing_fdm_artifact_message(
    missing: list[Path],
    model_dir: Path,
    checkpoint: Path,
    normalization: Path,
    metadata: Mapping[str, Any],
) -> str:
    lines = [
        "Missing learned FDM artifact(s) for experiment controller "
        f"'{metadata.get('controller_name', 'unknown')}'.",
        "Requested files:",
        f"  checkpoint: {checkpoint}",
        f"  normalization: {normalization}",
        "Missing:",
        *[f"  - {path}" for path in missing],
        "Fix:",
        "  - pass an existing model directory with --model-dir, or",
        "  - choose the matching checkpoint/normalization names, or",
        "  - train a residual FDM first:",
        "    /usr/bin/python3 tools/fdm_mppi.py train --dataset <dataset_splits> --output results/fdm_baselines/<name>",
    ]
    candidates = _discover_model_candidates(model_dir.parent if model_dir.parent != Path("") else DEFAULT_MODEL_SEARCH_ROOT)
    if candidates:
        lines.extend(["Available local model directories:", *[f"  - {candidate}" for candidate in candidates]])
    return "\n".join(lines)


def _discover_model_candidates(search_root: Path) -> list[Path]:
    roots = []
    if search_root.exists():
        roots.append(search_root)
    if DEFAULT_MODEL_SEARCH_ROOT.exists() and DEFAULT_MODEL_SEARCH_ROOT not in roots:
        roots.append(DEFAULT_MODEL_SEARCH_ROOT)
    candidates: list[Path] = []
    for root in roots:
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            has_normalization = (directory / "normalization.npz").exists()
            has_checkpoint = (directory / "best_model.pt").exists() or (directory / "model.pt").exists()
            if has_normalization and has_checkpoint and directory not in candidates:
                candidates.append(directory)
    return candidates[:8]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
