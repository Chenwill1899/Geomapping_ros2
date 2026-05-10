"""Configuration loading and validation for the MPPI simulation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_GROUPS = ("simulation", "mppi", "robot", "cbf", "obstacles", "results")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for group in REQUIRED_GROUPS:
        if group not in config:
            raise ValueError(f"Missing required config group: {group}")

    state_dim = int(config["mppi"].get("state_dim", 0))
    control_dim = int(config["mppi"].get("control_dim", 0))
    goal = config["simulation"].get("goal")
    if goal is not None and _mujoco_closed_loop_profile(config):
        raise ValueError("MuJoCo closed-loop profiles must use RViz /move_base_simple/goal, not simulation.goal")
    if goal is None and (_external_path_only(config) or _runtime_goal_required(config)):
        pass
    else:
        _require_len(goal, state_dim, "simulation.goal")
    _require_len(config["simulation"].get("initial_state"), state_dim, "simulation.initial_state")
    _require_len(config["mppi"].get("weights"), state_dim, "mppi.weights")
    _require_len(config["mppi"].get("std_normal"), control_dim, "mppi.std_normal")
    if int(config["robot"].get("state_dim", state_dim)) != state_dim:
        raise ValueError("robot.state_dim must match mppi.state_dim")

    for index, obstacle in enumerate(config["obstacles"].get("virtual", [])):
        _require_len(obstacle, 7, f"obstacles.virtual[{index}]")


def merged_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    _deep_update(merged, overrides)
    validate_config(merged)
    return merged


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _require_len(value: Any, expected: int, name: str) -> None:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"{name} must be a list with {expected} values")


def _external_path_only(config: dict[str, Any]) -> bool:
    return bool(config.get("external_path", {}).get("enabled", False)) and bool(
        config.get("simulation", {}).get("disable_goal_termination", False)
    )


def _runtime_goal_required(config: dict[str, Any]) -> bool:
    goal_topic = config.get("goal_topic", {})
    return bool(goal_topic.get("enabled", False)) and bool(goal_topic.get("required", False))


def _mujoco_closed_loop_profile(config: dict[str, Any]) -> bool:
    return "mujoco" in config
