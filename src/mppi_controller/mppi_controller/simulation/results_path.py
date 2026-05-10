"""Helpers for creating simulation result directories."""

from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path
from typing import Any


def create_results_path(results_config: dict[str, Any] | str) -> Path:
    if isinstance(results_config, str):
        root = Path(results_config)
        run_name = None
        overwrite = False
        timestamp_suffix = False
    else:
        root = Path(str(results_config["root"]))
        run_name = results_config.get("run_name")
        overwrite = bool(results_config.get("overwrite", False))
        timestamp_suffix = bool(results_config.get("timestamp_suffix", False))

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if timestamp_suffix and overwrite:
        raise ValueError("results.timestamp_suffix cannot be combined with overwrite")
    if run_name and timestamp_suffix:
        path = root / f"{run_name}_{timestamp}"
    elif run_name:
        path = root / str(run_name)
    else:
        path = root / timestamp
    if overwrite and path.exists():
        _remove_run_directory(path, root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _remove_run_directory(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to overwrite unsafe results path: {path}")
    shutil.rmtree(resolved_path)
