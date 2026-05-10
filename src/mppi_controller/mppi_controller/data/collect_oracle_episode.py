#!/usr/bin/env python3
"""Collect one oracle simulation episode and save it as an FDM npz sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mppi_controller.config import load_config
from mppi_controller.data.oracle_episode import build_episode_npz
from mppi_controller.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller


def collect_oracle_episode(
    *,
    config_path: str | Path,
    episode_id: int,
    seed: int,
    output_path: str | Path,
    backend: str | None = None,
) -> dict:
    output_path = Path(output_path)
    config = load_config(config_path)
    config.setdefault("scenario", {})["random_seed"] = int(seed)
    config.setdefault("oracle_residual", {})["seed"] = int(seed)
    if backend is not None:
        config["mppi"]["backend"] = str(backend).lower()
    _set_episode_results_path(config, output_path=output_path, episode_id=episode_id)

    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *, config, runner: create_omni_controller(config, seed=seed),
    )
    summary = runner.run()
    return build_episode_npz(summary.results_path, episode_id, output_path)


def _set_episode_results_path(config: dict, *, output_path: Path, episode_id: int) -> None:
    output_dir = output_path.parent.parent
    config["results"] = {
        **config.get("results", {}),
        "root": str(output_dir / "raw_results"),
        "run_name": f"episode_{int(episode_id):06d}",
        "timestamp_suffix": False,
        "overwrite": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["cuda", "numpy"], default=None)
    args = parser.parse_args(argv)

    metadata = collect_oracle_episode(
        config_path=args.config,
        episode_id=args.episode_id,
        seed=args.seed,
        output_path=args.output,
        backend=args.backend,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
