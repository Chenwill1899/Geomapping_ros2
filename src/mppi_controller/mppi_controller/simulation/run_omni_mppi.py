#!/usr/bin/env python3
"""Run the B2 omni MPPI simulation from a YAML config."""

from __future__ import annotations

import argparse
import sys

from mppi_controller.config import load_config
from mppi_controller.simulation.omni_runner import OmniMppiSimulationRunner, create_omni_controller


def run_omni_mppi(
    *,
    config_path: str,
    seed: int = 123,
    backend: str | None = None,
    fdm_enabled: bool = False,
    fdm_model_dir: str | None = None,
    fdm_checkpoint: str | None = None,
    fdm_normalization: str | None = None,
    fdm_device: str | None = None,
):
    args = argparse.Namespace(
        backend=backend,
        fdm_enabled=fdm_enabled,
        fdm_model_dir=fdm_model_dir,
        fdm_checkpoint=fdm_checkpoint,
        fdm_normalization=fdm_normalization,
        fdm_device=fdm_device,
    )
    config = load_config(config_path)
    config = apply_cli_overrides(config, args)
    runner = OmniMppiSimulationRunner(
        config,
        controller_factory=lambda *, config, runner: create_omni_controller(config, seed=seed),
    )
    return runner.run()


def print_run_summary(summary) -> None:
    print(f"results_path={summary.results_path}")
    print(f"steps={summary.steps}")
    print(f"reached_goal={summary.reached_goal}")
    print(f"failed={summary.failed}")
    print(f"run_time={summary.run_time}")
    animation_path = summary.results_path / "animation.gif"
    if animation_path.exists():
        print(f"animation={animation_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--backend", choices=["cuda", "numpy", "torch"], default=None)
    parser.add_argument("--fdm-enabled", action="store_true")
    parser.add_argument("--fdm-model-dir", default=None)
    parser.add_argument("--fdm-checkpoint", default=None)
    parser.add_argument("--fdm-normalization", default=None)
    parser.add_argument("--fdm-device", default=None)
    args = parser.parse_args(argv)

    summary = run_omni_mppi(
        config_path=args.config,
        seed=args.seed,
        backend=args.backend,
        fdm_enabled=args.fdm_enabled,
        fdm_model_dir=args.fdm_model_dir,
        fdm_checkpoint=args.fdm_checkpoint,
        fdm_normalization=args.fdm_normalization,
        fdm_device=args.fdm_device,
    )
    print_run_summary(summary)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.backend is not None:
        config.setdefault("mppi", {})["backend"] = str(args.backend).lower()
    if args.fdm_enabled:
        config.setdefault("fdm", {})["enabled"] = True
        config.setdefault("fdm", {})["mode"] = "sequence"
        if args.backend is None and str(config.setdefault("mppi", {}).get("backend", "numpy")).lower() == "numpy":
            config["mppi"]["backend"] = "torch"
    if args.fdm_model_dir is not None:
        config.setdefault("fdm", {})["model_dir"] = args.fdm_model_dir
    if args.fdm_checkpoint is not None:
        config.setdefault("fdm", {})["checkpoint"] = args.fdm_checkpoint
    if args.fdm_normalization is not None:
        config.setdefault("fdm", {})["normalization"] = args.fdm_normalization
    if args.fdm_device is not None:
        config.setdefault("fdm", {})["device"] = args.fdm_device
    return config


if __name__ == "__main__":
    main()
