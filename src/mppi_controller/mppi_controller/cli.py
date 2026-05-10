"""Unified command line interface for the slim FDM-MPPI pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence


class PipelineCommands:
    """Command handlers kept thin so tests can inject a recorder."""

    def experiment(self, args: argparse.Namespace) -> int:
        from mppi_controller.experiment import ExperimentConfigError, run_experiment_profile

        try:
            summary = run_experiment_profile(
                args.profile,
                controller_name=args.controller,
                seed=args.seed,
                backend=args.backend,
                output_root=args.output,
                results_dir=args.results_dir,
                model_dir=args.model_dir,
                checkpoint=args.checkpoint,
                normalization=args.normalization,
                device=args.device,
                residual_gain=args.residual_gain,
                enable_plots=args.plots,
                enable_animation=args.animation,
            )
        except ExperimentConfigError as exc:
            print(f"experiment error: {exc}", file=sys.stderr)
            return 2
        _print_json(summary)
        return 0

    def mujoco_closed_loop(self, args: argparse.Namespace) -> int:
        from mppi_controller.experiment import ExperimentConfigError
        from mppi_controller.mujoco_closed_loop import run_mujoco_closed_loop_profile

        try:
            summary = run_mujoco_closed_loop_profile(
                args.profile,
                controller_name=args.controller,
                seed=args.seed,
                backend=args.backend,
                results_dir=args.results_dir,
                max_steps=args.max_steps,
                odom_timeout=args.odom_timeout,
            )
        except ExperimentConfigError as exc:
            print(f"mujoco closed-loop error: {exc}", file=sys.stderr)
            return 2
        _print_json(summary)
        return 0

    def run(self, args: argparse.Namespace) -> int:
        from mppi_controller.simulation.run_omni_mppi import print_run_summary, run_omni_mppi

        summary = run_omni_mppi(
            config_path=args.config,
            seed=args.seed,
            backend=args.backend,
            fdm_enabled=args.fdm_enabled,
            fdm_model_dir=args.fdm_model_dir,
            fdm_checkpoint=args.fdm_checkpoint,
            fdm_normalization=args.fdm_normalization,
            fdm_device=args.fdm_device,
            fdm_residual_gain=args.fdm_residual_gain,
        )
        print_run_summary(summary)
        return 0

    def dataset_collect(self, args: argparse.Namespace) -> int:
        from mppi_controller.data.generate_oracle_episodes import generate_oracle_episodes

        summary = generate_oracle_episodes(
            config_path=args.config,
            episodes=args.episodes,
            base_seed=args.base_seed,
            output_dir=args.output,
            backend=args.backend,
            num_workers=args.num_workers,
        )
        _print_json(summary)
        return 0

    def dataset_build(self, args: argparse.Namespace) -> int:
        from mppi_controller.data.oracle_dataset import build_oracle_dataset

        summary = build_oracle_dataset(
            input_dir=args.input,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.seed,
        )
        _print_json(summary)
        return 0

    def dataset_validate(self, args: argparse.Namespace) -> int:
        from mppi_controller.data.oracle_dataset_validator import validate_oracle_dataset

        output = args.output or args.dataset
        quality = validate_oracle_dataset(Path(args.dataset), Path(output))
        _print_json(quality)
        return 0

    def train(self, args: argparse.Namespace) -> int:
        from mppi_controller.training.residual_fdm import (
            shell_join,
            train_residual_fdm,
            train_sequence_fdm,
        )

        argv = [sys.executable, "tools/fdm_mppi.py", *sys.argv[1:]]
        if args.sequence:
            metrics = train_sequence_fdm(
                dataset_dir=args.dataset,
                output_dir=args.output,
                sequence_horizon=args.sequence_horizon,
                include_history_controls=args.sequence_include_history,
                history_steps=args.sequence_history_steps,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
                device=args.device,
                tensorboard_log_dir=args.tensorboard_log_dir,
                command=shell_join(argv),
                argv=argv,
            )
        else:
            metrics = train_residual_fdm(
                dataset_dir=args.dataset,
                output_dir=args.output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
                device=args.device,
                tensorboard_log_dir=args.tensorboard_log_dir,
                command=shell_join(argv),
                argv=argv,
            )
        _print_json(metrics)
        return 0

    def eval_dataset(self, args: argparse.Namespace) -> int:
        from mppi_controller.evaluation.residual_fdm_dataset import evaluate_residual_fdm_dataset, shell_join

        metrics = evaluate_residual_fdm_dataset(
            dataset_dir=args.dataset,
            model_dir=args.model_dir,
            output_dir=args.output,
            checkpoint=args.checkpoint,
            normalization=args.normalization,
            device=args.device,
            command=shell_join([sys.executable, "tools/fdm_mppi.py", *sys.argv[1:]]),
        )
        _print_json(metrics)
        return 0

    def eval_rollout(self, args: argparse.Namespace) -> int:
        from mppi_controller.evaluation.residual_fdm_rollout import evaluate_residual_fdm_rollout, shell_join

        metrics = evaluate_residual_fdm_rollout(
            config_path=args.config,
            model_dir=args.model_dir,
            output_dir=args.output,
            seed=args.seed,
            backend=args.backend,
            device=args.device,
            checkpoint=args.checkpoint,
            normalization=args.normalization,
            generate_gif=not args.no_gif,
            gif_fps=args.gif_fps,
            gif_max_frames=args.gif_max_frames,
            command=shell_join([sys.executable, "tools/fdm_mppi.py", *sys.argv[1:]]),
        )
        _print_json(metrics)
        return 0

    def benchmark(self, args: argparse.Namespace) -> int:
        from mppi_controller.evaluation.benchmark import (
            parse_controllers,
            parse_learned_mppi_overrides,
            parse_mppi_overrides,
            run_benchmark,
            shell_join,
        )

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
            fdm_device=args.fdm_device,
            fdm_residual_gain=args.fdm_residual_gain,
            mppi_overrides=parse_mppi_overrides(args.mppi_override),
            learned_mppi_overrides=parse_learned_mppi_overrides(args.learned_mppi_override),
            command=shell_join([sys.executable, "tools/fdm_mppi.py", *sys.argv[1:]]),
            argv=[sys.executable, "tools/fdm_mppi.py", *sys.argv[1:]],
        )
        _print_json(summary)
        return 0

    def report(self, args: argparse.Namespace) -> int:
        from mppi_controller.reporting.pipeline_report import write_pipeline_report

        report = write_pipeline_report(
            output_path=args.output,
            run_path=args.run_path,
            dataset_path=args.dataset,
            training_path=args.training,
            benchmark_path=args.benchmark,
        )
        _print_json(report)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FDM-MPPI slim reproducible pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment = subparsers.add_parser("experiment", help="Run a profile-defined MPPI experiment")
    experiment.add_argument("--profile", default="configs/experiment.yaml")
    experiment.add_argument("--controller", default=None)
    experiment.add_argument("--seed", type=int, default=None)
    experiment.add_argument("--backend", choices=["cuda", "numpy", "torch"], default=None)
    experiment.add_argument("--output", default=None)
    experiment.add_argument("--results-dir", default=None)
    experiment.add_argument("--model-dir", default=None)
    experiment.add_argument("--checkpoint", default=None)
    experiment.add_argument("--normalization", default=None)
    experiment.add_argument("--device", default=None)
    experiment.add_argument("--residual-gain", type=float, default=None)
    experiment.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    experiment.add_argument("--animation", action=argparse.BooleanOptionalAction, default=None)
    experiment.set_defaults(handler="experiment")

    mujoco = subparsers.add_parser("mujoco-closed-loop", help="Run MPPI/FDM closed loop against ausim2 MuJoCo Scout")
    mujoco.add_argument("--profile", default="configs/mujoco_scout.yaml")
    mujoco.add_argument("--controller", default=None)
    mujoco.add_argument("--seed", type=int, default=None)
    mujoco.add_argument("--backend", choices=["cuda", "numpy", "torch"], default=None)
    mujoco.add_argument("--results-dir", default=None)
    mujoco.add_argument("--max-steps", type=int, default=None)
    mujoco.add_argument("--odom-timeout", type=float, default=None)
    mujoco.set_defaults(handler="mujoco_closed_loop")

    run = subparsers.add_parser("run", help="Run one MPPI simulation")
    run.add_argument("--config", default="configs/smoke.yaml")
    run.add_argument("--seed", type=int, default=123)
    run.add_argument("--backend", choices=["cuda", "numpy", "torch"], default=None)
    run.add_argument("--fdm-enabled", action="store_true")
    run.add_argument("--fdm-model-dir", default=None)
    run.add_argument("--fdm-checkpoint", default=None)
    run.add_argument("--fdm-normalization", default=None)
    run.add_argument("--fdm-device", default=None)
    run.add_argument("--fdm-residual-gain", type=float, default=None)
    run.set_defaults(handler="run")

    dataset = subparsers.add_parser("dataset", help="Collect, build, or validate oracle datasets")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    collect = dataset_sub.add_parser("collect", help="Collect oracle episodes")
    collect.add_argument("--config", required=True)
    collect.add_argument("--episodes", type=int, required=True)
    collect.add_argument("--base-seed", type=int, required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--backend", choices=["cuda", "numpy"], default=None)
    collect.add_argument("--num-workers", type=int, default=1)
    collect.set_defaults(handler="dataset_collect")

    build = dataset_sub.add_parser("build", help="Build train/val/test splits")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--train-ratio", type=float, default=0.7)
    build.add_argument("--val-ratio", type=float, default=0.15)
    build.add_argument("--test-ratio", type=float, default=0.15)
    build.add_argument("--seed", type=int, default=123)
    build.set_defaults(handler="dataset_build")

    validate = dataset_sub.add_parser("validate", help="Validate dataset splits")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--output", default=None)
    validate.set_defaults(handler="dataset_validate")

    train = subparsers.add_parser("train", help="Train residual FDM")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--device", default="cpu")
    train.add_argument("--tensorboard-log-dir", default=None)
    train.add_argument("--sequence", action="store_true", help="Train sequence FDM instead of step residual FDM")
    train.add_argument(
        "--sequence-horizon",
        type=int,
        default=25,
        help="Prediction horizon (in steps) for sequence FDM",
    )
    train.add_argument(
        "--sequence-include-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include previous controls in sequence FDM feature vector",
    )
    train.add_argument(
        "--sequence-history-steps",
        type=int,
        default=1,
        help="Number of previous control steps to include when --sequence-include-history is enabled",
    )
    train.set_defaults(handler="train")

    eval_parser = subparsers.add_parser("eval", help="Evaluate trained residual FDM")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_dataset = eval_sub.add_parser("dataset", help="Evaluate checkpoint on dataset splits")
    eval_dataset.add_argument("--dataset", required=True)
    eval_dataset.add_argument("--model-dir", required=True)
    eval_dataset.add_argument("--output", required=True)
    eval_dataset.add_argument("--checkpoint", default="best_model.pt")
    eval_dataset.add_argument("--normalization", default="normalization.npz")
    eval_dataset.add_argument("--device", default="cpu")
    eval_dataset.set_defaults(handler="eval_dataset")

    eval_rollout = eval_sub.add_parser("rollout", help="Evaluate open-loop rollout replay")
    eval_rollout.add_argument("--config", required=True)
    eval_rollout.add_argument("--model-dir", required=True)
    eval_rollout.add_argument("--output", required=True)
    eval_rollout.add_argument("--seed", type=int, default=123)
    eval_rollout.add_argument("--backend", choices=["cuda", "numpy"], default=None)
    eval_rollout.add_argument("--device", default="cpu")
    eval_rollout.add_argument("--checkpoint", default="model.pt")
    eval_rollout.add_argument("--normalization", default="normalization.npz")
    eval_rollout.add_argument("--no-gif", action="store_true")
    eval_rollout.add_argument("--gif-fps", type=int, default=8)
    eval_rollout.add_argument("--gif-max-frames", type=int, default=180)
    eval_rollout.set_defaults(handler="eval_rollout")

    benchmark = subparsers.add_parser("benchmark", help="Run nominal vs learned closed-loop benchmark")
    benchmark.add_argument("--config", default="configs/benchmark.yaml")
    benchmark.add_argument("--scenario-name", default="standard")
    benchmark.add_argument("--output", default="results/benchmark/standard_seed123")
    benchmark.add_argument("--episodes", type=int, default=1)
    benchmark.add_argument("--base-seed", type=int, default=123)
    benchmark.add_argument("--backend", choices=["numpy", "cuda", "torch"], default=None)
    benchmark.add_argument("--controllers", default="nominal,learned")
    benchmark.add_argument("--fdm-model-dir", default="results/fdm_baselines/stage4_mlp_seed123_hardened")
    benchmark.add_argument("--fdm-checkpoint", default="best_model.pt")
    benchmark.add_argument("--fdm-normalization", default="normalization.npz")
    benchmark.add_argument("--fdm-device", default=None)
    benchmark.add_argument("--fdm-residual-gain", type=float, default=1.0)
    benchmark.add_argument("--mppi-override", action="append", default=[], metavar="KEY=VALUE")
    benchmark.add_argument("--learned-mppi-override", action="append", default=[], metavar="KEY=VALUE")
    benchmark.set_defaults(handler="benchmark")

    report = subparsers.add_parser("report", help="Write a compact pipeline report")
    report.add_argument("--output", default="results/reports/pipeline_report.json")
    report.add_argument("--run-path", default=None)
    report.add_argument("--dataset", default=None)
    report.add_argument("--training", default=None)
    report.add_argument("--benchmark", default=None)
    report.set_defaults(handler="report")

    return parser


def main(argv: Sequence[str] | None = None, *, commands: PipelineCommands | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command_handlers = commands or PipelineCommands()
    handler = getattr(command_handlers, args.handler)
    return int(handler(args) or 0)


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2))


def shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in argv)


if __name__ == "__main__":
    raise SystemExit(main())
