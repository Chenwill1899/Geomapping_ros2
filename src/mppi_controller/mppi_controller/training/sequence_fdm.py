#!/usr/bin/env python3
"""Train a whole-sequence FDM from oracle dataset splits."""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from mppi_controller.core.sequence_fdm_model import (
    SEQUENCE_TARGET_COMPONENTS,
    SequenceFdmMlp,
    sequence_fdm_feature_names,
    sequence_fdm_target_names,
)

SPLITS = ("train", "val", "test")


def train_sequence_fdm(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    sequence_horizon: int = 25,
    include_history_controls: bool = True,
    history_steps: int = 1,
    epochs: int = 50,
    batch_size: int = 256,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 123,
    device: str = "cpu",
    tensorboard_log_dir: str | Path | None = None,
    command: str | None = None,
    argv: Sequence[str] | None = None,
) -> dict:
    _set_seed(seed)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir = Path(tensorboard_log_dir) if tensorboard_log_dir is not None else output_dir / "tensorboard"

    horizon = int(sequence_horizon)
    include_history = bool(include_history_controls)
    history_steps = int(history_steps)
    feature_names = sequence_fdm_feature_names(
        horizon,
        include_history_controls=include_history,
        include_history_steps=history_steps,
    )
    target_names = sequence_fdm_target_names(horizon)
    arrays = load_sequence_fdm_dataset(
        dataset_dir,
        sequence_horizon=horizon,
        include_history_controls=include_history,
        history_steps=history_steps,
    )

    x_mean, x_std = _normalization(arrays["train_features"])
    y_mean, y_std = _normalization(arrays["train_targets"])
    np.savez(
        output_dir / "normalization.npz",
        feature_mean=x_mean,
        feature_std=x_std,
        target_mean=y_mean,
        target_std=y_std,
        feature_names=np.asarray(feature_names),
        target_names=np.asarray(target_names),
        sequence_horizon=horizon,
        include_history_controls=include_history,
        history_steps=history_steps,
    )

    train_x = _standardize(arrays["train_features"], x_mean, x_std)
    train_y = _standardize(arrays["train_targets"], y_mean, y_std)
    val_x = _standardize(arrays["val_features"], x_mean, x_std)
    val_y = _standardize(arrays["val_targets"], y_mean, y_std)
    test_x = _standardize(arrays["test_features"], x_mean, x_std)
    test_y = _standardize(arrays["test_targets"], y_mean, y_std)

    torch_device = torch.device(device)
    model = SequenceFdmMlp(
        input_dim=train_x.shape[1],
        output_horizon=horizon,
        hidden_dim=int(hidden_dim),
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = nn.MSELoss()

    train_tensor_x = torch.as_tensor(train_x, dtype=torch.float32, device=torch_device)
    train_tensor_y = torch.as_tensor(train_y, dtype=torch.float32, device=torch_device)
    val_tensor_x = torch.as_tensor(val_x, dtype=torch.float32, device=torch_device)
    val_tensor_y = torch.as_tensor(val_y, dtype=torch.float32, device=torch_device)
    test_tensor_x = torch.as_tensor(test_x, dtype=torch.float32, device=torch_device)
    test_tensor_y = torch.as_tensor(test_y, dtype=torch.float32, device=torch_device)

    history_train = []
    history_val = []
    best_epoch = 0
    best_val_loss = float("inf")
    n_train = int(train_tensor_x.shape[0])
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
    try:
        for epoch in range(int(epochs)):
            model.train()
            order = torch.randperm(n_train, device=torch_device)
            batch_losses = []
            for start in range(0, n_train, int(batch_size)):
                idx = order[start : start + int(batch_size)]
                pred = model(train_tensor_x[idx])
                loss = loss_fn(pred, train_tensor_y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            val_loss = _eval_loss(model, loss_fn, val_tensor_x, val_tensor_y)
            history_train.append(train_loss)
            history_val.append(val_loss)
            step = epoch + 1
            if best_epoch == 0 or val_loss < best_val_loss:
                best_epoch = step
                best_val_loss = val_loss
                _save_checkpoint(
                    output_dir / "best_model.pt",
                    model=model,
                    input_dim=train_x.shape[1],
                    hidden_dim=hidden_dim,
                    epoch=best_epoch,
                    val_loss=best_val_loss,
                    checkpoint_type="best",
                    sequence_horizon=horizon,
                    feature_names=feature_names,
                    target_names=target_names,
                )
            writer.add_scalar("loss/train_standardized", train_loss, step)
            writer.add_scalar("loss/val_standardized", val_loss, step)
            writer.add_scalar("loss/best_val_standardized", best_val_loss, step)
            writer.add_scalar("checkpoint/best_epoch", best_epoch, step)
            writer.add_scalar("lr", float(optimizer.param_groups[0]["lr"]), step)
    finally:
        writer.flush()

    val_mse_axis = _eval_raw_mse_axis(model, val_tensor_x, arrays["val_targets"], y_mean, y_std)
    test_mse_axis = _eval_raw_mse_axis(model, test_tensor_x, arrays["test_targets"], y_mean, y_std)
    zero_val_mse_axis = _zero_prediction_mse_axis(arrays["val_targets"])
    zero_test_mse_axis = _zero_prediction_mse_axis(arrays["test_targets"])
    val_mse = float(np.mean(val_mse_axis))
    test_mse = float(np.mean(test_mse_axis))
    zero_val_mse = float(np.mean(zero_val_mse_axis))
    zero_test_mse = float(np.mean(zero_test_mse_axis))
    final_epoch = int(epochs)
    final_val_loss = float(history_val[-1]) if history_val else 0.0
    if best_epoch == 0:
        best_val_loss = final_val_loss

    metrics = {
        "command": command,
        "argv": [str(item) for item in argv] if argv is not None else None,
        "sys_argv": [str(item) for item in argv] if argv is not None else None,
        **current_git_metadata(),
        "dataset_dir": str(dataset_dir),
        "split_manifest_path": _artifact_path(dataset_dir / "split_manifest.json"),
        "dataset_summary_path": _artifact_path(dataset_dir / "dataset_summary.json"),
        "dataset_quality_path": _artifact_path(dataset_dir / "dataset_quality.json"),
        "output_dir": str(output_dir),
        "sequence_horizon": horizon,
        "include_history_controls": include_history,
        "history_steps": history_steps,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "device": str(device),
        "feature_names": feature_names,
        "target_names": target_names,
        "train_samples": int(arrays["train_features"].shape[0]),
        "val_samples": int(arrays["val_features"].shape[0]),
        "test_samples": int(arrays["test_features"].shape[0]),
        "train_loss": history_train,
        "val_loss": history_val,
        "test_loss": _eval_loss(model, loss_fn, test_tensor_x, test_tensor_y),
        "val_mse": val_mse,
        "test_mse": test_mse,
        "zero_prediction_val_mse": zero_val_mse,
        "zero_prediction_test_mse": zero_test_mse,
        "tensorboard_enabled": True,
        "tensorboard_log_dir": str(tensorboard_log_dir),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_epoch": int(final_epoch),
        "final_val_loss": float(final_val_loss),
        "checkpoint_policy": "best_model.pt tracks minimum validation standardized loss; model.pt stores final epoch",
        "best_checkpoint_path": str(output_dir / "best_model.pt"),
        "final_checkpoint_path": str(output_dir / "model.pt"),
    }
    metrics.update(_sequence_component_metric_fields("val", val_mse_axis, zero_val_mse_axis, horizon))
    metrics.update(_sequence_component_metric_fields("test", test_mse_axis, zero_test_mse_axis, horizon))
    _write_tensorboard_final_diagnostics(
        writer=writer,
        model=model,
        val_features=val_tensor_x,
        val_targets=arrays["val_targets"],
        target_mean=y_mean,
        target_std=y_std,
        target_names=target_names,
        metrics=metrics,
        seed=seed,
        step=int(epochs),
    )
    writer.close()
    _save_checkpoint(
        output_dir / "model.pt",
        model=model,
        input_dim=train_x.shape[1],
        hidden_dim=hidden_dim,
        epoch=final_epoch,
        val_loss=final_val_loss,
        checkpoint_type="final",
        metrics=metrics,
        sequence_horizon=horizon,
        feature_names=feature_names,
        target_names=target_names,
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_sequence_fdm_dataset(
    dataset_dir: str | Path,
    *,
    sequence_horizon: int,
    include_history_controls: bool = True,
    history_steps: int = 1,
) -> dict[str, np.ndarray]:
    dataset_dir = Path(dataset_dir)
    arrays = {}
    for split in SPLITS:
        with np.load(dataset_dir / f"{split}.npz") as data:
            features, targets = _sequence_features_targets_from_split(
                data,
                sequence_horizon=int(sequence_horizon),
                include_history_controls=bool(include_history_controls),
                history_steps=int(history_steps),
            )
        arrays[f"{split}_features"] = features
        arrays[f"{split}_targets"] = targets
    return arrays


def _angle_diff(base: float, target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    return np.arctan2(np.sin(target - float(base)), np.cos(target - float(base))).astype(np.float32, copy=False)


def _sequence_features_targets_from_split(
    data,
    *,
    sequence_horizon: int,
    include_history_controls: bool,
    history_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(data["states"], dtype=np.float32)
    next_states = np.asarray(data["next_states"], dtype=np.float32)
    cmd_controls = np.asarray(data["cmd_controls"], dtype=np.float32)
    terrain_features = np.asarray(data["terrain_features"], dtype=np.float32)
    terrain_risk = np.asarray(data["terrain_risk"], dtype=np.float32).reshape(-1)

    sample_count = int(states.shape[0])
    horizon = int(sequence_horizon)
    max_start = sample_count - horizon
    if max_start <= 0 or states.shape[1] != 6:
        feature_dim = len(
            sequence_fdm_feature_names(
                horizon,
                include_history_controls=include_history_controls,
                include_history_steps=history_steps,
            )
        )
        return (
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0, horizon * len(SEQUENCE_TARGET_COMPONENTS)), dtype=np.float32),
        )

    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for start in range(max_start):
        state = states[start]
        future_commands = cmd_controls[start : start + horizon]
        future_next_states = next_states[start : start + horizon]
        future_risk = terrain_risk[start + 1 : start + horizon + 1]
        if (
            future_commands.shape[0] != horizon
            or future_next_states.shape[0] != horizon
            or future_risk.shape[0] != horizon
        ):
            break

        rel_xy = future_next_states[:, :2] - state[:2]
        rel_yaw = _angle_diff(state[2], future_next_states[:, 2])
        target = np.stack([rel_xy[:, 0], rel_xy[:, 1], rel_yaw, future_risk], axis=1).reshape(-1)

        history = np.zeros((history_steps * 3,), dtype=np.float32)
        if include_history_controls and history_steps > 0:
            for index in range(history_steps):
                previous_step = start - index - 1
                if previous_step < 0:
                    break
                history[index * 3 : (index + 1) * 3] = cmd_controls[previous_step]

        per_step_features = []
        terrain_step = np.concatenate([terrain_features[start], np.asarray([terrain_risk[start]], dtype=np.float32)])
        for step in range(horizon):
            per_step_features.append(np.concatenate([future_commands[step], terrain_step]))
        feature = np.concatenate([state, history, np.concatenate(per_step_features, axis=0)]).astype(
            np.float32,
            copy=False,
        )
        feature_rows.append(feature)
        target_rows.append(target.astype(np.float32, copy=False))

    assert len(feature_rows) == len(target_rows)
    return np.asarray(feature_rows, dtype=np.float32), np.asarray(target_rows, dtype=np.float32)


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(values, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32, copy=False)


def _eval_loss(model: nn.Module, loss_fn: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> float:
    if int(features.shape[0]) == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(features), targets).detach().cpu())


def _eval_raw_mse_axis(
    model: nn.Module,
    features: torch.Tensor,
    targets: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    targets_arr = np.asarray(targets, dtype=np.float32)
    if int(features.shape[0]) == 0:
        return np.zeros(targets_arr.shape[1] if targets_arr.ndim == 2 else len(target_mean), dtype=np.float32)
    pred_raw = _predict_raw(model, features, target_mean, target_std)
    diff = pred_raw - targets_arr
    return np.mean(diff * diff, axis=0, dtype=np.float64).astype(np.float32)


def _write_tensorboard_final_diagnostics(
    *,
    writer: SummaryWriter,
    model: nn.Module,
    val_features: torch.Tensor,
    val_targets: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    target_names: Sequence[str],
    metrics: dict,
    seed: int,
    step: int,
) -> None:
    writer.add_scalar("mse/val_raw", float(metrics["val_mse"]), step)
    writer.add_scalar("mse/test_raw", float(metrics["test_mse"]), step)
    writer.add_scalar("baseline/zero_prediction_val_mse", float(metrics["zero_prediction_val_mse"]), step)
    writer.add_scalar("baseline/zero_prediction_test_mse", float(metrics["zero_prediction_test_mse"]), step)
    if int(val_features.shape[0]) == 0:
        writer.flush()
        return
    predictions = _predict_raw(model, val_features, target_mean, target_std)
    targets = np.asarray(val_targets, dtype=np.float32)
    indices = _diagnostic_sample_indices(len(targets), seed=seed, max_points=5000)
    pred_sample = predictions[indices]
    target_sample = targets[indices]
    selected_dims = _diagnostic_target_indices(len(target_names))
    writer.add_figure(
        "diagnostics/val_prediction_vs_target",
        _prediction_scatter_figure(pred_sample, target_sample, target_names, selected_dims),
        global_step=step,
        close=True,
    )
    writer.add_figure(
        "diagnostics/val_error_histogram",
        _error_histogram_figure(pred_sample - target_sample, target_names, selected_dims),
        global_step=step,
        close=True,
    )
    writer.flush()


def _predict_raw(
    model: nn.Module,
    features: torch.Tensor,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(features).detach().cpu().numpy()
    return (pred * target_std + target_mean).astype(np.float32, copy=False)


def _diagnostic_sample_indices(count: int, *, seed: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(count, size=int(max_points), replace=False))


def _diagnostic_target_indices(target_count: int) -> list[int]:
    if target_count <= 6:
        return list(range(target_count))
    last_step = target_count - len(SEQUENCE_TARGET_COMPONENTS)
    return [0, 1, 2, 3, last_step, last_step + 3]


def _prediction_scatter_figure(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: Sequence[str],
    selected_dims: Sequence[int],
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(selected_dims), figsize=(3.5 * len(selected_dims), 3.5))
    axes = np.atleast_1d(axes)
    for axis, target_idx in zip(axes, selected_dims):
        axis.scatter(targets[:, target_idx], predictions[:, target_idx], s=6, alpha=0.35)
        low = float(min(np.min(targets[:, target_idx]), np.min(predictions[:, target_idx])))
        high = float(max(np.max(targets[:, target_idx]), np.max(predictions[:, target_idx])))
        axis.plot([low, high], [low, high], color="black", linewidth=1.0, alpha=0.6)
        axis.set_title(str(target_names[target_idx]))
        axis.set_xlabel("target")
        axis.set_ylabel("prediction")
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _error_histogram_figure(errors: np.ndarray, target_names: Sequence[str], selected_dims: Sequence[int]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(selected_dims), figsize=(3.5 * len(selected_dims), 3.5))
    axes = np.atleast_1d(axes)
    for axis, target_idx in zip(axes, selected_dims):
        axis.hist(errors[:, target_idx], bins=40, alpha=0.8)
        axis.set_title(f"{target_names[target_idx]} error")
        axis.set_xlabel("prediction - target")
        axis.set_ylabel("count")
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _zero_prediction_mse_axis(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    if targets.size == 0:
        return np.zeros(targets.shape[1] if targets.ndim == 2 else 0, dtype=np.float32)
    return np.mean(targets * targets, axis=0, dtype=np.float64).astype(np.float32)


def _sequence_component_metric_fields(
    split: str,
    mse_axis: np.ndarray,
    zero_mse_axis: np.ndarray,
    sequence_horizon: int,
) -> dict:
    fields: dict[str, float | dict[str, float]] = {}
    horizon = int(sequence_horizon)
    component_count = len(SEQUENCE_TARGET_COMPONENTS)
    if mse_axis.size != horizon * component_count:
        fields[f"{split}_mse_relative_improvement_pct"] = _improvement_pct(
            float(np.mean(zero_mse_axis)) if zero_mse_axis.size else 0.0,
            float(np.mean(mse_axis)) if mse_axis.size else 0.0,
        )
        return fields

    mse_by_step = mse_axis.reshape(horizon, component_count)
    zero_by_step = zero_mse_axis.reshape(horizon, component_count)
    reductions: dict[str, float] = {}
    for idx, component in enumerate(SEQUENCE_TARGET_COMPONENTS):
        mse = float(np.mean(mse_by_step[:, idx]))
        zero_mse = float(np.mean(zero_by_step[:, idx]))
        fields[f"{split}_mse_{component}"] = mse
        fields[f"{split}_rmse_{component}"] = float(np.sqrt(mse))
        fields[f"zero_prediction_{split}_mse_{component}"] = zero_mse
        reductions[component] = _improvement_pct(zero_mse, mse)

    baseline = float(np.mean(zero_mse_axis)) if zero_mse_axis.size else 0.0
    candidate = float(np.mean(mse_axis)) if mse_axis.size else 0.0
    reduction = _improvement_pct(baseline, candidate)
    fields[f"per_component_{split}_mse_reduction_pct"] = reductions
    fields[f"{split}_mse_relative_improvement_pct"] = reduction
    fields[f"overall_{split}_mse_reduction_pct"] = reduction
    fields[f"overall_{split}_improvement_x"] = float(baseline / candidate) if candidate > 1e-12 else 0.0
    return fields


def _improvement_pct(baseline: float, candidate: float) -> float:
    if baseline <= 1e-12:
        return 0.0
    return float((1.0 - candidate / baseline) * 100.0)


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    input_dim: int,
    hidden_dim: int,
    epoch: int,
    val_loss: float,
    checkpoint_type: str,
    metrics: dict | None = None,
    sequence_horizon: int,
    feature_names: list[str] | tuple[str, ...],
    target_names: list[str] | tuple[str, ...],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "feature_names": np.asarray(feature_names),
        "target_names": np.asarray(target_names),
        "target_dim": int(len(target_names)),
        "sequence_horizon": int(sequence_horizon),
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "checkpoint_type": str(checkpoint_type),
    }
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, path)


def current_git_metadata() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "git_sha": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git_output(repo_root, "branch", "--show-current"),
        "git_dirty": _git_dirty(repo_root),
    }


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = result.stdout.strip()
    return output or None


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _artifact_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-horizon", type=int, default=25)
    parser.add_argument("--sequence-include-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sequence-history-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensorboard-log-dir", default=None)
    args = parser.parse_args(argv)

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
        command=shell_join([sys.executable, *sys.argv]),
        argv=[sys.executable, *sys.argv],
    )
    print(json.dumps(metrics, indent=2))


def shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in argv)


if __name__ == "__main__":
    main()
