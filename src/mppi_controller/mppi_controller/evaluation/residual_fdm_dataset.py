#!/usr/bin/env python3
"""Evaluate a trained residual FDM checkpoint on dataset splits."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from mppi_controller.core.residual_fdm_model import FEATURE_NAMES, ResidualFdmMlp
from mppi_controller.training.residual_fdm import (
    SPLITS,
    _axis_metric_fields,
    _zero_residual_mse_axis,
    current_git_metadata,
    load_residual_fdm_dataset,
)


def evaluate_residual_fdm_dataset(
    *,
    dataset_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path = "best_model.pt",
    normalization: str | Path = "normalization.npz",
    device: str = "cpu",
    command: str | None = None,
) -> dict:
    dataset_dir = Path(dataset_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_model_artifact_path(model_dir, checkpoint)
    normalization_path = resolve_model_artifact_path(model_dir, normalization)
    arrays = load_residual_fdm_dataset(dataset_dir)
    model, normalizer = load_model_and_normalizer(checkpoint_path, normalization_path, device=device)

    metrics = {
        "command": command,
        **current_git_metadata(),
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "normalization_path": str(normalization_path),
        "device": str(device),
        "feature_names": FEATURE_NAMES,
    }
    split_metrics = {}
    for split in SPLITS:
        split_result = evaluate_split(
            model=model,
            features=arrays[f"{split}_features"],
            targets=arrays[f"{split}_targets"],
            feature_mean=normalizer["feature_mean"],
            feature_std=normalizer["feature_std"],
            target_mean=normalizer["target_mean"],
            target_std=normalizer["target_std"],
            device=device,
        )
        split_metrics[split] = split_result
        metrics[f"{split}_mse"] = split_result["mse"]
        metrics[f"zero_residual_{split}_mse"] = split_result["zero_residual_mse"]
        metrics.update(_axis_metric_fields(split, split_result["mse_axis"], split_result["zero_residual_mse_axis"]))
    metrics["splits"] = {
        split: {
            key: [float(v) for v in value.tolist()] if isinstance(value, np.ndarray) else value
            for key, value in split_result.items()
        }
        for split, split_result in split_metrics.items()
    }

    (output_dir / "ood_residual_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model_and_normalizer(
    checkpoint_path: str | Path,
    normalization_path: str | Path,
    *,
    device: str,
) -> tuple[ResidualFdmMlp, dict[str, np.ndarray]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_dim = int(checkpoint["input_dim"])
    if input_dim != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} input features, checkpoint has {input_dim}")
    torch_device = torch.device(device)
    model = ResidualFdmMlp(input_dim=input_dim, hidden_dim=int(checkpoint["hidden_dim"])).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    normalizer_npz = np.load(normalization_path)
    normalizer = {
        "feature_mean": np.asarray(normalizer_npz["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(normalizer_npz["feature_std"], dtype=np.float32),
        "target_mean": np.asarray(normalizer_npz["target_mean"], dtype=np.float32),
        "target_std": np.asarray(normalizer_npz["target_std"], dtype=np.float32),
    }
    return model, normalizer


def evaluate_split(
    *,
    model: ResidualFdmMlp,
    features: np.ndarray,
    targets: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: str,
) -> dict:
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    standardized = ((features - feature_mean) / feature_std).astype(np.float32, copy=False)
    tensor = torch.as_tensor(standardized, dtype=torch.float32, device=torch.device(device))
    model.eval()
    with torch.no_grad():
        pred = model(tensor).detach().cpu().numpy()
    pred_raw = (pred * target_std + target_mean).astype(np.float32, copy=False)
    diff = pred_raw - targets
    mse_axis = np.mean(diff * diff, axis=0, dtype=np.float64).astype(np.float32)
    zero_axis = _zero_residual_mse_axis(targets)
    return {
        "transitions": int(features.shape[0]),
        "mse": float(np.mean(mse_axis)),
        "zero_residual_mse": float(np.mean(zero_axis)),
        "mse_axis": mse_axis,
        "zero_residual_mse_axis": zero_axis,
    }


def resolve_model_artifact_path(model_dir: str | Path, artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return Path(model_dir) / path


def shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--normalization", default="normalization.npz")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    metrics = evaluate_residual_fdm_dataset(
        dataset_dir=args.dataset,
        model_dir=args.model_dir,
        output_dir=args.output,
        checkpoint=args.checkpoint,
        normalization=args.normalization,
        device=args.device,
        command=shell_join([sys.executable, *sys.argv]),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
