"""Validate and visualize merged oracle dataset splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = [
    "states",
    "next_states",
    "cmd_controls",
    "real_controls",
    "exec_residuals",
    "oracle_residuals",
    "terrain_features",
    "terrain_risk",
    "episode_ids",
    "steps",
]

EXPECTED_TRAILING_SHAPES = {
    "states": (6,),
    "next_states": (6,),
    "cmd_controls": (3,),
    "real_controls": (3,),
    "exec_residuals": (3,),
    "oracle_residuals": (3,),
    "terrain_features": (4,),
    "terrain_risk": (),
    "episode_ids": (),
    "steps": (),
}

FLOAT_FIELDS = {
    "states",
    "next_states",
    "cmd_controls",
    "real_controls",
    "exec_residuals",
    "oracle_residuals",
    "terrain_features",
    "terrain_risk",
}

SPLITS = ("train", "val", "test")


def validate_oracle_dataset(dataset_dir: Path, output_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _require_dataset_files(dataset_dir)
    split_manifest = json.loads((dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
    dataset_summary = json.loads((dataset_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    split_data = {split: _load_split(dataset_dir / f"{split}.npz") for split in SPLITS}

    split_shapes = {}
    num_transitions = {}
    num_episodes = {}
    nan_count = 0
    inf_count = 0
    valid = True
    residual_errors = []
    residual_targets = []
    for split, data in split_data.items():
        split_shapes[split] = {}
        split_valid, split_nan, split_inf = _validate_split_fields(data, split_shapes[split])
        valid = valid and split_valid
        nan_count += split_nan
        inf_count += split_inf
        num_transitions[split] = _field_length(data, "states")
        num_episodes[split] = _episode_count(data)
        expected_transitions = int(dataset_summary.get("splits", {}).get(split, {}).get("transitions", num_transitions[split]))
        if num_transitions[split] != expected_transitions:
            valid = False
        if _has_required_residual_fields(data):
            diff = data["exec_residuals"] - (data["real_controls"] - data["cmd_controls"])
            residual_errors.append(diff.reshape(-1))
            residual_targets.append(data["exec_residuals"].reshape(-1))

    residual_error = np.concatenate(residual_errors) if residual_errors else np.zeros(0, dtype=np.float32)
    residual_target = np.concatenate(residual_targets) if residual_targets else np.zeros(0, dtype=np.float32)
    max_error = float(np.max(np.abs(residual_error))) if residual_error.size else 0.0
    mean_error = float(np.mean(np.abs(residual_error))) if residual_error.size else 0.0
    zero_baseline_mse = float(np.mean(residual_target * residual_target)) if residual_target.size else 0.0
    leakage = _episode_leakage(split_manifest, split_data)
    valid = valid and nan_count == 0 and inf_count == 0 and max_error <= 1e-5 and bool(leakage["pass"])

    quality = {
        "split_shapes": split_shapes,
        "num_transitions": num_transitions,
        "num_episodes": num_episodes,
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "max_exec_residual_error": max_error,
        "mean_exec_residual_error": mean_error,
        "zero_residual_baseline_mse": zero_baseline_mse,
        "episode_leakage_check": leakage,
        "pass": bool(valid),
    }
    (output_dir / "dataset_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    if _can_plot(split_data):
        _plot_dataset_summary(split_data, quality, output_dir / "dataset_summary.png")
    return quality


def _require_dataset_files(dataset_dir: Path) -> None:
    required = [
        "train.npz",
        "val.npz",
        "test.npz",
        "split_manifest.json",
        "dataset_summary.json",
    ]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{dataset_dir} is missing split dataset files: {missing}. "
            "Run the dataset build step first, for example: "
            "python3 tools/fdm_mppi.py dataset build --input datasets/oracle_debug "
            "--output datasets/oracle_debug_splits"
        )


def _load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {field: np.asarray(data[field]) for field in data.files}


def _validate_split_fields(data: dict[str, np.ndarray], shapes: dict[str, list[int]]) -> tuple[bool, int, int]:
    valid = True
    nan_count = 0
    inf_count = 0
    transition_count = None
    for field in REQUIRED_FIELDS:
        if field not in data:
            shapes[field] = []
            valid = False
            continue
        arr = data[field]
        shapes[field] = list(arr.shape)
        if transition_count is None:
            transition_count = int(arr.shape[0])
        elif int(arr.shape[0]) != transition_count:
            valid = False
        if tuple(arr.shape[1:]) != EXPECTED_TRAILING_SHAPES[field]:
            valid = False
        if field in FLOAT_FIELDS and not np.issubdtype(arr.dtype, np.floating):
            valid = False
        if field not in FLOAT_FIELDS and not np.issubdtype(arr.dtype, np.integer):
            valid = False
        if np.issubdtype(arr.dtype, np.number):
            nan_count += int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
            inf_count += int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    return valid, nan_count, inf_count


def _field_length(data: dict[str, np.ndarray], field: str) -> int:
    if field not in data:
        return 0
    return int(data[field].shape[0])


def _episode_count(data: dict[str, np.ndarray]) -> int:
    if "episode_ids" not in data:
        return 0
    return int(len(set(data["episode_ids"].astype(int).tolist())))


def _has_required_residual_fields(data: dict[str, np.ndarray]) -> bool:
    return all(field in data for field in ("exec_residuals", "real_controls", "cmd_controls"))


def _can_plot(split_data: dict[str, dict[str, np.ndarray]]) -> bool:
    return all(all(field in data for field in REQUIRED_FIELDS) for data in split_data.values())


def _episode_leakage(split_manifest: dict, split_data: dict[str, dict[str, np.ndarray]]) -> dict:
    ids = {}
    mismatches = []
    for split in SPLITS:
        manifest_ids = set(int(value) for value in split_manifest.get(f"{split}_episode_ids", []))
        if "episode_ids" in split_data[split]:
            data_ids = set(int(value) for value in split_data[split]["episode_ids"].tolist())
        else:
            data_ids = set()
        if manifest_ids != data_ids:
            mismatches.append(
                {
                    "split": split,
                    "manifest_only": sorted(manifest_ids - data_ids),
                    "data_only": sorted(data_ids - manifest_ids),
                }
            )
        ids[split] = data_ids
    overlaps = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(ids[left] & ids[right])
        if overlap:
            overlaps.append({"splits": [left, right], "episode_ids": overlap})
    return {
        "pass": not overlaps and not mismatches,
        "overlaps": overlaps,
        "manifest_data_mismatch": mismatches,
    }


def _plot_dataset_summary(split_data: dict[str, dict[str, np.ndarray]], quality: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.reshape(-1)
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    for split, data in split_data.items():
        color = colors[split]
        axes[0].scatter(data["states"][:, 0], data["states"][:, 1], s=8, alpha=0.5, label=split, color=color)
        for idx, name in enumerate(["cmd_vx", "cmd_vy", "cmd_wz"]):
            axes[1].hist(data["cmd_controls"][:, idx], bins=30, alpha=0.35, color=color, label=split if idx == 0 else None)
        for idx, name in enumerate(["real_vx", "real_vy", "real_wz"]):
            axes[2].hist(data["real_controls"][:, idx], bins=30, alpha=0.35, color=color, label=split if idx == 0 else None)
        for idx, name in enumerate(["exec_du_vx", "exec_du_vy", "exec_du_wz"]):
            axes[3].hist(data["exec_residuals"][:, idx], bins=30, alpha=0.35, color=color, label=split if idx == 0 else None)
        axes[4].hist(data["terrain_risk"], bins=30, alpha=0.35, color=color, label=split)
        axes[5].hist(data["terrain_features"][:, 2], bins=30, alpha=0.35, color=color, label=f"{split} roughness")
        axes[5].hist(data["terrain_features"][:, 3], bins=30, alpha=0.25, color=color, histtype="step", label=f"{split} friction")
        lengths = [np.sum(data["episode_ids"] == episode_id) for episode_id in sorted(set(data["episode_ids"].tolist()))]
        axes[6].hist(lengths, bins=max(1, min(20, len(lengths))), alpha=0.35, color=color, label=split)
    axes[7].bar(list(quality["num_transitions"].keys()), list(quality["num_transitions"].values()), color=["tab:blue", "tab:orange", "tab:green"])
    axes[8].axis("off")
    axes[8].text(
        0.0,
        0.95,
        "\n".join(
            [
                f"pass: {quality['pass']}",
                f"nan_count: {quality['nan_count']}",
                f"inf_count: {quality['inf_count']}",
                f"max_exec_error: {quality['max_exec_residual_error']:.3e}",
                f"zero_baseline_mse: {quality['zero_residual_baseline_mse']:.3e}",
                f"leakage: {quality['episode_leakage_check']['pass']}",
            ]
        ),
        va="top",
        family="monospace",
    )
    titles = [
        "x-y coverage",
        "cmd control distributions",
        "real control distributions",
        "exec residual distributions",
        "terrain risk distribution",
        "roughness / friction distributions",
        "episode length distribution",
        "split transition counts",
        "quality summary",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        if title != "quality summary":
            ax.grid(True, alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
