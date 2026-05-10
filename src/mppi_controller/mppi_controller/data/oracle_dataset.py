"""Merge oracle episode npz files into train/val/test dataset splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ARRAY_FIELDS = [
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

FIELD_TRAILING_SHAPES = {
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

FIELD_DTYPES = {
    "states": np.float32,
    "next_states": np.float32,
    "cmd_controls": np.float32,
    "real_controls": np.float32,
    "exec_residuals": np.float32,
    "oracle_residuals": np.float32,
    "terrain_features": np.float32,
    "terrain_risk": np.float32,
    "episode_ids": np.int64,
    "steps": np.int64,
}


def build_oracle_dataset(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: int = 123,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    rows = _read_manifest(input_dir / "manifest.jsonl")
    usable_rows, skipped_rows = _filter_usable_rows(rows, input_dir)
    split_rows = _split_rows(
        usable_rows,
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        split_seed=int(split_seed),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = {}
    split_manifest = {
        "split_seed": int(split_seed),
        "train_episode_ids": [int(row["episode_id"]) for row in split_rows["train"]],
        "val_episode_ids": [int(row["episode_id"]) for row in split_rows["val"]],
        "test_episode_ids": [int(row["episode_id"]) for row in split_rows["test"]],
        "skipped_episode_ids": [int(row.get("episode_id", -1)) for row in skipped_rows],
    }

    total_transitions = 0
    for split_name, rows_for_split in split_rows.items():
        arrays = _merge_rows(rows_for_split)
        np.savez_compressed(output_dir / f"{split_name}.npz", **arrays)
        transitions = int(arrays["states"].shape[0])
        total_transitions += transitions
        split_summary[split_name] = {
            "episodes": len(rows_for_split),
            "transitions": transitions,
            "path": str(output_dir / f"{split_name}.npz"),
        }

    summary = {
        "total_input_episodes": len(rows),
        "usable_episodes": len(usable_rows),
        "skipped_episodes": len(skipped_rows),
        "total_transitions": total_transitions,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "split_seed": int(split_seed),
        "ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "splits": split_summary,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _filter_usable_rows(rows: list[dict], input_dir: Path) -> tuple[list[dict], list[dict]]:
    usable = []
    skipped = []
    for row in rows:
        path = _resolve_manifest_path(row.get("path", ""), input_dir)
        if bool(row.get("success", False)) and not bool(row.get("failed", False)) and path.exists():
            usable.append({**row, "path": str(path)})
        else:
            skipped.append(row)
    return usable, skipped


def _resolve_manifest_path(path_value: str, input_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    input_dir = Path(input_dir)
    if input_dir.name in path.parts:
        idx = len(path.parts) - 1 - list(reversed(path.parts)).index(input_dir.name)
        return input_dir / Path(*path.parts[idx + 1 :])
    return input_dir / path


def _split_rows(
    rows: list[dict],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> dict[str, list[dict]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0.0:
        raise ValueError("split ratios must sum to a positive value")
    normalized = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64) / ratio_sum
    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    counts = np.floor(normalized * len(rows)).astype(int)
    remainder = len(rows) - int(np.sum(counts))
    if remainder:
        fractional = normalized * len(rows) - counts
        for idx in np.argsort(-fractional)[:remainder]:
            counts[idx] += 1
    train_end = int(counts[0])
    val_end = train_end + int(counts[1])
    shuffled = [rows[int(idx)] for idx in indices]
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _merge_rows(rows: list[dict]) -> dict[str, np.ndarray]:
    arrays_by_field = {field: [] for field in ARRAY_FIELDS}
    for row in rows:
        with np.load(row["path"]) as episode:
            for field in ARRAY_FIELDS:
                if field not in episode.files:
                    raise ValueError(f"{row['path']} missing field {field}")
                arrays_by_field[field].append(np.asarray(episode[field]))

    merged = {}
    for field, arrays in arrays_by_field.items():
        if arrays:
            merged[field] = np.concatenate(arrays, axis=0).astype(FIELD_DTYPES[field], copy=False)
        else:
            merged[field] = np.empty((0, *FIELD_TRAILING_SHAPES[field]), dtype=FIELD_DTYPES[field])
    return merged
