"""Small JSON report for the recommended FDM-MPPI pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def write_pipeline_report(
    *,
    output_path: str | Path,
    run_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    training_path: str | Path | None = None,
    benchmark_path: str | Path | None = None,
) -> dict:
    report = {
        "run": _artifact_group(run_path, ["summary.json", "trajectory.csv", "controls.csv", "animation.gif"]),
        "dataset": _artifact_group(dataset_path, ["manifest.jsonl", "summary.json", "split_manifest.json", "dataset_summary.json", "dataset_quality.json"]),
        "training": _artifact_group(training_path, ["metrics.json", "best_model.pt", "model.pt", "normalization.npz"]),
        "benchmark": _artifact_group(benchmark_path, ["stage5_benchmark_summary.json", "benchmark_summary.json"]),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"output_path": str(output_path), **report}


def _artifact_group(root: str | Path | None, names: list[str]) -> dict:
    if root is None:
        return {"root": None, "artifacts": {}}
    root_path = Path(root)
    return {
        "root": str(root_path),
        "artifacts": {name: str(root_path / name) if (root_path / name).exists() else None for name in names},
    }
