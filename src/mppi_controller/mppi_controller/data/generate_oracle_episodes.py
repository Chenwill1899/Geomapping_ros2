#!/usr/bin/env python3
"""Collect oracle episodes and write manifest/summary files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
from typing import Callable

from mppi_controller.data.collect_oracle_episode import collect_oracle_episode


Collector = Callable[..., dict]


def generate_oracle_episodes(
    *,
    config_path: str | Path,
    episodes: int,
    base_seed: int,
    output_dir: str | Path,
    backend: str | None = None,
    collector: Collector = collect_oracle_episode,
    num_workers: int = 1,
) -> dict:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    episodes_dir = output_dir / "episodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    total_episodes = int(episodes)
    num_workers = int(num_workers)
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    episode_args = [
        {
            "config_path": config_path,
            "episode_id": episode_id,
            "seed": int(base_seed) + episode_id,
            "episode_path": episodes_dir / f"episode_{episode_id:06d}.npz",
            "manifest_path": Path("episodes") / f"episode_{episode_id:06d}.npz",
            "backend": backend,
            "collector": collector,
        }
        for episode_id in range(total_episodes)
    ]

    if num_workers == 1:
        rows = [_run_episode_task(args) for args in episode_args]
    else:
        rows = []
        executor_kwargs = {}
        if collector is collect_oracle_episode:
            executor_kwargs["mp_context"] = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, **executor_kwargs) as executor:
            futures = [executor.submit(_run_episode_task, args) for args in episode_args]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())

    rows.sort(key=lambda row: int(row["episode_id"]))

    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    success_episodes = sum(1 for row in rows if bool(row["success"]) and not bool(row["failed"]))
    failed_episodes = total_episodes - success_episodes
    total_transitions = sum(int(row["num_transitions"]) for row in rows)
    summary = {
        "total_episodes": total_episodes,
        "success_episodes": success_episodes,
        "failed_episodes": failed_episodes,
        "total_transitions": total_transitions,
        "success_rate": success_episodes / total_episodes if total_episodes else 0.0,
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "base_seed": int(base_seed),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_episode_task(args: dict) -> dict:
    episode_id = int(args["episode_id"])
    seed = int(args["seed"])
    episode_path = Path(args["episode_path"])
    manifest_path = Path(args["manifest_path"])
    try:
        metadata = args["collector"](
            config_path=Path(args["config_path"]),
            episode_id=episode_id,
            seed=seed,
            output_path=episode_path,
            backend=args["backend"],
        )
        return _success_row_from_metadata(
            episode_id=episode_id,
            seed=seed,
            episode_path=episode_path,
            manifest_path=manifest_path,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - exact failures are exercised through tests.
        return _failure_row(
            episode_id=episode_id,
            seed=seed,
            manifest_path=manifest_path,
            error=_format_error(exc),
        )


def _success_row_from_metadata(
    *,
    episode_id: int,
    seed: int,
    episode_path: Path,
    manifest_path: Path,
    metadata: dict,
) -> dict:
    return {
        "episode_id": episode_id,
        "seed": seed,
        "path": str(manifest_path),
        "success": bool(metadata.get("success", False)),
        "failed": bool(metadata.get("failed", False)),
        "num_transitions": int(metadata.get("num_transitions", 0)),
        "start_goal_distance": _optional_float(metadata.get("start_goal_distance")),
        "final_distance": _optional_float(metadata.get("final_distance")),
        "min_obstacle_clearance": _optional_float(metadata.get("min_obstacle_clearance")),
    }


def _failure_row(*, episode_id: int, seed: int, manifest_path: Path, error: str) -> dict:
    return {
        "episode_id": episode_id,
        "seed": seed,
        "path": str(manifest_path),
        "success": False,
        "failed": True,
        "num_transitions": 0,
        "start_goal_distance": None,
        "final_distance": None,
        "min_obstacle_clearance": None,
        "error": error,
    }


def _format_error(exc: Exception) -> str:
    return str(exc) or repr(exc)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["cuda", "numpy"], default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args(argv)

    summary = generate_oracle_episodes(
        config_path=args.config,
        episodes=args.episodes,
        base_seed=args.base_seed,
        output_dir=args.output,
        backend=args.backend,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
