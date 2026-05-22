#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./tools/train_data1_seqfdm_v2.sh

Environment overrides:
  DATA_ROOT=results/data1                 Raw dataset root.
  OUTPUT_ROOT=results/fdm_training        Training runs root and TensorBoard logdir.
  OUT=<path>                              Exact output directory. Defaults to timestamped OUTPUT_ROOT child.
  START_TENSORBOARD=1                     Start TensorBoard in background when 1.
  TB_HOST=0.0.0.0                         TensorBoard host.
  TB_PORT=6006                            TensorBoard port.
  DEVICE=auto                             auto, cpu, or cuda. auto falls back to cpu when nvidia-smi fails.
  STRIDE=5                                Sliding-window stride for raw episodes.
  HORIZON=20                              Max sequence horizon for raw window extraction.
  BATCH_SIZE=256                          Training batch size.
  STATUS_FILTER=all                       all or success.
  MAX_EPISODES=0                          0 means all episodes.
  DRY_RUN=0                               1 builds windows and exits before training.
  PHASES=5:20:1e-3,10:20:5e-4,20:40:1e-4 Curriculum phases horizon:epochs:lr.
  HIDDEN_DIMS=256,256,256                 MLP hidden dimensions.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

set +u
if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi
if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
fi
set -u

export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/src/mppi_controller}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcfg}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
mkdir -p "$MPLCONFIGDIR"

DATA_ROOT="${DATA_ROOT:-results/data1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fdm_training}"
RUN_NAME="${RUN_NAME:-data1_seqfdm_v2_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$OUTPUT_ROOT/$RUN_NAME}"
START_TENSORBOARD="${START_TENSORBOARD:-1}"
TB_HOST="${TB_HOST:-0.0.0.0}"
TB_PORT="${TB_PORT:-6006}"

STRIDE="${STRIDE:-5}"
HORIZON="${HORIZON:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
STATUS_FILTER="${STATUS_FILTER:-all}"
MAX_EPISODES="${MAX_EPISODES:-0}"
DRY_RUN="${DRY_RUN:-0}"
PHASES="${PHASES:-5:20:1e-3,10:20:5e-4,20:40:1e-4}"
HIDDEN_DIMS="${HIDDEN_DIMS:-256,256,256}"

mkdir -p "$OUT" "$OUTPUT_ROOT"
if [[ "$DEVICE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    DEVICE="auto"
  else
    DEVICE="cpu"
    echo "[train_data1] nvidia-smi unavailable or unhealthy; using DEVICE=cpu"
  fi
fi
export DATA_ROOT OUTPUT_ROOT OUT START_TENSORBOARD TB_HOST TB_PORT
export STRIDE HORIZON BATCH_SIZE DEVICE STATUS_FILTER MAX_EPISODES DRY_RUN PHASES HIDDEN_DIMS

echo "[train_data1] data_root=$DATA_ROOT"
echo "[train_data1] output=$OUT"
echo "[train_data1] tensorboard_logdir=$OUTPUT_ROOT"
echo "[train_data1] tensorboard_url=http://localhost:$TB_PORT"

if [[ "$START_TENSORBOARD" == "1" ]]; then
  if pgrep -af "tensorboard .*--port[ =]$TB_PORT|tensorboard .*port=$TB_PORT" >/dev/null 2>&1; then
    echo "[train_data1] tensorboard already running on port $TB_PORT"
  elif command -v tensorboard >/dev/null 2>&1; then
    nohup tensorboard --logdir "$OUTPUT_ROOT" --host "$TB_HOST" --port "$TB_PORT" >"$OUT/tensorboard.log" 2>&1 &
    echo "$!" > "$OUT/tensorboard.pid"
    echo "[train_data1] tensorboard_started pid=$(cat "$OUT/tensorboard.pid") log=$OUT/tensorboard.log"
  else
    echo "[train_data1] tensorboard command not found; install tensorboard or run: python3 -m tensorboard.main --logdir $OUTPUT_ROOT --host $TB_HOST --port $TB_PORT" >&2
  fi
fi

python3 - <<'PY' 2>&1 | tee "$OUT/train.log"
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from mppi_controller.data.sequence_fdm_collector import build_sequence_fdm_windows_from_raw_episode
from mppi_controller.training.sequence_fdm_v2 import train_sequence_fdm_v2


def parse_phases(value: str) -> list[tuple[int, int, float]]:
    phases: list[tuple[int, int, float]] = []
    for raw_phase in value.split(","):
        raw_phase = raw_phase.strip()
        if not raw_phase:
            continue
        horizon_s, epochs_s, lr_s = raw_phase.split(":")
        phases.append((int(horizon_s), int(epochs_s), float(lr_s)))
    if not phases:
        raise ValueError("PHASES must contain at least one horizon:epochs:lr entry")
    return phases


def parse_hidden_dims(value: str) -> list[int]:
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims:
        raise ValueError("HIDDEN_DIMS must contain at least one integer")
    return dims


root = Path(os.environ["DATA_ROOT"])
out = Path(os.environ["OUT"])
stride = int(os.environ["STRIDE"])
horizon = int(os.environ["HORIZON"])
batch_size = int(os.environ["BATCH_SIZE"])
status_filter = os.environ["STATUS_FILTER"].strip().lower()
max_episodes = int(os.environ["MAX_EPISODES"])
dry_run = os.environ["DRY_RUN"] == "1"
device_env = os.environ["DEVICE"].strip().lower()
device = "cuda" if device_env == "auto" and torch.cuda.is_available() else device_env
if device == "auto":
    device = "cpu"
if device == "cuda" and not torch.cuda.is_available():
    print("[train_data1] requested DEVICE=cuda but torch.cuda.is_available() is false; falling back to cpu", flush=True)
    device = "cpu"

if status_filter not in {"all", "success"}:
    raise ValueError("STATUS_FILTER must be all or success")
if not root.exists():
    raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")

episodes = sorted(root.glob("seed_*/episodes/episode_*"))
if max_episodes > 0:
    episodes = episodes[:max_episodes]

windows: list[dict] = []
skipped: list[dict] = []
status_counts: dict[str, int] = {}

for idx, episode_dir in enumerate(episodes, 1):
    episode_json = episode_dir / "episode.json"
    status = "missing"
    if episode_json.exists():
        try:
            episode_data = json.loads(episode_json.read_text(encoding="utf-8"))
            status = str(episode_data.get("status", "unknown"))
            reason = episode_data.get("failure_reason")
            if reason:
                status = f"{status}:{reason}"
        except Exception as exc:
            skipped.append({"episode": str(episode_dir), "error": f"episode_json:{exc!r}"})
            continue
    status_counts[status] = status_counts.get(status, 0) + 1
    if status_filter == "success" and status != "success":
        skipped.append({"episode": str(episode_dir), "status": status, "error": "filtered"})
        continue
    try:
        episode_windows = build_sequence_fdm_windows_from_raw_episode(
            episode_dir,
            horizon_steps=horizon,
            stride=stride,
        )
    except Exception as exc:
        skipped.append({"episode": str(episode_dir), "status": status, "error": repr(exc)})
        continue
    if not episode_windows:
        skipped.append({"episode": str(episode_dir), "status": status, "error": "no_windows"})
        continue
    windows.extend(episode_windows)
    if idx % 50 == 0:
        print(
            f"[build] episodes={idx}/{len(episodes)} windows={len(windows)} skipped={len(skipped)}",
            flush=True,
        )

summary = {
    "data_root": str(root),
    "output_dir": str(out),
    "episodes_seen": len(episodes),
    "status_counts": status_counts,
    "windows": len(windows),
    "skipped_count": len(skipped),
    "skipped": skipped,
    "stride": stride,
    "horizon": horizon,
    "batch_size": batch_size,
    "device": device,
    "status_filter": status_filter,
    "phases": os.environ["PHASES"],
    "hidden_dims": os.environ["HIDDEN_DIMS"],
    "dry_run": dry_run,
}
out.mkdir(parents=True, exist_ok=True)
(out / "data1_training_input_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "skipped"}, indent=2), flush=True)

if not windows:
    raise RuntimeError("no training windows built from data1 episodes")
if dry_run:
    print("[train_data1] DRY_RUN=1; skipping training", flush=True)
    raise SystemExit(0)

metrics = train_sequence_fdm_v2(
    windows=windows,
    output_dir=out,
    hidden_dims=parse_hidden_dims(os.environ["HIDDEN_DIMS"]),
    curriculum_phases=parse_phases(os.environ["PHASES"]),
    batch_size=batch_size,
    device=device,
    use_tensorboard=True,
)
(out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2), flush=True)
print(f"[train_data1] done output={out}", flush=True)
print(f"[train_data1] tensorboard_run_dir={out / 'runs'}", flush=True)
PY
