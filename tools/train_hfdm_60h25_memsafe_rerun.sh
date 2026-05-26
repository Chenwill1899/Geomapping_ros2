#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./tools/train_hfdm_60h25_memsafe_rerun.sh

Environment overrides:
  RUN_ROOT=/home/mexxiie/prj/Geomapping_ros2/results/hfdm_training/geomapping_data1_60_h25_30hz_20260525_130644
  CONFIG=$RUN_ROOT/run/config.yaml
  TRAIN_LOG=$RUN_ROOT/logs/03_train_rerun_memsafe.log
  BATCH_SIZE=128              训练 batch size（建议 64~128）
  NUM_WORKERS=0               Dataset worker 数（建议 0）
  EPOCHS=20                   训练 epoch
  DEVICE=auto                  auto|cpu|cuda
  HFDM_EPISODE_CACHE_SIZE=64   每个进程最多缓存的 episode 数（建议 32~128）
  SHUFFLE_TRAIN=0             训练 DataLoader 是否 shuffle；0 更快，1 更随机
  TRAIN_PID=$RUN_ROOT/train_resume_memsafe.pid
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/results/hfdm_training/geomapping_data1_60_h25_30hz_20260525_130644}"
CONFIG="${CONFIG:-$RUN_ROOT/run/config.yaml}"
TRAIN_LOG="${TRAIN_LOG:-$RUN_ROOT/logs/03_train_rerun_memsafe.log}"
TRAIN_PID="${TRAIN_PID:-$RUN_ROOT/train_resume_memsafe.pid}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-20}"
DEVICE="${DEVICE:-auto}"
HFDM_EPISODE_CACHE_SIZE="${HFDM_EPISODE_CACHE_SIZE:-64}"
SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-0}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[hfdm_memsafe] error: config not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -d "$RUN_ROOT/dataset" ]]; then
  echo "[hfdm_memsafe] error: dataset not found: $RUN_ROOT/dataset" >&2
  exit 1
fi
if [[ ! -f "$RUN_ROOT/dataset/manifest.json" ]]; then
  echo "[hfdm_memsafe] error: dataset manifest missing: $RUN_ROOT/dataset/manifest.json" >&2
  exit 1
fi

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
fi

mkdir -p "$(dirname "$TRAIN_LOG")"
CONFIG_BAK="$CONFIG.$(date +%Y%m%d_%H%M%S).bak"
cp "$CONFIG" "$CONFIG_BAK"
echo "[hfdm_memsafe] config backup: $CONFIG_BAK"

export PYTHONPATH="/home/mexxiie/prj/high_level_fdm/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64,garbage_collection_threshold:0.8}"

echo "[hfdm_memsafe] before tweak mem=$(free -h | awk 'NR==2{print $3\"/\"$2\" used/total\"}')"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[hfdm_memsafe] nvidia-smi:"
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits
fi

echo "[hfdm_memsafe] patch config: batch=$BATCH_SIZE workers=$NUM_WORKERS epochs=$EPOCHS device=$DEVICE"
python3 - "$CONFIG" "$BATCH_SIZE" "$NUM_WORKERS" "$EPOCHS" "$DEVICE" "$SHUFFLE_TRAIN" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
epochs = int(sys.argv[4])
device = sys.argv[5]
shuffle_train = int(sys.argv[6])

cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
cfg.setdefault("train", {})
cfg["train"]["batch_size"] = batch_size
cfg["train"]["num_workers"] = num_workers
cfg["train"]["epochs"] = epochs
cfg["train"]["device"] = device
cfg.setdefault("data", {})
cfg["data"]["shuffle"] = bool(shuffle_train)

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[hfdm_memsafe] config patched: {cfg_path}")
PY

export HFDM_EPISODE_CACHE_SIZE="$HFDM_EPISODE_CACHE_SIZE"
echo "[hfdm_memsafe] data shuffle=$SHUFFLE_TRAIN episode_cache_size=$HFDM_EPISODE_CACHE_SIZE"

echo "[hfdm_memsafe] start time: $(date '+%F %T')"
echo "[hfdm_memsafe] logging to: $TRAIN_LOG"
echo "[hfdm_memsafe] pid file: $TRAIN_PID"

python3 -m hfdm.cli.train --config "$CONFIG" 2>&1 | tee "$TRAIN_LOG" &
TRAIN_PID_VAL=$!
echo "$TRAIN_PID_VAL" > "$TRAIN_PID"
echo "[hfdm_memsafe] train pid=$TRAIN_PID_VAL"

wait "$TRAIN_PID_VAL"
exit_code=$?

if [[ $exit_code -ne 0 ]]; then
  echo "[hfdm_memsafe] train failed with exit_code=$exit_code at $(date '+%F %T')."
  echo "[hfdm_memsafe] check kernel OOM/系统杀死: journalctl -k --since '1 minute ago' | rg -i 'Out of memory|Killed process|oom'"
  exit $exit_code
fi

BEST_CKPT="$RUN_ROOT/run/checkpoints/best.pt"
if [[ -f "$BEST_CKPT" ]]; then
  echo "[hfdm_memsafe] done: best checkpoint exists -> $BEST_CKPT"
else
  echo "[hfdm_memsafe] done but best checkpoint missing: $BEST_CKPT"
  exit 1
fi
