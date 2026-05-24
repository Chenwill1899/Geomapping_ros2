#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./tools/train_hfdm_data1_150_h25.sh

Environment overrides:
  DATA_ROOT=results/data1
  HIGH_LEVEL_FDM_ROOT=/home/mexxiie/prj/high_level_fdm
  BASE_CONFIG=/home/mexxiie/prj/high_level_fdm/configs/experiments/geomapping_h25.yaml
  OUTPUT_ROOT=results/hfdm_training
  RUN_ID=geomapping_data1_150_h25_<timestamp>
  RUN_ROOT=<OUTPUT_ROOT>/<RUN_ID>
  TRAIN_EPISODES=1200
  VAL_EPISODES=300
  MIN_EPISODES=1500
  DEVICE=auto                         auto, cpu, or cuda
  START_TENSORBOARD=0                 set 1 to start TensorBoard after export
  TB_HOST=0.0.0.0
  TB_PORT=6007
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATA_ROOT="${DATA_ROOT:-results/data1}"
HIGH_LEVEL_FDM_ROOT="${HIGH_LEVEL_FDM_ROOT:-/home/mexxiie/prj/high_level_fdm}"
BASE_CONFIG="${BASE_CONFIG:-$HIGH_LEVEL_FDM_ROOT/configs/experiments/geomapping_h25.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hfdm_training}"
RUN_ID="${RUN_ID:-geomapping_data1_150_h25_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/$OUTPUT_ROOT/$RUN_ID}"
TRAIN_EPISODES="${TRAIN_EPISODES:-1200}"
VAL_EPISODES="${VAL_EPISODES:-300}"
MIN_EPISODES="${MIN_EPISODES:-1500}"
DEVICE="${DEVICE:-auto}"
START_TENSORBOARD="${START_TENSORBOARD:-0}"
TB_HOST="${TB_HOST:-0.0.0.0}"
TB_PORT="${TB_PORT:-6007}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[hfdm_data1_150] error: DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$HIGH_LEVEL_FDM_ROOT/src" ]]; then
  echo "[hfdm_data1_150] error: high_level_fdm src not found: $HIGH_LEVEL_FDM_ROOT/src" >&2
  exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[hfdm_data1_150] error: BASE_CONFIG not found: $BASE_CONFIG" >&2
  exit 1
fi

export PYTHONPATH="$HIGH_LEVEL_FDM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcfg}"
mkdir -p "$MPLCONFIGDIR" "$RUN_ROOT"/{raw,dataset,run,logs,export}

echo "[hfdm_data1_150] data_root=$DATA_ROOT"
echo "[hfdm_data1_150] run_root=$RUN_ROOT"

python3 - "$DATA_ROOT" "$MIN_EPISODES" <<'PY'
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

root = Path(sys.argv[1])
minimum = int(sys.argv[2])
counts = defaultdict(int)
status = Counter()
for episode_json in root.glob("seed_*/episodes/episode_*/episode.json"):
    data = json.loads(episode_json.read_text(encoding="utf-8"))
    counts[int(data["seed"])] += 1
    status[str(data.get("status"))] += 1
total = sum(counts.values())
complete = sum(1 for count in counts.values() if count >= 10)
partial = {seed: count for seed, count in sorted(counts.items()) if count < 10}
print(
    "[hfdm_data1_150] episode_json="
    f"{total} unique_seeds={len(counts)} complete_10_goal_seeds={complete} "
    f"partial={partial} status={dict(status)}",
    flush=True,
)
if total < minimum:
    raise SystemExit(f"need at least {minimum} episodes, found {total}")
PY

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
fi
echo "[hfdm_data1_150] device=$DEVICE"

cp "$BASE_CONFIG" "$RUN_ROOT/config.yaml"
python3 - "$RUN_ROOT/config.yaml" "$ROOT_DIR/$DATA_ROOT" "$RUN_ROOT" "$TRAIN_EPISODES" "$VAL_EPISODES" "$DEVICE" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
data_root = Path(sys.argv[2]).resolve()
run_root = Path(sys.argv[3]).resolve()
train_episodes = int(sys.argv[4])
val_episodes = int(sys.argv[5])
device = sys.argv[6]

cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
cfg["run_dir"] = str(run_root / "run")
cfg["data"]["geomapping_root"] = str(data_root)
cfg["data"]["raw_dir"] = str(run_root / "raw")
cfg["data"]["dataset_dir"] = str(run_root / "dataset")
cfg["data"]["num_train_episodes"] = train_episodes
cfg["data"]["num_val_episodes"] = val_episodes
cfg["data"].setdefault("tensorboard", {})
cfg["data"]["tensorboard"]["enabled"] = True
cfg["data"]["tensorboard"]["log_dir"] = str(run_root / "tensorboard" / "dataset")
cfg["train"]["device"] = device
config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[hfdm_data1_150] config={config_path}", flush=True)
PY

echo "[hfdm_data1_150] converting Geomapping episodes"
python3 -m hfdm.cli.convert_geomapping \
  --config "$RUN_ROOT/config.yaml" \
  --input-root "$ROOT_DIR/$DATA_ROOT" \
  --output-dir "$RUN_ROOT/raw" \
  2>&1 | tee "$RUN_ROOT/logs/01_convert.log"

echo "[hfdm_data1_150] building H-FDM dataset"
python3 -m hfdm.cli.build_dataset \
  --config "$RUN_ROOT/config.yaml" \
  2>&1 | tee "$RUN_ROOT/logs/02_build_dataset.log"

echo "[hfdm_data1_150] training H-FDM H25"
python3 -m hfdm.cli.train \
  --config "$RUN_ROOT/config.yaml" \
  2>&1 | tee "$RUN_ROOT/logs/03_train.log"

BEST_CKPT="$RUN_ROOT/run/checkpoints/best.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "[hfdm_data1_150] error: missing best checkpoint: $BEST_CKPT" >&2
  exit 1
fi

echo "[hfdm_data1_150] exporting TorchScript"
python3 -m hfdm.cli.export_model \
  --config "$RUN_ROOT/config.yaml" \
  --checkpoint "$BEST_CKPT" \
  --out "$RUN_ROOT/export/fdm_ts.pt" \
  2>&1 | tee "$RUN_ROOT/logs/04_export.log"

if [[ "$START_TENSORBOARD" == "1" ]]; then
  if command -v tensorboard >/dev/null 2>&1; then
    nohup tensorboard --logdir "$RUN_ROOT" --host "$TB_HOST" --port "$TB_PORT" \
      > "$RUN_ROOT/logs/tensorboard.log" 2>&1 &
    echo "$!" > "$RUN_ROOT/tensorboard.pid"
    echo "[hfdm_data1_150] tensorboard=http://localhost:$TB_PORT pid=$(cat "$RUN_ROOT/tensorboard.pid")"
  else
    echo "[hfdm_data1_150] tensorboard command not found; skipping" >&2
  fi
fi

echo "[hfdm_data1_150] done"
echo "[hfdm_data1_150] run_root=$RUN_ROOT"
echo "[hfdm_data1_150] export=$RUN_ROOT/export/fdm_ts.pt"
