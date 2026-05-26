#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./tools/train_hfdm_data1_60_h25_30hz.sh

Environment overrides:
  DATA_ROOT=results/data1
  HIGH_LEVEL_FDM_ROOT=/home/mexxiie/prj/high_level_fdm
  BASE_CONFIG=/home/mexxiie/prj/high_level_fdm/configs/experiments/geomapping_h25.yaml
  SOURCE_RAW_DIR=/home/mexxiie/prj/high_level_fdm/data/raw/geomapping_data1_h25
  OUTPUT_ROOT=results/hfdm_training
  RUN_ID=geomapping_data1_60_h25_30hz_<timestamp>
  RUN_ROOT=<OUTPUT_ROOT>/<RUN_ID>
  TRAIN_EPISODES=480
  VAL_EPISODES=120
  TARGET_DT=0.03333333333333333
  HORIZON=25
  HISTORY_LEN=10
  DEVICE=auto                         auto, cpu, or cuda
  EXPORT_CUDA_TRACE=auto              auto, 0, or 1
  START_TENSORBOARD=0                 set 1 to start TensorBoard after export
  TB_HOST=0.0.0.0
  TB_PORT=6008
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATA_ROOT="${DATA_ROOT:-results/data1}"
HIGH_LEVEL_FDM_ROOT="${HIGH_LEVEL_FDM_ROOT:-/home/mexxiie/prj/high_level_fdm}"
BASE_CONFIG="${BASE_CONFIG:-$HIGH_LEVEL_FDM_ROOT/configs/experiments/geomapping_h25.yaml}"
SOURCE_RAW_DIR="${SOURCE_RAW_DIR:-$HIGH_LEVEL_FDM_ROOT/data/raw/geomapping_data1_h25}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hfdm_training}"
RUN_ID="${RUN_ID:-geomapping_data1_60_h25_30hz_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/$OUTPUT_ROOT/$RUN_ID}"
TRAIN_EPISODES="${TRAIN_EPISODES:-480}"
VAL_EPISODES="${VAL_EPISODES:-120}"
TARGET_DT="${TARGET_DT:-0.03333333333333333}"
HORIZON="${HORIZON:-25}"
HISTORY_LEN="${HISTORY_LEN:-10}"
DEVICE="${DEVICE:-auto}"
EXPORT_CUDA_TRACE="${EXPORT_CUDA_TRACE:-auto}"
START_TENSORBOARD="${START_TENSORBOARD:-0}"
TB_HOST="${TB_HOST:-0.0.0.0}"
TB_PORT="${TB_PORT:-6008}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[hfdm_data1_60_30hz] error: DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$HIGH_LEVEL_FDM_ROOT/src" ]]; then
  echo "[hfdm_data1_60_30hz] error: high_level_fdm src not found: $HIGH_LEVEL_FDM_ROOT/src" >&2
  exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[hfdm_data1_60_30hz] error: BASE_CONFIG not found: $BASE_CONFIG" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_RAW_DIR" ]]; then
  echo "[hfdm_data1_60_30hz] error: SOURCE_RAW_DIR does not exist: $SOURCE_RAW_DIR" >&2
  exit 1
fi

export PYTHONPATH="$HIGH_LEVEL_FDM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcfg}"
mkdir -p "$MPLCONFIGDIR" "$RUN_ROOT"/{raw,dataset,run,logs,export}

echo "[hfdm_data1_60_30hz] data_root=$DATA_ROOT"
echo "[hfdm_data1_60_30hz] source_raw_dir=$SOURCE_RAW_DIR"
echo "[hfdm_data1_60_30hz] run_root=$RUN_ROOT"
echo "[hfdm_data1_60_30hz] target_dt=$TARGET_DT horizon=$HORIZON history_len=$HISTORY_LEN"

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
fi
echo "[hfdm_data1_60_30hz] device=$DEVICE"

cp "$BASE_CONFIG" "$RUN_ROOT/config.yaml"
python3 - "$RUN_ROOT/config.yaml" "$ROOT_DIR/$DATA_ROOT" "$RUN_ROOT" "$TRAIN_EPISODES" "$VAL_EPISODES" "$DEVICE" "$TARGET_DT" "$HORIZON" "$HISTORY_LEN" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
data_root = Path(sys.argv[2]).resolve()
run_root = Path(sys.argv[3]).resolve()
train_episodes = int(sys.argv[4])
val_episodes = int(sys.argv[5])
device = sys.argv[6]
target_dt = float(sys.argv[7])
horizon = int(sys.argv[8])
history_len = int(sys.argv[9])

cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
cfg["run_dir"] = str(run_root / "run")
cfg["data"]["geomapping_root"] = str(data_root)
cfg["data"]["raw_dir"] = str(run_root / "raw")
cfg["data"]["dataset_dir"] = str(run_root / "dataset")
cfg["data"]["num_train_episodes"] = train_episodes
cfg["data"]["num_val_episodes"] = val_episodes
cfg["data"]["dt"] = target_dt
cfg["data"]["history_len"] = history_len
cfg["data"]["horizon"] = horizon
cfg["data"].setdefault("tensorboard", {})
cfg["data"]["tensorboard"]["enabled"] = True
cfg["data"]["tensorboard"]["log_dir"] = str(run_root / "tensorboard" / "dataset")
cfg["geomapping"]["resample_to_dt"] = True
cfg["model"]["name"] = f"geomapping_single_channel_h{horizon}_30hz"
cfg["model"]["dt"] = target_dt
cfg["train"]["device"] = device
cfg["planner"]["dt"] = target_dt
cfg["planner"]["horizon"] = horizon
config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[hfdm_data1_60_30hz] config={config_path}", flush=True)
PY

echo "[hfdm_data1_60_30hz] converting old 60-seed episode set at 30Hz"
python3 - "$RUN_ROOT/config.yaml" "$SOURCE_RAW_DIR" "$RUN_ROOT/raw" "$TRAIN_EPISODES" "$VAL_EPISODES" <<'PY' 2>&1 | tee "$RUN_ROOT/logs/01_convert.log"
import json
import sys
from pathlib import Path

import numpy as np
from hfdm.adapters.geomapping import convert_geomapping_episode
from hfdm.common.config import load_config

config_path = Path(sys.argv[1])
source_raw_dir = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
expected = int(sys.argv[4]) + int(sys.argv[5])
cfg = load_config(config_path)
data_cfg = cfg["data"]
geomap = cfg["geomapping"]

raw_files = sorted(source_raw_dir.glob("episode_*.npz"))
if len(raw_files) < expected:
    raise SystemExit(f"need at least {expected} source raw episodes in {source_raw_dir}, found {len(raw_files)}")
out_dir.mkdir(parents=True, exist_ok=True)
summary = {
    "source_raw_dir": str(source_raw_dir),
    "output_dir": str(out_dir),
    "expected_episodes": expected,
    "episodes_converted": 0,
    "valid_samples": 0,
    "windows": 0,
    "target_dt": float(data_cfg["dt"]),
    "horizon": int(data_cfg["horizon"]),
    "history_len": int(data_cfg["history_len"]),
    "files": [],
}
for out_index, raw_path in enumerate(raw_files[:expected]):
    arr = np.load(raw_path, allow_pickle=False)
    metadata = json.loads(str(arr["metadata"]))
    episode_dir = Path(metadata["episode_dir"])
    episode = convert_geomapping_episode(
        episode_dir,
        map_size=int(geomap.get("map_size", cfg["model"].get("map_size", 64))),
        dt=float(data_cfg["dt"]),
        resample_to_dt=True,
        map_layer=str(geomap.get("map_layer", "reward_cost")),
        map_max_age_s=float(geomap.get("map_max_age_s", 0.75)),
        map_max_cost=float(geomap.get("map_max_cost", 100.0)),
        collision_threshold=float(geomap.get("collision_threshold", 0.95)),
        high_cost_threshold=float(geomap.get("high_cost_threshold", 0.65)),
        untraversable_threshold=float(geomap.get("untraversable_threshold", 0.85)),
    )
    samples = int(len(episode.states))
    windows = max(0, samples - int(data_cfg["history_len"]) - int(data_cfg["horizon"]))
    out_path = out_dir / f"episode_{out_index:06d}.npz"
    episode.save(out_path)
    summary["episodes_converted"] += 1
    summary["valid_samples"] += samples
    summary["windows"] += windows
    summary["files"].append(
        {
            "source_raw": str(raw_path),
            "source_episode": str(episode_dir),
            "output": str(out_path),
            "samples": samples,
            "windows": windows,
        }
    )
print(json.dumps(summary, indent=2), flush=True)
(out_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
PY

echo "[hfdm_data1_60_30hz] building H-FDM dataset"
python3 -m hfdm.cli.build_dataset \
  --config "$RUN_ROOT/config.yaml" \
  2>&1 | tee "$RUN_ROOT/logs/02_build_dataset.log"

echo "[hfdm_data1_60_30hz] training H-FDM H25 at 30Hz"
python3 -m hfdm.cli.train \
  --config "$RUN_ROOT/config.yaml" \
  2>&1 | tee "$RUN_ROOT/logs/03_train.log"

BEST_CKPT="$RUN_ROOT/run/checkpoints/best.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "[hfdm_data1_60_30hz] error: missing best checkpoint: $BEST_CKPT" >&2
  exit 1
fi

echo "[hfdm_data1_60_30hz] exporting CPU TorchScript"
python3 -m hfdm.cli.export_model \
  --config "$RUN_ROOT/config.yaml" \
  --checkpoint "$BEST_CKPT" \
  --out "$RUN_ROOT/export/fdm_ts.pt" \
  2>&1 | tee "$RUN_ROOT/logs/04_export.log"

if [[ "$EXPORT_CUDA_TRACE" == "auto" ]]; then
  EXPORT_CUDA_TRACE="$(python3 - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)"
fi

if [[ "$EXPORT_CUDA_TRACE" == "1" ]]; then
  echo "[hfdm_data1_60_30hz] exporting CUDA TorchScript trace"
  python3 - "$RUN_ROOT/config.yaml" "$BEST_CKPT" "$RUN_ROOT/export_cuda_trace" <<'PY' 2>&1 | tee "$RUN_ROOT/logs/05_export_cuda_trace.log"
import json
import shutil
import sys
from pathlib import Path

import torch

from hfdm.common.config import load_config
from hfdm.models.high_level_fdm import build_model

config_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
cfg = load_config(config_path)
device = torch.device("cuda")
ckpt = torch.load(checkpoint_path, map_location=device)
model_cfg = ckpt.get("model_cfg", cfg["model"])
model = build_model(model_cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
h = int(cfg["data"]["history_len"])
n = int(cfg["data"]["horizon"])
dummy_history = torch.zeros(1, h, model_cfg["state_dim"], device=device)
dummy_map = torch.zeros(1, model_cfg["map_channels"], model_cfg["map_size"], model_cfg["map_size"], device=device)
dummy_actions = torch.zeros(1, n, model_cfg["action_dim"], device=device)

class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, history, local_map, actions):
        out = self.m(history, local_map, actions)
        return out.pose, torch.sigmoid(out.risk_logits), out.applied_twist

out_dir.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    traced = torch.jit.trace(Wrapper(model), (dummy_history, dummy_map, dummy_actions), check_trace=False)
traced.save(str(out_dir / "fdm_ts.pt"))
shutil.copy2(Path(cfg["run_dir"]).parent / "export" / "fdm_metadata.json", out_dir / "fdm_metadata.json")
runtime = torch.jit.load(str(out_dir / "fdm_ts.pt"), map_location=device)
with torch.inference_mode(), torch.jit.optimized_execution(False):
    pose, risk, twist = runtime(dummy_history, dummy_map, dummy_actions)
    pose2, risk2, twist2 = runtime(dummy_history, dummy_map, dummy_actions)
meta = json.loads((out_dir / "fdm_metadata.json").read_text(encoding="utf-8"))
print(json.dumps({"out_dir": str(out_dir), "metadata": meta, "pose_shape": list(pose.shape)}, indent=2), flush=True)
PY
fi

if [[ "$START_TENSORBOARD" == "1" ]]; then
  if command -v tensorboard >/dev/null 2>&1; then
    nohup tensorboard --logdir "$RUN_ROOT" --host "$TB_HOST" --port "$TB_PORT" \
      > "$RUN_ROOT/logs/tensorboard.log" 2>&1 &
    echo "$!" > "$RUN_ROOT/tensorboard.pid"
    echo "[hfdm_data1_60_30hz] tensorboard=http://localhost:$TB_PORT pid=$(cat "$RUN_ROOT/tensorboard.pid")"
  else
    echo "[hfdm_data1_60_30hz] tensorboard command not found; skipping" >&2
  fi
fi

echo "[hfdm_data1_60_30hz] done"
echo "[hfdm_data1_60_30hz] run_root=$RUN_ROOT"
echo "[hfdm_data1_60_30hz] export=$RUN_ROOT/export/fdm_ts.pt"
if [[ -f "$RUN_ROOT/export_cuda_trace/fdm_ts.pt" ]]; then
  echo "[hfdm_data1_60_30hz] export_cuda_trace=$RUN_ROOT/export_cuda_trace/fdm_ts.pt"
fi
