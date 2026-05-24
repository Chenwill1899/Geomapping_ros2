#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT="${OUTPUT:-results/data1}"
BACKUP_ROOT="${BACKUP_ROOT:-$OUTPUT/_continue_backups}"
BACKUP_DIR="$BACKUP_ROOT/$(date +%Y%m%d_%H%M%S)"

CONTINUE_SEEDS=(
  133263447 825926137 327371665 213002999 576437519
  489763602 910013409 517077298 764355285 224702916
  427435773 424437574 580300667 300907444 605431403
  205882820 140952379 92688820 141768120 947456082
  334978139 708309389 366383943 342932393 253211777
  812905771 656554183 651459536 961891663 676047393
  896948021 811525376 261448318 998675441 265981134
  478082932 270563162 82330522 115160682 711506838
  219270505 152985983 966109235 464986499 605861972
  725240773 457807992 610409332 569619802 817781102
  838034978 752958241 140554670 936727090 885424762
  32234425 778643493 488454068 125742766 313174096
  341847103 256407863 171718556 634311768 813462112
  125467773 102947535 840794063 595793734 377231442
  900076509 868480653 992666241 65930618 464121780
  275237809 692046781 233326764 66913231 543664951
  783485144 404923831 411955235 150988920 264701936
  985491522 789507480 482639684 368079239 503075426
)

SEED_CSV="$(IFS=,; echo "${CONTINUE_SEEDS[*]}")"

mkdir -p "$BACKUP_DIR"
for file in manifest.jsonl summary.json config.yaml; do
  if [[ -f "$OUTPUT/$file" ]]; then
    cp "$OUTPUT/$file" "$BACKUP_DIR/$file.before"
  fi
done

echo "[data1_continue] output=$OUTPUT"
echo "[data1_continue] backup_dir=$BACKUP_DIR"
echo "[data1_continue] seed_count=${#CONTINUE_SEEDS[@]}"

set +e
OUTPUT="$OUTPUT" ./tools/run_data1_100seeds.sh --seeds "$SEED_CSV" "$@"
status=$?
set -e

if [[ -f "$BACKUP_DIR/manifest.jsonl.before" && -f "$OUTPUT/manifest.jsonl" ]]; then
  cp "$OUTPUT/manifest.jsonl" "$BACKUP_DIR/manifest.jsonl.new"
  python3 - "$OUTPUT" "$BACKUP_DIR/manifest.jsonl.before" "$BACKUP_DIR/manifest.jsonl.new" <<'PY'
import json
import sys
from collections import OrderedDict
from pathlib import Path

root = Path(sys.argv[1])
before_manifest = Path(sys.argv[2])
new_manifest = Path(sys.argv[3])
manifest = root / "manifest.jsonl"
entries_by_key = OrderedDict()
for path in (before_manifest, new_manifest):
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        goal = entry.get("goal", {})
        key = (
            int(entry["seed"]),
            int(entry["episode_index"]),
            str(goal.get("id", "")),
        )
        if key in entries_by_key:
            del entries_by_key[key]
        entries_by_key[key] = entry
entries = list(entries_by_key.values())
manifest.write_text(
    "".join(json.dumps(entry, sort_keys=False) + "\n" for entry in entries),
    encoding="utf-8",
)
seed_order = []
seen_seeds = set()
goal_order = []
seen_goals = set()
for entry in entries:
    seed = int(entry["seed"])
    if seed not in seen_seeds:
        seen_seeds.add(seed)
        seed_order.append(seed)
    goal_key = json.dumps(entry.get("goal", {}), sort_keys=True)
    if goal_key not in seen_goals:
        seen_goals.add(goal_key)
        goal_order.append(entry.get("goal", {}))

summary = {
    "output_dir": str(root.resolve()),
    "seeds": seed_order,
    "goals": goal_order,
    "episodes": len(entries),
    "success": sum(1 for entry in entries if entry.get("status") == "success"),
    "failed": sum(1 for entry in entries if entry.get("status") != "success"),
    "seeds_completed": len(seed_order),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(
    "[data1_continue] merged_manifest "
    f"episodes={summary['episodes']} seeds={summary['seeds_completed']} "
    f"success={summary['success']} failed={summary['failed']}",
    flush=True,
)
PY
fi

exit "$status"
