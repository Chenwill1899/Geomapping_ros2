#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

set +u
source /opt/ros/humble/setup.bash
AUSIM_ROS_WS_SETUP="${AUSIM_ROS_WS_SETUP:-/home/mexxiie/prj/ausim2/build/ros_ws/install/setup.bash}"
if [[ -f "$AUSIM_ROS_WS_SETUP" ]]; then
  source "$AUSIM_ROS_WS_SETUP"
else
  echo "[data1] warning: ausim_msg overlay not found: $AUSIM_ROS_WS_SETUP" >&2
fi
source "$ROOT_DIR/install/setup.bash"
set -u

if ! python3 -c 'from ausim_msg.msg import BoundingBox3DArray' >/dev/null 2>&1; then
  echo "[data1] error: cannot import ausim_msg.msg.BoundingBox3DArray; source ausim2/build/ros_ws before collecting data1" >&2
  exit 1
fi

export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_logs}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcfg}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
mkdir -p "$ROS_LOG_DIR" "$MPLCONFIGDIR"

cleanup_geomapping_runtime() {
  pkill -f "ros2 launch ausim_geomapping_adapter ausim_scout_mppi_frontend.launch.py" 2>/dev/null || true
  pkill -f "/home/mexxiie/prj/ausim2/build/bin/ausim_ros_bridge" 2>/dev/null || true
  pkill -f "/home/mexxiie/prj/ausim2/build/bin/scout" 2>/dev/null || true
  pkill -f "fdm_mppi mujoco-closed-loop" 2>/dev/null || true
  pkill -f "/home/mexxiie/prj/ausim2/em_run.sh" 2>/dev/null || true
}

cleanup_geomapping_runtime
trap cleanup_geomapping_runtime EXIT

GOALS="${GOALS:-results/nav_dataset/_smoke_inputs/seed17_random10_obstacle_10m.yaml}"
OUTPUT="${OUTPUT:-results/data1}"
PROFILE="${PROFILE:-src/mppi_controller/configs/mujoco_rviz_goal.yaml}"
CONTROLLER="${CONTROLLER:-nominal_cuda}"
OBSTACLE_CONFIG="${OBSTACLE_CONFIG:-$ROOT_DIR/src/mppi_controller/configs/obstacle_scout_sparse.yaml}"
NO_PROGRESS_WINDOW_S="${NO_PROGRESS_WINDOW_S:-45.0}"
TLTRAJECTORY_TIMEOUT_S="${TLTRAJECTORY_TIMEOUT_S:-0.0}"
ROS_DOMAIN_ID_ARG="${ROS_DOMAIN_ID_ARG:-143}"

SEEDS=(
  213638760 852310503 994638366 182004044 820825164
  995890435 255752412 178486804 312954753 256957635
  287631022 812865709 258995197 878704925 839686098
  475998854 920376905 15482891 205916005 922671903
  746814397 37877819 740827879 224655900 665308730
  734371636 735667118 741878813 741435905 985942154
  458587463 161818044 683311562 295164216 524996852
  224154740 137686771 708608355 44132355 425313478
  449907290 257111230 192439649 67778890 171934173
  322362058 565657843 346536475 591750551 909166258
  713119653 300084769 660003925 534762471 308207327
  200792925 529471185 671255798 326811264 612332172
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

SEED_CSV="$(IFS=,; echo "${SEEDS[*]}")"

python3 tools/geomapping_dataset_collect.py \
  --goals "$GOALS" \
  --seeds "$SEED_CSV" \
  --output "$OUTPUT" \
  --profile "$PROFILE" \
  --controller "$CONTROLLER" \
  --obstacle-config "$OBSTACLE_CONFIG" \
  --continue-on-failure \
  --no-progress-window-s "$NO_PROGRESS_WINDOW_S" \
  --tltrajectory-timeout-s "$TLTRAJECTORY_TIMEOUT_S" \
  --ros-domain-id "$ROS_DOMAIN_ID_ARG" \
  "$@"
