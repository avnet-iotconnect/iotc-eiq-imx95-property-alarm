#!/usr/bin/env bash
# run.sh - launch the demo ON THE BOARD, with the Wayland env the preview needs.
#   CPU path : ./run.sh --model yolov8n_int8.tflite
#   NPU path : ./run.sh --model yolov8n_neutron.tflite --delegate
# Override the interpreter with PYTHON=... ./run.sh ... (defaults to system python3,
# which is where gi/tflite_runtime/cv2 live on this BSP).
set -euo pipefail
cd "$(dirname "$0")"
export XDG_RUNTIME_DIR=/run/user/0
export WAYLAND_DISPLAY=wayland-0
"${PYTHON:-python3}" detect.py "$@"
