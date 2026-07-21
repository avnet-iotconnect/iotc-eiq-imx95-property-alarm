#!/usr/bin/env bash
# deploy.sh - push demo code + models from the host to the board's ~/dolphin.
# Run from the host: bash dolphin/deploy.sh
set -euo pipefail
cd "$(dirname "$0")"
scp camera.py yolo.py detect.py run.sh root@192.168.38.203:dolphin/
scp ../work/models/yolov8n_int8.tflite ../work/models/yolov8n_neutron.tflite root@192.168.38.203:dolphin/
echo "Deployed. On the board:  cd ~/dolphin && ./run.sh --model yolov8n_int8.tflite"
