#!/usr/bin/env bash
# run.sh - launch the elephant pilot ON THE BOARD.
#   NPU path : ./run.sh --model yolo11n_neutron.tflite --delegate
#   CPU path : ./run.sh --model yolo11n_int8.tflite
# YOLO uses the NPU with --delegate; the face models (YuNet + SFace) always run on the CPU.
# Uses the project venv at ~/elephant/venv by default (see requirements.txt - no pip installs needed).
# Override with PYTHON=/path/to/python ./run.sh ...
# Register a user while it runs:  echo 'register user Joe' > command.txt
set -euo pipefail
cd "$(dirname "$0")"

# Wayland env the preview (waylandsink) needs when we launch over SSH.
export XDG_RUNTIME_DIR=/run/user/0
export WAYLAND_DISPLAY=wayland-0

# Pin the A55 cores to max clock so timings are stable (reverts to 'ondemand' on reboot). See dolphin.
echo "CPU governor before: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
for governor_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance > "$governor_file" 2>/dev/null || true
done
echo "CPU governor now   : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) @ \
$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz"

"${PYTHON:-./venv/bin/python}" main.py "$@"
