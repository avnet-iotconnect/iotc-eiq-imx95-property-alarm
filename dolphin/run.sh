#!/usr/bin/env bash
# run.sh - launch the demo ON THE BOARD.
#   CPU path : ./run.sh --model yolov8n_int8.tflite
#   NPU path : ./run.sh --model yolov8n_neutron.tflite --delegate
# Uses the project venv at ~/dolphin/venv by default (see requirements.txt for its deps).
# Override with PYTHON=/path/to/python ./run.sh ...
set -euo pipefail
cd "$(dirname "$0")"

# Wayland env the preview (waylandsink) needs when we launch over SSH.
export XDG_RUNTIME_DIR=/run/user/0
export WAYLAND_DISPLAY=wayland-0

# Pin the A55 cores to max clock so timings are stable. The default 'ondemand' governor
# ping-pongs frequency and makes inference bounce between ~5 ms and ~50 ms. This is a runtime
# sysfs change only - it reverts to 'ondemand' on the next reboot. To undo now without rebooting:
#   echo ondemand | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo "CPU governor before: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
for governor_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance > "$governor_file" 2>/dev/null || true
done
echo "CPU governor now   : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) @ \
$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz"

"${PYTHON:-./venv/bin/python}" detect.py "$@"
