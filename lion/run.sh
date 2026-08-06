#!/usr/bin/env bash
# run.sh - launch the demo ON THE BOARD.
#   ./run.sh                                     # NPU (default): YOLO + SFace on Neutron, voice on
#   ./run.sh --no-voice                          # video only; no eIQ payload, no venv packages
#   ./run.sh --model models/yolo11n_int8.tflite  # full-CPU baseline
#   ./run.sh --no-ask                            # without the LLM on the Ara-240
#   ./run.sh --text-commands command.txt         # debug: also take commands from a file
# The backend follows the model name: a '..._neutron.tflite' model auto-uses the delegate.
# YuNet face detection and the whole voice stack always run on the CPU.
# Uses the venv at ./venv (created by install.sh). The LLM is a separate process with a venv
# of its own -- see connector/run.sh -- and this script neither starts nor waits for it.
# Override with PYTHON=/path/to/python ./run.sh ...
set -euo pipefail
cd "$(dirname "$0")"

# Wayland env the preview (waylandsink) needs when we launch over SSH.
export XDG_RUNTIME_DIR=/run/user/0
export WAYLAND_DISPLAY=wayland-0

# espeak-ng lives under /opt/dm-eiq rather than /usr, so the loader and the phonetiser both have to
# be told where to look. ESPEAK_DATA_PATH points at the directory *containing* espeak-ng-data.
export LD_LIBRARY_PATH="/opt/dm-eiq/lib:${LD_LIBRARY_PATH:-}"
export ESPEAK_DATA_PATH="/opt/dm-eiq/share"
export PATH="/opt/dm-eiq/bin:$PATH"

# torch and onnxruntime will otherwise spawn a thread per core and starve the 30 fps video loop.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Redirected to a log file, Python would buffer our prints in 8 KB blocks while the delegates write
# straight to stderr -- so a crash looks like it happened before anything ran. Unbuffer it.
export PYTHONUNBUFFERED=1

# Pin the A55 cores to max clock so timings are stable (reverts to 'ondemand' on reboot).
echo "CPU governor before: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
for governor_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance > "$governor_file" 2>/dev/null || true
done
echo "CPU governor now   : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) @ \
$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz"

"${PYTHON:-./venv/bin/python}" main.py "$@"
