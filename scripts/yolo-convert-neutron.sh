#!/usr/bin/env bash
# yolo-convert-neutron.sh - host-side model prep for the dolphin YOLO/Neutron demo.
# Read it, then run the lines you need. Run from the repo root: bash scripts/yolo-convert-neutron.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# 1. Need a python3 with a working _ctypes (torch imports it). The pyenv 3.13.12
#    behind ./.venv lacks it. Fix once: dnf install libffi-devel && pyenv install --force 3.13.12
python3 -c "import ctypes; print('ctypes OK')"

# 2. Fresh venv + ultralytics (heavy: pulls torch).
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ultralytics
.venv/bin/python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

# 3. Export via scripts/yolo-export-prep.py, NOT plain `yolo export`. Two reasons, both in that file:
#    - format=tflite is redirected to litert-torch (ultralytics 8.4.83+), whose graph neutron rejects;
#      so it exports via the onnx2tf format=saved_model path instead.
#    - it patches the Detect head to emit RAW pre-decode tensors, because YOLOv8's box decode
#      (DFL + anchors) collapses to zero under int8; dolphin/yolo.py does that decode in float on CPU.
#    Produces yolo11n_saved_model/yolo11n_full_integer_quant.tflite (output shape 1x144x2100).
.venv/bin/python scripts/yolo-export-prep.py

# 4. Convert for Neutron. Watch "Number of Neutron operators = N" (0 = nothing delegated).
#    The converter version must match the board's Neutron firmware exactly -- see README.md. The 3.0.0
#    SDK ships its or-tools/abseil/protobuf as shared objects in lib/ and sets no RPATH, so the
#    converter will not start without LD_LIBRARY_PATH (3.1.x linked them statically and did not need it).
mkdir -p work/models
mv yolo11n_saved_model/yolo11n_full_integer_quant.tflite work/models/yolo11n_int8.tflite
export LD_LIBRARY_PATH="$PWD/../imx-eiq-neutron-sdk/lib"
../imx-eiq-neutron-sdk/bin/neutron-converter \
  --input work/models/yolo11n_int8.tflite \
  --output work/models/yolo11n_neutron.tflite \
  --target imx95

# The SFace embedder takes the same step, from the int8 file the elephant pilot built by hand.
../imx-eiq-neutron-sdk/bin/neutron-converter \
  --input work/models/sface_int8.tflite \
  --output work/models/sface_neutron.tflite \
  --target imx95

# 5. Keep BOTH: yolo11n_int8.tflite (CPU path) and yolo11n_neutron.tflite (NPU path).
ls -la work/models
