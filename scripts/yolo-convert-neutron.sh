#!/usr/bin/env bash
# yolo-convert-neutron.sh - export YOLO and convert both vision models for Neutron.
#
# Normally invoked by scripts/package-models.sh, which expands the SDK and installs the python
# dependencies first. Runnable on its own once those two things are true.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

SDK="work-refs/imx-eiq-neutron-sdk"

python3 -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

# 1. Export via scripts/yolo-export-prep.py, NOT plain `yolo export`. Two reasons, both in that file:
#    - format=tflite is redirected to litert-torch (ultralytics 8.4.83+), whose graph neutron rejects;
#      so it exports via the onnx2tf format=saved_model path instead.
#    - it patches the Detect head to emit RAW pre-decode tensors, because YOLOv8's box decode
#      (DFL + anchors) collapses to zero under int8; dolphin/yolo.py does that decode in float on CPU.
#    Produces yolo11n_saved_model/yolo11n_full_integer_quant.tflite (output shape 1x144x2100).
python3 scripts/yolo-export-prep.py

# 2. Convert for Neutron. Watch the conversion ratio: 0 means nothing delegated, and a *low* ratio
#    is worth reading too - 3.0.0 managed 303/475 in nine graphs where 3.1.3 does 351/385 in two,
#    which is 12.9 ms of inference against 5.9 ms. The 3.0.0 SDK shipped or-tools/abseil/protobuf as
#    shared objects with no RPATH and needed LD_LIBRARY_PATH; 3.1.x links them statically.
mkdir -p work/models
mv yolo11n_saved_model/yolo11n_full_integer_quant.tflite work/models/yolo11n_int8.tflite
"$SDK/bin/neutron-converter" \
  --input work/models/yolo11n_int8.tflite \
  --output work/models/yolo11n_neutron.tflite \
  --target imx95

# The SFace embedder takes the same step, from an int8 file that was quantized by hand and is an
# input here rather than a product - that step is not yet scripted.
"$SDK/bin/neutron-converter" \
  --input work/models/sface_int8.tflite \
  --output work/models/sface_neutron.tflite \
  --target imx95

# 3. Keep BOTH: yolo11n_int8.tflite (CPU path) and yolo11n_neutron.tflite (NPU path).
ls -la work/models
