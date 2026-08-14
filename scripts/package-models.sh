#!/usr/bin/env bash
# package-models.sh - build every model from scratch and package them for downloads.iotconnect.io.
#
#   bash scripts/package-models.sh     ->  iotc-property-alarm-models.tgz in the repo root
#
# Run it again whenever the Neutron SDK changes: a new SDK means new microcode, new models and a new
# tarball, with the code untouched. It is idempotent - it re-expands the SDK and re-converts.
#
# **Drop the SDK zip in the repo root first.** It is NXP-licensed and cannot be committed or hosted
# by us, so it is the one input you supply by hand. Everything else is fetched or built here.
#
# Why the SDK version is pinned in this script rather than discovered: converter, delegate and
# firmware have to be the same release. A model converted by one SDK and run against another's
# firmware fails *silently* - wrong numbers, no error (3.0.0's SFace embedded every face to the same
# vector, and a 3.0.0 delegate on 3.1.3 firmware detects nothing). Upgrading is deliberate: change
# the line below, re-run this, and reinstall the SDK on the board.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

SDK_ZIP="eiq-neutron-sdk-linux-3.1.3.zip"
SDK_DIR="work-refs/imx-eiq-neutron-sdk"
FIRMWARE="$SDK_DIR/target/imx95/imx95/NeutronFirmware.elf"

if [ ! -f "$SDK_ZIP" ]; then
  echo "Missing $SDK_ZIP in $(pwd)."
  echo "Download the eIQ Neutron SDK from nxp.com (login required) and drop the zip here."
  exit 1
fi

echo "== expanding $SDK_ZIP -> $SDK_DIR"
rm -rf "$SDK_DIR"
unzip -q "$SDK_ZIP" -d "$SDK_DIR"

# Only when actually missing, and into whatever python is active. YOLO's export needs ultralytics
# (and the torch it drags in); SFace's quantization needs onnx2tf + tensorflow, and onnx2tf shells
# out to the `onnxsim` command, so the venv's bin must be on PATH - run this from an activated venv.
# The Neutron conversion itself is a native binary and needs nothing. A system python will refuse to
# install (PEP 668) and say so.
echo "== python dependencies"
python3 -c "import ultralytics" 2>/dev/null || python3 -m pip install --quiet ultralytics
python3 -c "import onnx2tf, onnxsim, tensorflow" 2>/dev/null \
  || python3 -m pip install --quiet onnx2tf onnxsim tensorflow

# The face ONNX models come from the OpenCV zoo. Only fetched when absent: they never change, and a
# rebuild should not wait on the network.
if [ ! -f work/models/face_recognition_sface_2021dec.onnx ]; then
  bash scripts/face-models-fetch.sh
fi

# SFace: ONNX -> int8, calibrated on the public-domain face chips in files/sface-calib/. Then YOLO:
# export from ultralytics, quantize to int8, and convert both models for Neutron.
python3 scripts/sface-quantize.py
bash scripts/yolo-convert-neutron.sh

# What the board checks at startup. The firmware md5 is the exact handle: the delegate reports only
# a build hash, so the .elf itself is the only thing that identifies the release both sides can see.
echo "== version stamp"
{
  echo "sdk=${SDK_ZIP%.zip}"
  echo "firmware_md5=$(md5sum "$FIRMWARE" | cut -d' ' -f1)"
} > work/models/version.txt
cat work/models/version.txt

# Named one by one rather than `-C work models`, so that whatever else has accumulated in the
# scratch model directory stays out of a released tarball - and so a missing model fails here.
tar czf iotc-property-alarm-models.tgz -C work \
  models/version.txt \
  models/yolo11n_neutron.tflite models/yolo11n_int8.tflite \
  models/sface_neutron.tflite models/sface_int8.tflite \
  models/face_detection_yunet_2023mar.onnx models/face_recognition_sface_2021dec.onnx

tar tzf iotc-property-alarm-models.tgz | sort
ls -la iotc-property-alarm-models.tgz
