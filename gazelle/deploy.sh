#!/usr/bin/env bash
# deploy.sh - push gazelle from the host to the board's ~/gazelle.
# Run from the host:  bash gazelle/deploy.sh [path/to/dm-eiq-<tag>.tgz]
#
# Three kinds of payload, moving at three different rates:
#   our code       a few kilobytes, changes constantly            -> every time
#   vision models  ~25 MB, changes when a model is reconverted    -> every time (cheap enough)
#   the eIQ blob   139 MB, changes when eIQ does, ie. never       -> only when a tarball is named
#
# The vision models are built on the host (neutron-converter does not run on the board); point
# MODELS_DIR at wherever yours live. BOARD overrides the target.
set -euo pipefail
cd "$(dirname "$0")"
BOARD="${BOARD:-root@192.168.38.203}"
MODELS_DIR="${MODELS_DIR:-../work/models}"

ssh "$BOARD" 'mkdir -p gazelle/nxp-lib'
scp *.py *.sh *.json *.txt *.md "$BOARD":gazelle/
ssh "$BOARD" 'chmod +x gazelle/*.sh gazelle/main.py gazelle/mic-check.py gazelle/diag-face.py'

scp "$MODELS_DIR"/yolo11n_int8.tflite "$MODELS_DIR"/yolo11n_neutron.tflite \
    "$MODELS_DIR"/face_detection_yunet_2023mar.onnx \
    "$MODELS_DIR"/face_recognition_sface_2021dec.onnx \
    "$MODELS_DIR"/sface_int8.tflite "$MODELS_DIR"/sface_neutron.tflite "$BOARD":gazelle/

if [ $# -ge 1 ]; then
    echo "Pushing the eIQ payload $1 (139 MB, a minute or two) ..."
    scp "$1" "$BOARD":/tmp/dm-eiq-payload.tgz
    ssh "$BOARD" 'tar xzf /tmp/dm-eiq-payload.tgz -C gazelle/nxp-lib/ && rm /tmp/dm-eiq-payload.tgz'
fi

echo "Deployed. On the board (cd ~/gazelle):"
echo "  ./install.sh          # once: payload, espeak-ng into /opt/dm-eiq, then the venv"
echo "  ./run.sh --prefetch   # once: pull SmolVLM while the network is good"
echo "  ./run.sh              # the demo: say \"Hey NXP\", then a command"
