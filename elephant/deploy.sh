#!/usr/bin/env bash
# deploy.sh - push elephant code + models from the host to the board's ~/elephant.
# Run from the host: bash elephant/deploy.sh
# Assumes the face ONNX models are in ../work/models (fetch them with scripts/face-models-fetch.sh).
set -euo pipefail
cd "$(dirname "$0")"
BOARD=root@192.168.38.203

ssh "$BOARD" 'mkdir -p elephant'
scp *.py *.sh *.md *.txt "$BOARD":elephant/    # the whole pilot, then the models below
scp ../work/models/yolo11n_int8.tflite ../work/models/yolo11n_neutron.tflite \
    ../work/models/face_detection_yunet_2023mar.onnx \
    ../work/models/face_recognition_sface_2021dec.onnx "$BOARD":elephant/
echo "Deployed. On the board:"
echo "  cd ~/elephant && python3 diag-face.py face_detection_yunet_2023mar.onnx face_recognition_sface_2021dec.onnx"
echo "  ./run.sh --model yolo11n_neutron.tflite --delegate"
