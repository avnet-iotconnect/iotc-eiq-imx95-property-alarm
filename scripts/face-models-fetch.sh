#!/usr/bin/env bash
# face-models-fetch.sh - download YuNet + SFace ONNX (the elephant pilot's face models) to work/models.
# Host-side only. These are the exact OpenCV Zoo releases cv2's FaceDetectorYN / FaceRecognizerSF expect.
# Run: bash scripts/face-models-fetch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
dest=work/models
mkdir -p "$dest"

# opencv_zoo stores models in Git LFS; media.githubusercontent.com/media/ serves the real binary
# (raw.githubusercontent.com would hand back a tiny LFS pointer file instead).
base=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
declare -A models=(
  [face_detection_yunet_2023mar.onnx]="$base/face_detection_yunet/face_detection_yunet_2023mar.onnx"
  [face_recognition_sface_2021dec.onnx]="$base/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

for name in "${!models[@]}"; do
  echo "Fetching $name ..."
  curl -fSL "${models[$name]}" -o "$dest/$name"
  # An LFS pointer is only ~130 bytes; a real model is >>100 KB. Catch the pointer-file case early.
  size=$(stat -c%s "$dest/$name")
  if [ "$size" -lt 100000 ]; then
    echo "ERROR: $name is only ${size} bytes - likely a Git LFS pointer, not the model. Check the URL." >&2
    exit 1
  fi
  echo "  -> $dest/$name (${size} bytes)"
done
echo "Done. Now: bash elephant/deploy.sh"
