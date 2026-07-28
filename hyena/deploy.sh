#!/usr/bin/env bash
# deploy.sh - push hyena from the host to the board's ~/hyena.
# Run from the host:  bash hyena/deploy.sh [path/to/dm-eiq-<tag>.tgz]
#
# Four kinds of payload, moving at four different rates:
#   our code       a few kilobytes, changes constantly            -> every time
#   vision models  ~25 MB, changes when a model is reconverted    -> every time (cheap enough)
#   the eIQ blob   139 MB, changes when eIQ does, ie. never       -> only when a tarball is named
#   /IOTCONNECT    device config + certificate + key, per device  -> whenever they are here
#
# The /IOTCONNECT credentials are yours, are never committed (see .gitignore) and are not created
# by anything here: download iotcDeviceConfig.json and the certificate pair from your device's info
# panel into this directory, and they go along with the code. Without them the demo still runs; it
# just says "cloud: unconfigured" on the HUD.
#
# The vision models are built on the host (neutron-converter does not run on the board); point
# MODELS_DIR at wherever yours live. BOARD overrides the target.
set -euo pipefail
cd "$(dirname "$0")"
BOARD="${BOARD:-root@192.168.38.203}"
MODELS_DIR="${MODELS_DIR:-../work/models}"

ssh "$BOARD" 'mkdir -p hyena/nxp-lib'
scp *.py *.sh *.json *.txt *.md "$BOARD":hyena/
ssh "$BOARD" 'chmod +x hyena/*.sh hyena/main.py hyena/mic-check.py hyena/diag-face.py hyena/iotc-check.py'

# The private key goes over with restrictive permissions and stays that way on the board.
if compgen -G "*-key.pem" > /dev/null || compgen -G "device-pkey.pem" > /dev/null; then
    scp *-crt.pem *-key.pem device-*.pem "$BOARD":hyena/ 2>/dev/null || true
    ssh "$BOARD" 'chmod 600 hyena/*-key.pem hyena/device-pkey.pem 2>/dev/null || true'
    echo "Pushed the /IOTCONNECT certificate pair."
else
    echo "No /IOTCONNECT certificates here -- the board will report 'cloud: unconfigured'."
fi

scp "$MODELS_DIR"/yolo11n_int8.tflite "$MODELS_DIR"/yolo11n_neutron.tflite \
    "$MODELS_DIR"/face_detection_yunet_2023mar.onnx \
    "$MODELS_DIR"/face_recognition_sface_2021dec.onnx \
    "$MODELS_DIR"/sface_int8.tflite "$MODELS_DIR"/sface_neutron.tflite "$BOARD":hyena/

if [ $# -ge 1 ]; then
    echo "Pushing the eIQ payload $1 (139 MB, a minute or two) ..."
    scp "$1" "$BOARD":/tmp/dm-eiq-payload.tgz
    ssh "$BOARD" 'tar xzf /tmp/dm-eiq-payload.tgz -C hyena/nxp-lib/ && rm /tmp/dm-eiq-payload.tgz'
fi

echo "Deployed. On the board (cd ~/hyena):"
echo "  ./install.sh          # once: payload, espeak-ng into /opt/dm-eiq, then the venv"
echo "  ./run.sh --prefetch   # once: pull SmolVLM while the network is good"
echo "  ./iotc-check.py       # once: does this device reach /IOTCONNECT, S3 and KVS?"
echo "  ./run.sh              # the demo: say \"Hey NXP\", or send a command from the dashboard"
