#!/usr/bin/env bash
# deploy.sh - push iguana from the host to the board's ~/iguana.
# Run from the host:  bash iguana/deploy.sh
#
# Three kinds of payload, moving at three different rates:
#   our code       a few kilobytes, changes constantly            -> every time
#   vision models  ~25 MB, changes when a model is reconverted    -> every time (cheap enough)
#   /IOTCONNECT    device config + certificate + key, per device  -> whenever they are here
#
# The 139 MB eIQ blob is deliberately NOT here: install.sh downloads it on the board. Pushing a local
# copy would skip that download, which is the part of a clean install most worth exercising.
#
# The /IOTCONNECT credentials are yours, are never committed (see .gitignore) and are not created
# by anything here. Download them from your device's info panel into this directory, and rename the
# certificate pair to the two fixed names iguana expects:
#
#     iotcDeviceConfig.json     as downloaded
#     device-cert.pem           <duid>-crt.pem, renamed
#     device-key.pem            <duid>-key.pem, renamed
#
# Fixed names rather than hyena's <duid>-crt.pem, so that nothing in the deployment has to read the
# config to work out which files to copy. Without them the demo still runs; it just says
# "cloud: unconfigured" on the HUD, and with no cloud there is no channel ARN and so no streaming.
#
# The vision models are built on the host (neutron-converter does not run on the board); point
# MODELS_DIR at wherever yours live. BOARD overrides the target.
set -euo pipefail
cd "$(dirname "$0")"
BOARD="${BOARD:-root@192.168.38.203}"
MODELS_DIR="${MODELS_DIR:-../work/models}"

ssh "$BOARD" 'mkdir -p iguana'
scp *.py *.sh *.json *.txt *.md "$BOARD":iguana/
ssh "$BOARD" 'chmod +x iguana/*.sh iguana/main.py iguana/mic-check.py iguana/diag-face.py iguana/iotc-check.py'

# The private key goes over with restrictive permissions and stays that way on the board.
if [ -f device-cert.pem ] && [ -f device-key.pem ]; then
    scp device-cert.pem device-key.pem "$BOARD":iguana/
    ssh "$BOARD" 'chmod 600 iguana/device-key.pem'
    echo "Pushed the /IOTCONNECT certificate pair."
else
    echo "No device-cert.pem/device-key.pem here -- the board will report 'cloud: unconfigured'."
fi

scp "$MODELS_DIR"/yolo11n_int8.tflite "$MODELS_DIR"/yolo11n_neutron.tflite \
    "$MODELS_DIR"/face_detection_yunet_2023mar.onnx \
    "$MODELS_DIR"/face_recognition_sface_2021dec.onnx \
    "$MODELS_DIR"/sface_int8.tflite "$MODELS_DIR"/sface_neutron.tflite "$BOARD":iguana/

echo "Deployed. On the board (cd ~/iguana):"
echo "  ./install.sh          # once: payload, espeak-ng into /opt/dm-eiq, then the venv"
echo "  ./run.sh --prefetch   # once: pull SmolVLM while the network is good"
echo "  ./iotc-check.py       # once: does this device reach /IOTCONNECT, S3 and KVS?"
echo "  ./run.sh              # the demo: say \"Hey NXP\", or send a command from the dashboard"
echo "  ...then press play on the device's dashboard to watch the display over WebRTC."
