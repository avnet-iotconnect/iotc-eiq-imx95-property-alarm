#!/usr/bin/env bash
# package-src.sh - the demo's sources as one tarball, for downloads.iotconnect.io.
#
#   bash scripts/package-src.sh        ->  iotc-property-alarm-src.tgz in the repo root
#
# Members are relative to koala/, so unpacking it IS the deployment: the user makes a directory on
# the board, unpacks there and runs ./install.sh. Rename it to *-v<x.y.z>.tgz before uploading.
#
# Excluded: everything the board builds or the owner supplies -- the two venvs, NXP's payload,
# the connector's downloaded source, the models (their own tarball), and anything holding a
# device's identity or a visitor's face.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

tar czf iotc-property-alarm-src.tgz -C koala \
    --exclude=venv --exclude=nxp-lib --exclude=models --exclude=__pycache__ \
    --exclude=connector/venv --exclude=connector/source --exclude=connector/source.tar.gz \
    --exclude=connector/server.log \
    --exclude=faces.json --exclude=state.json --exclude=capture.jpg --exclude=scene.jpg \
    --exclude=command.txt --exclude=iotcDeviceConfig.json --exclude='*.pem' \
    .

tar tzf iotc-property-alarm-src.tgz | sort
ls -la iotc-property-alarm-src.tgz
