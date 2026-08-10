#!/usr/bin/env bash
# package-src.sh - the demo's sources as one tarball, for downloads.iotconnect.io.
#
#   bash scripts/package-src.sh        ->  iotc-property-alarm-src.tgz in the repo root
#
# Run it from the pilot directory (lion/). Members are relative to that, so unpacking the tarball IS
# the deployment: the user makes a directory on the board, unpacks there and runs ./install.sh.
# Rename it to *-v<x.y.z>.tgz before uploading.
#
# Excluded: everything the board builds or the owner supplies -- the two venvs, NXP's payload,
# the connector's downloaded source, the models (their own tarball), and anything holding a
# device's identity or a visitor's face.
set -euo pipefail

if [ ! -f main.py ]; then
    echo "Run this script from the the source root!"
    exit 1
fi

tar czf ../iotc-property-alarm-src.tgz \
  --exclude=venv --exclude=nxp-lib --exclude=models --exclude=agenttools  --exclude=__pycache__ \
  --exclude=iotcDeviceConfig.json --exclude='*.pem' \
  .

tar tzf ../iotc-property-alarm-src.tgz | sort
ls -la ../iotc-property-alarm-src.tgz
