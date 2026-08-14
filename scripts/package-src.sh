#!/usr/bin/env bash
# package-src.sh - the demo's sources as one tarball, for downloads.iotconnect.io.
#
#   bash scripts/package-src.sh        ->  iotc-property-alarm-src.tgz in the repo root
#
# Rename it to *-v<x.y.z>.tgz before uploading.
#
# Excluded: everything the board builds or the owner supplies -- the two venvs, NXP's payload,
# the connector's downloaded source, the models (their own tarball), and anything holding a
# device's identity or a visitor's face.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/../src/"

tar czf ../iotc-property-alarm-src.tgz \
  --exclude=venv --exclude=nxp-lib --exclude=models --exclude=agenttools  --exclude=__pycache__ \
  --exclude=iotcDeviceConfig.json --exclude='*.pem' \
  .

tar tzf ../iotc-property-alarm-src.tgz | sort
ls -la ../iotc-property-alarm-src.tgz
