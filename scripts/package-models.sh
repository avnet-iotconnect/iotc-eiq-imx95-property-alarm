#!/usr/bin/env bash
# package-models.sh - the converted models as one tarball, for downloads.iotconnect.io.
#
#   bash scripts/package-models.sh     ->  iotc-property-alarm-models.tgz in the repo root
#
# Members are under models/, so it unpacks beside the sources into the layout main.py expects.
# Build them first: scripts/yolo-convert-neutron.sh and scripts/face-models-fetch.sh both write
# work/models. They are packaged separately from the sources because they change on a different
# clock -- a new BSP means new Neutron microcode and a new models tarball, with the code untouched.
#
# SmolVLM is not in here: it comes from Hugging Face on the board, via ./run.sh --prefetch.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

tar czf iotc-property-alarm-models.tgz -C work models

tar tzf iotc-property-alarm-models.tgz | sort
ls -la iotc-property-alarm-models.tgz
