#!/bin/bash
# Install NXP's eIQ AAF Connector into this directory. Run it ON the board.
#
#   scp -r <demo-dir> root@<board>:                  # this directory rides along
#   ssh root@<board> 'cd <demo-dir>/connector && ./install.sh'
#
# Kept out of the demo's own install.sh on purpose: this is gigabytes and minutes for
# one optional command ('ask'), and the demo runs without it.
#
# This directory is already laid out the way the connector expects, so copying it
# over is the whole deployment: config/server_config.json is read from here, and
# install.sh only adds source/ and venv/ beside it. Delete those two to rebuild.
#
# Needs internet: the source comes from GitHub and the Python dependencies
# (torch, transformers, ...) come from PyPI. Takes several minutes and the venv
# ends up around 1.4 GB.
#
# Prerequisite: the rt-sdk-ara2 2.0.4 .deb, which supplies the Ara runtime and the
# optimum-ara wheel this installs.

set -e

cd "$(dirname "$0")"

# v2.0.0 is the release whose pyproject.toml pins optimum_ara-2.0.0.2 -- the wheel
# the rt-sdk-ara2 2.0.4 .deb already dropped on this board, so nothing else needs
# downloading. The newer 2.1 line pins optimum_ara-1.0.0, which this BSP does not
# ship; to use it, set TAG=main and point WHEEL at an optimum_ara-1.0.1 wheel from
# the Optimum-Ara GitHub releases.
TAG=v2.0.0
WHEEL=/usr/share/python-wheels/optimum_ara-2.0.0.2-py3-none-any.whl

[ -f "$WHEEL" ] || { echo "missing $WHEEL -- install the rt-sdk-ara2 .deb first"; exit 1; }

echo "==> fetching connector $TAG"
curl -fsSL -o source.tar.gz \
    "https://codeload.github.com/nxp-imx-support/eiq-aaf-connector/tar.gz/refs/tags/$TAG"
rm -rf source && mkdir source
tar xzf source.tar.gz -C source --strip-components=1

echo "==> creating venv"
# Plain stdlib venv, and deliberately WITHOUT --system-site-packages: this venv
# brings its own torch, which must not mix with the BSP's tflite_runtime / gi / cv2.
python3 -m venv venv

echo "==> installing optimum-ara"
# Explicitly, and before the connector. The connector declares "optimum-ara" as a
# dependency but it is not on PyPI -- pyproject.toml locates it through a
# [tool.uv.sources] path that pip ignores. Installing it first means the
# requirement is already satisfied by the time the connector is installed.
venv/bin/pip install --disable-pip-version-check "$WHEEL"

echo "==> installing connector and dependencies (several minutes)"
venv/bin/pip install --disable-pip-version-check ./source

echo "==> patching for OpenAI tool-call conformance"
./apply-patch.sh

echo
echo "installed and patched in $(pwd)"
echo "next:  ./run.sh"
