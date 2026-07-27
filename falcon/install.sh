#!/bin/bash
#
# Prepare a board to run falcon. Replaces NXP's install.sh, which does considerably more than we
# want: it enables NTP, installs Python packages system-wide with sudo, appends a venv activation
# block to ~/.bashrc, and -- when ALSA headers are missing, as they are on our image -- builds all of
# alsa-lib into /usr/local, leaving a second libasound.so.2 that audio programs can silently bind to.
#
# We do four things and nothing else:
#   1. download and unpack NXP's payload into nxp-lib/
#   2. build espeak-ng into /opt/dm-eiq, the only native prerequisite TTS has
#   3. create a --system-site-packages venv, so the BSP's numpy, onnxruntime, tflite_runtime and gi
#      stay visible alongside what pip adds
#   4. pip install requirements.txt
#
# Everything it writes lives in nxp-lib/, /opt/dm-eiq and the venv. Nothing lands in /usr, so
# `rm -rf` of those paths returns the image to its shipped state -- compare against a
# `find /opt /usr | sort` taken beforehand to confirm.
#
# Usage:  ./install.sh [--skip-payload] [--skip-espeak] [--skip-pip]

set -euo pipefail

script=$(readlink -f "$0")
cd "$(dirname "$script")"

skip_payload=false
skip_espeak=false
skip_pip=false
for argument in "$@"; do
    case "$argument" in
        --skip-payload) skip_payload=true ;;
        --skip-espeak)  skip_espeak=true ;;
        --skip-pip)     skip_pip=true ;;
        *) echo "Unknown option: $argument"; sed -n '3,21p' "$script" | sed 's/^# \?//'; exit 1 ;;
    esac
done

# --- NXP's payload ---------------------------------------------------------------------------
# 129 MB of compiled modules and encrypted weights, hosted rather than committed: the licence does
# not permit redistributing it from a public repo. Built by scripts/dm-eiq-package.sh.

if [ "$skip_payload" = false ] && [ ! -d nxp-lib/src/vit ]; then
    echo "=== downloading the eIQ payload (129 MB) ==="
    wget -O /tmp/dm-eiq-payload.tgz \
        https://downloads.iotconnect.io/partners/nxp/packages/dm-eiq-genai-flow-lib-v1.0.0.tgz
    mkdir -p nxp-lib
    tar xzf /tmp/dm-eiq-payload.tgz -C nxp-lib/
    rm -f /tmp/dm-eiq-payload.tgz
    echo "payload unpacked: $(sed -n 's/^commit:  *//p' nxp-lib/VERSION.txt)"
else
    echo "=== payload: already present or skipped ==="
fi

if [ ! -d nxp-lib/src/vit ]; then
    echo "ERROR: no payload at nxp-lib/src and --skip-payload was given." >&2
    exit 1
fi

# --- espeak-ng -------------------------------------------------------------------------------
# TTS's phonetizer links against it at runtime and the image has no packaged version (apt on this
# Yocto build is a non-functional stub). NXP builds it with --prefix=/usr; we keep /usr pristine and
# point falcon at /opt/dm-eiq instead. --with-extdict-cmn is NXP's Mandarin dictionary, dropped
# because we are English-only.

if [ "$skip_espeak" = false ] && [ ! -x /opt/dm-eiq/bin/espeak-ng ]; then
    echo "=== building espeak-ng 1.51 into /opt/dm-eiq (a few minutes) ==="
    rm -rf /tmp/espeak-build
    mkdir -p /tmp/espeak-build
    curl -L -o /tmp/espeak-build/espeak-ng.tar.gz \
        https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.51.tar.gz
    tar xf /tmp/espeak-build/espeak-ng.tar.gz -C /tmp/espeak-build
    cd /tmp/espeak-build/espeak-ng-1.51
    ./autogen.sh
    ./configure --prefix=/opt/dm-eiq \
        --with-klatt=no --with-speechplayer=no --with-mbrola=no \
        --with-extdict-ru=no --with-extdict-cmn=no --with-extdict-yue=no
    make -j"$(nproc)"
    make install
    cd - > /dev/null
    rm -rf /tmp/espeak-build
    echo "espeak-ng installed under /opt/dm-eiq"
else
    echo "=== espeak-ng: already present or skipped ==="
fi

# --- venv and Python packages ----------------------------------------------------------------

if [ "$skip_pip" = false ]; then
    if [ ! -d venv ]; then
        echo "=== creating venv (--system-site-packages) ==="
        python3 -m venv --system-site-packages venv
    fi
    echo "=== installing requirements.txt (torch is large; expect several minutes) ==="
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt

    echo "=== checking that the BSP packages still win ==="
    ./venv/bin/python -c "
import numpy, onnxruntime
print('numpy       ', numpy.__version__, numpy.__file__)
print('onnxruntime ', onnxruntime.__version__)
assert 'NeutronExecutionProvider' in onnxruntime.get_available_providers(), 'Neutron provider lost!'
print('Neutron provider present')
"
fi

echo
echo "Done. Run the pilot with:  ./run.sh <stage>"
