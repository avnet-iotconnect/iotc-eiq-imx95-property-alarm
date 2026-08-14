#!/usr/bin/env bash
#
# Pull NXP's eIQ payload into ./nxp-lib so your editor can resolve `import vit`, `import tts` and
# friends, and you can read the sources you are calling into.
#
#   source .venv/bin/activate && scripts/local-setup.sh
#
# Activate your venv first -- this adds a .pth to it and will not create one for you. Install
# src/requirements.txt yourself if you want the third-party imports to resolve too.
#
# Nothing here runs on a PC: the payload's .so files are cpython-313-aarch64 and only load on the
# board. This is for reading and navigating. To run anything on the board, see the top-level README.md.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

[ -n "${VIRTUAL_ENV:-}" ] || { echo "ERROR: activate a venv first (this script will not make one)." >&2; exit 1; }

[ -f dm-eiq-genai-flow-lib.tgz ] || curl -fL -o dm-eiq-genai-flow-lib.tgz \
    https://downloads.iotconnect.io/partners/nxp/packages/dm-eiq-genai-flow-lib-v1.0.0.tgz

rm -rf nxp-lib && mkdir nxp-lib && tar xzf dm-eiq-genai-flow-lib.tgz -C nxp-lib
echo "$PWD/nxp-lib/src" > "$(echo "$VIRTUAL_ENV"/lib/python*/site-packages)/nxp-lib.pth"

echo "Unpacked into nxp-lib/, registered with $VIRTUAL_ENV"
