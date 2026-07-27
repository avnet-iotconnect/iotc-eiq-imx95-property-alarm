#!/usr/bin/env bash
# run.sh - launch a falcon stage ON THE BOARD.
#   ./run.sh devices     # what audio hardware exists, and how loud the mic is
#   ./run.sh wake        # say "Hey NXP"
#   ./run.sh voice       # say "Hey NXP", then a command -- the real flow
#   ./run.sh tts         # listen, say nothing
#   ./run.sh vlm         # describe the reference doorbell image
#   ./run.sh prefetch    # pull everything off the network NOW, not at the trade show
# Uses the pilot venv at ~/falcon/venv (created by install.sh).
# Override with PYTHON=/path/to/python ./run.sh ...
#
# espeak-ng lives under /opt/dm-eiq rather than /usr, so the loader and the phonetiser both have to
# be told where to look. ESPEAK_DATA_PATH points at the directory *containing* espeak-ng-data.
set -euo pipefail
cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/dm-eiq/lib:${LD_LIBRARY_PATH:-}"
export ESPEAK_DATA_PATH="/opt/dm-eiq/share"
export PATH="/opt/dm-eiq/bin:$PATH"

# torch will otherwise spawn a thread per core. Leave headroom for the vision loop.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

"${PYTHON:-./venv/bin/python}" main.py "$@"
