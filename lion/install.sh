#!/bin/bash
# Prepare a board to run the demo. Run it ON the board, once:  ./install.sh
#
# NXP's own install.sh is not used: it installs packages system-wide, edits ~/.bashrc, and builds
# alsa-lib into /usr/local, leaving a second libasound.so.2 for audio programs to bind to. This
# writes to nxp-lib/, /opt/dm-eiq and venv/ only, so removing those three returns the image to
# stock. The vision models are converted on the host and copied over; the LLM behind 'ask' has its
# own installer in connector/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# NXP's eIQ payload: 139 MB of compiled modules and encrypted weights, hosted rather than committed
# because the licence does not allow redistributing it. Built by scripts/package-dm-eiq.sh.
wget -O /tmp/dm-eiq-payload.tgz https://downloads.iotconnect.io/partners/nxp/packages/dm-eiq-genai-flow-lib-v1.0.0.tgz
mkdir -p nxp-lib
tar xzf /tmp/dm-eiq-payload.tgz -C nxp-lib/
rm -f /tmp/dm-eiq-payload.tgz

# --system-site-packages so the BSP's numpy, cv2, gi, onnxruntime and tflite_runtime stay visible:
# those are ABI-matched to the Neutron delegate and must never be replaced by pip's builds.
python3 -m venv --system-site-packages venv
./venv/bin/pip install -r requirements.txt

# espeak-ng, which TTS's phonetizer links against at runtime and this image does not package. NXP
# builds it into /usr; we keep /usr pristine. The --with flags drop voices and dictionaries we never
# use and would otherwise wait on. Last, because it is the only step that leaves this directory.
curl -L https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.51.tar.gz | tar xz -C /tmp
pushd /tmp/espeak-ng-1.51 >/dev/null
cd /tmp/espeak-ng-1.51
./autogen.sh
./configure --prefix=/opt/dm-eiq --with-klatt=no --with-speechplayer=no --with-mbrola=no --with-extdict-ru=no --with-extdict-cmn=no --with-extdict-yue=no
make -j"$(nproc)"
make install
popd >/dev/null

print_warning="no"
if [ ! -d models/ ]; then
  wget -O /tmp/iotc-property-alarm-models.tgz https://downloads.iotconnect.io/partners/nxp/packages/iotc-property-alarm-models-v1.0.0.tgz
  tar xzf /tmp/iotc-property-alarm-models.tgz
  rm -f /tmp/iotc-property-alarm-models.tgz
else
  print_warning="yes" # after prefetch stdout spam
fi

./run.sh --prefetch

if [ "$print_warning" = "yes" ]; then
  echo "=========================== WARNING ====================================="
  echo "models/ directory found. Not installing to allow for local updates."
  echo "If you intend to use the recommended models, delete the models/ directory"
  echo "========================================================================="
fi

echo "Done. Next, execute:"
echo "connector/install.sh && connector/run.sh # If you have Ara240"
echo "./run.sh # will execute the application"
