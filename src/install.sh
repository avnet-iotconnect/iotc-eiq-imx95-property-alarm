#!/bin/bash
# Prepare a board to run the demo. Run it ON the board, once:  ./install.sh
#
# NXP's own install.sh is not used: it installs packages system-wide, edits ~/.bashrc, and builds
# alsa-lib into /usr/local, leaving a second libasound.so.2 for audio programs to bind to. This
# writes to nxp-lib/, /opt/dm-eiq and venv/ only, so removing those three returns the image to
# stock. The vision models are converted on the host and copied over; the LLM behind 'agent' has its
# own installer in connector/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if [ ! -f 'device-pkey.pem' ]; then
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 365 -keyout device-pkey.pem -out device-cert.pem -subj "/CN=YourCommonName"
fi

# The eIQ Neutron SDK, which you drop beside this script as a zip - NXP-licensed, so like the eIQ
# payload below it cannot be redistributed, and unlike the payload we cannot even host it. It is
# unpacked to a fixed, unversioned directory name so run.sh finds it without being told the version.
# The BSP has a Neutron runtime of its own and this replaces *nothing*: run.sh points the delegate
# and the kernel's firmware search at this copy, so removing the directory reverts to the BSP.
# Its release must match the one that converted models/ - scripts/package-models.sh stamps that
# into models/version.txt and the demo checks it at startup.
# `|| true` because `set -e` would otherwise take the missing-zip case as a fatal error rather than
# letting it reach the warning below.
NEUTRON_SDK_ZIP=$(ls eiq-neutron-sdk-linux-*.zip 2>/dev/null | head -1 || true)
if [ -n "$NEUTRON_SDK_ZIP" ]; then
  rm -rf imx-eiq-neutron-sdk
  unzip -q "$NEUTRON_SDK_ZIP" -d imx-eiq-neutron-sdk
  echo "Neutron SDK: $NEUTRON_SDK_ZIP -> imx-eiq-neutron-sdk/"
else
  echo "WARNING: no eiq-neutron-sdk-linux-*.zip here. The demo will use the BSP's Neutron runtime,"
  echo "         which on some BSPs silently produces wrong results. See README.md."
fi

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
  wget -O /tmp/iotc-property-alarm-models.tgz https://downloads.iotconnect.io/partners/nxp/packages/iotc-property-alarm-models-v2.0.0.tgz
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
