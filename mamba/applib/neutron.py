"""Where the Neutron runtime is, and whether it matches the models we are about to load.

The BSP ships a Neutron delegate in `/usr/lib` and its firmware in `/lib/firmware`. We do not use
them: the SDK release they came from miscompiled the face embedder into a vector that ignored its
input, so every face matched every registered user. `install.sh` unpacks a known-good SDK into
`imx-eiq-neutron-sdk/` beside the demo and `run.sh` points the *kernel* at its firmware, leaving
`/usr` untouched and a fallback one environment variable away.

Three pieces have to be the same release - converter, delegate and firmware - and when they are not,
nothing raises: the numbers are simply wrong. That is the entire reason this file exists.
`check_models()` compares the firmware the board will actually run against the one that converted
the models, and `main.py` prints what it says. `models/version.txt` is written by
`scripts/package-models.sh` and travels inside the models tarball.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDK_DIR = ROOT / "imx-eiq-neutron-sdk"          # what install.sh unpacks; absent = BSP fallback
SDK_TARGET = SDK_DIR / "target" / "imx95"
BSP_DELEGATE_PATH = "/usr/lib/libneutron_delegate.so"

# The delegate is loaded by path, so pointing at the SDK's copy needs no system change at all.
# `run.sh` puts the SDK's lib directory on LD_LIBRARY_PATH for the libNeutronDriver.so beside it.
_SDK_DELEGATE = SDK_TARGET / "delegate" / "libneutron_delegate.so"
DELEGATE_PATH = str(_SDK_DELEGATE) if _SDK_DELEGATE.exists() else BSP_DELEGATE_PATH

# The firmware the kernel will boot: run.sh writes this directory into
# /sys/module/firmware_class/parameters/path, which the kernel searches *before* /lib/firmware.
_SDK_FIRMWARE = SDK_TARGET / "imx95" / "NeutronFirmware.elf"
FIRMWARE_PATH = _SDK_FIRMWARE if _SDK_FIRMWARE.exists() else Path("/lib/firmware/NeutronFirmware.elf")


def check_models(models_dir: Path) -> str | None:
    """Return a warning to print, or None when the models match the runtime.

    Deliberately shallow. It answers one question - were these models converted by the SDK whose
    firmware is about to run them - and says nothing when the answer is yes.
    """
    version_file = models_dir / "version.txt"
    if not version_file.exists():
        return (f"{models_dir}/version.txt is missing, so the models cannot be matched to the "
                f"Neutron runtime. Rebuild them with scripts/package-models.sh.")
    stamped = dict(
        line.split("=", 1) for line in version_file.read_text().split() if "=" in line
    )
    if stamped.get("firmware_md5") == _md5(FIRMWARE_PATH):
        return None
    where = "the SDK beside the demo" if _SDK_FIRMWARE.exists() else "the BSP's own /lib/firmware"
    return (f"The models were converted by {stamped.get('sdk', 'an unknown SDK')}, but the Neutron "
            f"firmware about to run them is {where}, which is a different build. Expect wrong "
            f"results rather than errors - faces that all match, or nothing detected at all. "
            f"Unpack the matching SDK beside install.sh and re-run it.")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else ""
