"""Audio device selection and levels for the i.MX95 EVK.

Two jobs: turn a friendly device name into an ALSA device string, and set the gain that NXP's own
code does not.

**Device naming.** `audio.json` may name a device three ways, so swapping in a USB microphone later
needs no knowledge of ALSA syntax:

    "micfil"                      an alias from ALIASES below
    "Fifine"                      any substring of a card name in `arecord -l`
    "plughw:CARD=C920,DEV=0"      a full ALSA device string, passed through untouched

**Gain.** NXP's audio manager raises mixer levels on startup in `set_audio_device_config.py`, but only
for the wm8960 and wm8962 codecs fitted to their other EVKs -- its final branch is a debug log. This
board has `micfil` for capture and `mqs` for playback, so nothing fires and the levels stay at their
quiet defaults. That is the whole explanation for the barely audible loopback on this hardware. We
cannot edit the payload, so we set the levels here, before the audio manager is constructed.

The two ends are not symmetric:

- **Capture has hardware gain.** `micfil` exposes a 0-15 `Range` per channel; the C920's USB
  descriptor exposes a 0-15 `Mic Capture Volume` spanning 20-50 dB, already at maximum out of the box.
- **Playback has none.** `mqsaudio` exposes zero mixer controls, so output level can only be raised
  digitally, by scaling samples before they reach the sink.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "audio.json"

ALIASES = {
    "micfil": "micfilaudio",
    "c920": "C920",
    "mqs": "mqsaudio",
}

CARD_LINE = re.compile(r"^card (\d+): (\S+) \[([^\]]+)\], device (\d+):")


@dataclass
class AudioConfig:
    capture_device: str
    playback_device: str
    micfil_range: int = 10
    capture_gain: float = 1.0
    playback_gain: float = 1.0


def load_audio_config(path: Path = CONFIG_PATH) -> AudioConfig:
    """Read audio.json, resolving both device names to ALSA device strings."""
    settings = {key: value for key, value in json.loads(path.read_text()).items() if not key.startswith("_")}
    config = AudioConfig(**settings)
    config.capture_device = resolve_device(config.capture_device, is_capture=True)
    config.playback_device = resolve_device(config.playback_device, is_capture=False)
    return config


def list_devices(is_capture: bool) -> list[tuple[str, str]]:
    """Every (card_id, description) ALSA reports, e.g. ("C920", "HD Pro Webcam C920")."""
    result = subprocess.run(["arecord" if is_capture else "aplay", "-l"], capture_output=True, text=True)
    found = []
    for line in result.stdout.splitlines():
        match = CARD_LINE.match(line)
        if match:
            found.append((match.group(2), match.group(3)))
    return found


def resolve_device(name: str, is_capture: bool) -> str:
    """Turn an alias, a card-name substring, or a full ALSA string into an ALSA device string."""
    if ":" in name or name in ("default", "null", "pipewire"):
        return name

    wanted = ALIASES.get(name.lower(), name)
    for card_id, description in list_devices(is_capture):
        if wanted.lower() in card_id.lower() or wanted.lower() in description.lower():
            return f"plughw:CARD={card_id},DEV=0"

    available = ", ".join(card_id for card_id, _ in list_devices(is_capture)) or "none"
    raise ValueError(f"No {'capture' if is_capture else 'playback'} device matching '{name}'. Available: {available}")


def _run_amixer(*arguments: str) -> bool:
    result = subprocess.run(("amixer", *arguments), capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("amixer %s failed: %s", " ".join(arguments), result.stderr.strip())
        return False
    return True


def set_micfil_range(range_level: int) -> None:
    """Set the PDM range on all eight micfil channels. Valid range is 0-15, ships at 6."""
    for channel in range(1, 9):
        _run_amixer("-c", "micfilaudio", "cset", f"numid={channel}", str(range_level))
    logger.info("micfil capture range set to %d/15 on 8 channels", range_level)


def prepare_capture_device(capture_device: str, micfil_range: int = 10) -> None:
    """Raise the hardware capture gain for whichever microphone we are about to open."""
    if "micfil" in capture_device:
        set_micfil_range(micfil_range)
    elif "C920" in capture_device:
        # Already 15/15 from the factory, so this is a guard against something having lowered it.
        _run_amixer("-c", "C920", "sset", "Mic Capture Volume", "15")
    else:
        logger.info("No gain profile for '%s' -- leaving mixer levels alone", capture_device)


def apply_digital_gain(samples, gain: float):
    """Scale samples by `gain`, clipping at full scale, preserving the input dtype.

    Needed at both ends of this board: to lift a quiet microphone above VIT's noise floor, and to get
    usable loudness out of `mqs`, which has no hardware volume control at all.
    """
    import numpy as np

    if gain == 1.0:
        return samples
    scaled = samples.astype(np.float32) * gain
    if np.issubdtype(samples.dtype, np.integer):
        limits = np.iinfo(samples.dtype)
        return np.clip(scaled, limits.min, limits.max).astype(samples.dtype)
    return np.clip(scaled, -1.0, 1.0).astype(samples.dtype)


def get_peak_dbfs(samples) -> float:
    """Peak level of a buffer in dBFS, for judging whether a microphone is actually working."""
    import numpy as np

    if not len(samples):
        return -999.0
    if np.issubdtype(samples.dtype, np.integer):
        peak = float(np.abs(samples).max()) / float(np.iinfo(samples.dtype).max)
    else:
        peak = float(np.abs(samples).max())
    return 20.0 * np.log10(peak) if peak > 0 else -999.0
