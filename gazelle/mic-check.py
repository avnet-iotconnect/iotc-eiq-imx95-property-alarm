#!/usr/bin/env python3
"""Record, meter, play back — the tool for setting up audio on this board.

Deliberately standalone: pure GStreamer plus numpy, no eIQ payload and no venv, so it works before
anything else is installed and cannot be broken by the rest of the pilot. It is meant to stay in the
demo permanently — it is the first thing to run when a booth microphone misbehaves.

    ./mic-check.py --list                   # what audio hardware this board has
    ./mic-check.py                          # record, meter and play back, using audio.json
    ./mic-check.py --loop                   # keep going: record, listen, adjust, repeat
    ./mic-check.py --mic Fifine --range 8   # try another mic and another hardware gain
    ./mic-check.py --gain 4                 # digital boost on playback only

Devices come from audio.json and may be named by alias, by any substring of a card name, or by a
full ALSA device string — see audio_devices.py. Anything passed on the command line wins.

Reading the meter: the bar is loudness *right now*, on a scale from silence to digital full scale.
Aim to keep speech peaks between -20 and -6 dBFS. Below -40 the wake word will struggle; above -1
the signal is clipping and distorting, which hurts recognition more than being slightly quiet.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import gi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_devices  # noqa: E402

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

SAMPLE_RATE = 16000
METER_CELLS = 40
METER_FLOOR_DBFS = -60.0


def to_dbfs(samples: np.ndarray) -> float:
    if not len(samples):
        return -999.0
    level = float(np.abs(samples).max()) / 32768.0
    return 20.0 * np.log10(level) if level > 0 else -999.0


def meter_legend() -> str:
    """Tick marks aligned to the bar below, which spans -60..0 dBFS across 40 cells."""
    return "   -60" + " " * 10 + "-40" + " " * 11 + "-20" + " " * 6 + "-6" + " " * 2 + "0  dBFS"


def draw_meter(dbfs: float) -> str:
    fraction = (dbfs - METER_FLOOR_DBFS) / (0.0 - METER_FLOOR_DBFS)
    filled = int(max(0.0, min(1.0, fraction)) * METER_CELLS)
    warning = "  CLIPPING" if dbfs > -1.0 else ""
    return f"  [{'#' * filled}{'.' * (METER_CELLS - filled)}] {dbfs:6.1f}{warning}"


def show_devices() -> None:
    for label, is_capture in (("capture (microphones)", True), ("playback (speakers)", False)):
        print(f"\n{label}:")
        devices = audio_devices.list_devices(is_capture=is_capture)
        if not devices:
            print("  none")
        for card_id, description in devices:
            print(f"  {card_id:<16} {description}")
            print(f"  {'':<16} plughw:CARD={card_id},DEV=0")
    print("\nName any of these in audio.json by alias, by a substring, or in full.")


def record(device: str, seconds: float) -> np.ndarray:
    pipeline = Gst.parse_launch(
        f"alsasrc device={device} ! "
        f"audioconvert ! audioresample ! "
        f"audio/x-raw,format=S16LE,rate={SAMPLE_RATE},channels=1 ! "
        f"appsink name=sink max-buffers=8 drop=false sync=false"
    )
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)

    print(f"\nRecording {seconds:g} s from {device} — talk now\n")
    print(meter_legend())

    collected: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    wanted = int(SAMPLE_RATE * seconds)
    meter_window = SAMPLE_RATE // 10  # one line per 100 ms, readable over ssh
    total = 0
    while total < wanted:
        sample = sink.emit("pull-sample")
        if sample is None:
            break
        ok, info = sample.get_buffer().map(Gst.MapFlags.READ)
        if not ok:
            continue
        chunk = np.frombuffer(info.data, dtype=np.int16).copy()
        sample.get_buffer().unmap(info)
        collected.append(chunk)
        pending.append(chunk)
        total += len(chunk)

        if sum(len(piece) for piece in pending) >= meter_window:
            print(draw_meter(to_dbfs(np.concatenate(pending))), flush=True)
            pending = []

    pipeline.set_state(Gst.State.NULL)
    return np.concatenate(collected) if collected else np.zeros(0, dtype=np.int16)


def play(samples: np.ndarray, device: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = handle.name
    with wave.open(path, "w") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(samples.tobytes())

    pipeline = Gst.parse_launch(
        f"filesrc location={path} ! wavparse ! audioconvert ! audioresample ! "
        f"alsasink device={device} sync=false"
    )
    pipeline.set_state(Gst.State.PLAYING)
    pipeline.get_bus().timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipeline.set_state(Gst.State.NULL)
    Path(path).unlink(missing_ok=True)


def report(samples: np.ndarray) -> None:
    peak = to_dbfs(samples)
    rms_level = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) / 32768.0
    rms = 20.0 * np.log10(rms_level) if rms_level > 0 else -999.0
    print(f"\n  peak {peak:6.1f} dBFS    rms {rms:6.1f} dBFS    {len(samples) / SAMPLE_RATE:.1f} s")
    if peak < -40:
        print("  -> too quiet. Raise --range, or move closer to the mic.")
    elif peak > -1:
        print("  -> clipping. Lower --range.")
    elif peak > -6:
        print("  -> a little hot. Peaks this close to full scale distort; consider lowering --range.")
    else:
        print("  -> good level.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show the board's audio devices and exit")
    parser.add_argument("--mic", help="override audio.json's capture device")
    parser.add_argument("--speaker", help="override audio.json's playback device")
    parser.add_argument("--range", type=int, help="override audio.json's micfil hardware gain (0-15)")
    parser.add_argument("--gain", type=float, help="override audio.json's playback digital gain")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--no-playback", action="store_true", help="measure only, stay silent")
    parser.add_argument("--loop", action="store_true", help="repeat until Ctrl-C")
    args = parser.parse_args()

    if args.list:
        show_devices()
        return 0

    config = audio_devices.load_audio_config()
    capture = audio_devices.resolve_device(args.mic, is_capture=True) if args.mic else config.capture_device
    speaker = audio_devices.resolve_device(args.speaker, is_capture=False) if args.speaker else config.playback_device
    micfil_range = args.range if args.range is not None else config.micfil_range
    gain = args.gain if args.gain is not None else config.playback_gain

    audio_devices.prepare_capture_device(capture, micfil_range)

    while True:
        samples = record(capture, args.seconds)
        if not len(samples):
            print(f"No audio captured from {capture}. Try --list.", file=sys.stderr)
            return 1
        report(samples)

        if not args.no_playback:
            print(f"\nPlaying back on {speaker} at {gain:g}x ...")
            play(audio_devices.apply_digital_gain(samples, gain), speaker)

        if not args.loop:
            return 0
        input("\nEnter to record again, Ctrl-C to stop.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
