#!/usr/bin/env python3
"""gazelle pilot - the anti-theft demo, driven by voice, on the FRDM-IMX95.

elephant proved the eyes (camera -> YOLO11n on Neutron -> tracking -> faces -> alarm) and falcon
proved the ears and the mouth (NXP's eIQ wake word, STT, TTS and SmolVLM, used as a library).
gazelle is both at once: the same 30 fps video loop, now taking spoken commands.

    ./run.sh                                   # NPU for YOLO and SFace, voice on
    ./run.sh --no-voice                        # just the video loop, no eIQ payload needed
    ./run.sh --model yolo11n_int8.tflite       # full-CPU baseline

Say "Hey NXP", wait for the blip, then one of:

    register user Michael      unregister user Michael     unregister last user
    arm the alarm              disarm the alarm
    lock the laptop            unlock the laptop           describe the scene

This file is the *composition root*: the only place modules are constructed and wired. Read `main()`
top to bottom to see the whole object graph; read `app.py` to see every reaction to an event.

Three threads, and the whole design is about keeping them apart:

    main       camera -> YOLO -> tracking -> overlay, 30 fps, never blocked by anything below
    face       one background thread, ~5 Hz, fills in who each person is        (face_worker.py)
    voice      one thread: loads eIQ, waits for the wake word, transcribes      (voice.py)
    commands   a pool of 4: runs what was said, and speaks the answer           (commands.py)

The voice thread does its own loading, so the picture is live in about a second while the eIQ models
take their ~15 s. Watch the HUD's `voice:` line for when it is ready.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

# NXP's library lives inside the pilot, but under its own directory: everything in nxp-lib/ is
# theirs and never edited, everything beside it is ours. One sys.path root replaces the nine that
# their `pip install -e .` would have set up. Done before importing voice.py or scene.py.
PAYLOAD = Path(__file__).resolve().parent / "nxp-lib"
sys.path.insert(0, str(PAYLOAD / "src"))

import audio_devices  # noqa: E402
import vocabulary  # noqa: E402
from app import AntiTheftApp  # noqa: E402
from camera import Camera  # noqa: E402
from commands import CommandService  # noqa: E402
from detector import Detector  # noqa: E402
from face import FaceRecognizer  # noqa: E402
from face_worker import FaceWorker  # noqa: E402
from overlay import Overlay  # noqa: E402
from registry import Registry  # noqa: E402
from scene import SceneDescriber  # noqa: E402
from state import SessionState  # noqa: E402
from tracking import Tracker  # noqa: E402
from voice import VoiceLoop  # noqa: E402

VLM_WEIGHTS = Path(__file__).resolve().parent / "models"


class TextCommandFile:
    """Debug-only command source: a line written into a file is run as if it had been spoken.

    Off unless `--text-commands` is given. It exists because a booth microphone can fail in ways
    that have nothing to do with the demo, and because typing a command is a faster way to test a
    handler than saying it forty times. It goes through the *same* CommandService as voice - which
    is exactly the seam /IOTCONNECT C2D will plug into next.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def poll(self) -> list[str]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        lines = [line for line in self.path.read_text().splitlines() if line.strip()]
        self.path.write_text("")  # consume: a command runs once, not every frame until edited
        return lines


def run_loop(
    camera: Camera, detector: Detector, tracker: Tracker, face_worker: FaceWorker,
    overlay: Overlay, app: AntiTheftApp, service: CommandService,
    text_commands: TextCommandFile | None, max_frames: int, log_every: int,
) -> None:
    steady_state: list[dict[str, float]] = []
    frame_index = 0
    while max_frames == 0 or frame_index < max_frames:
        start = perf_counter()
        frame = camera.read()
        if frame is None:
            print("camera returned no frame (timeout/EOS) - stopping")
            break
        after_capture = perf_counter()

        detections = detector.detect(frame)
        after_inference = perf_counter()

        tracks = tracker.update(detections)
        face_worker.maybe_dispatch(frame, tracks)  # async, at most every ~200ms on a spare core
        face_worker.apply(tracks)                   # merge the last cycle's names onto the tracks (fast)
        after_track = perf_counter()

        app.on_frame(frame, tracks)                 # alarm state + overlay; never blocks
        if text_commands is not None:
            for line in text_commands.poll():
                service.submit(line, source="file")  # fire and forget: the answer is spoken

        stage_ms = {
            "capture": (after_capture - start) * 1000,
            "inference": (after_inference - after_capture) * 1000,
            "track": (after_track - after_inference) * 1000,
            "app": (perf_counter() - after_track) * 1000,
        }
        overlay.set_timing(stage_ms["inference"], sum(stage_ms.values()))
        if log_every and frame_index % log_every == 0:
            report_frame(frame_index, tracks, stage_ms)
        if frame_index >= 1:  # frame 0 is warm-up
            steady_state.append(stage_ms)
        frame_index += 1

    report_summary(steady_state)


def report_frame(index: int, tracks, stage_ms: dict[str, float]) -> None:
    labels = ", ".join(f"#{t.short_id or '-'} {t.identity or t.class_name}" for t in tracks) or "(nothing)"
    timing = "  ".join(f"{stage}={ms:5.1f}ms" for stage, ms in stage_ms.items())
    total = sum(stage_ms.values())
    print(f"frame {index:04d} | {timing}  total={total:5.1f}ms ({1000 / total:4.1f} fps) | {labels}")


def report_summary(steady_state: list[dict[str, float]]) -> None:
    if not steady_state:
        return
    print("-" * 60)
    stages = steady_state[0].keys()
    means = {stage: sum(f[stage] for f in steady_state) / len(steady_state) for stage in stages}
    total = sum(means.values())
    print(f"Steady-state mean over {len(steady_state)} frames (warm-up frame 0 excluded):")
    for stage, ms in means.items():
        print(f"  {stage:11s}: {ms:6.2f} ms")
    print(f"  {'TOTAL':11s}: {total:6.2f} ms  ->  {1000 / total:.1f} fps end-to-end")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    vision = parser.add_argument_group("vision (as in elephant)")
    vision.add_argument("--model", default="yolo11n_neutron.tflite", help="path to the YOLO .tflite model")
    vision.add_argument("--delegate", action="store_true",
                        help="force the Neutron NPU delegate (auto-on when the model name contains 'neutron')")
    vision.add_argument("--yunet", default="face_detection_yunet_2023mar.onnx", help="YuNet ONNX (detect)")
    vision.add_argument("--sface", default="face_recognition_sface_2021dec.onnx", help="SFace ONNX (align only)")
    vision.add_argument("--sface-model", default="",
                        help="SFace embedder tflite (default: sface_neutron.tflite with --delegate, else sface_int8.tflite)")
    vision.add_argument("--face-interval", type=float, default=0.2,
                        help="min seconds between async face-recognition passes (0.2 = ~5 Hz)")
    vision.add_argument("--device", default="/dev/video4", help="v4l2 camera device (C920 capture node)")
    vision.add_argument("--width", type=int, default=640, help="camera capture width")
    vision.add_argument("--height", type=int, default=480, help="camera capture height")
    vision.add_argument("--conf", type=float, default=0.25, help="minimum YOLO score to keep a detection")
    vision.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    vision.add_argument("--threads", type=int, default=0, help="tflite CPU threads (0 = auto: 6 CPU / 2 NPU)")
    vision.add_argument("--no-preview", action="store_true", help="skip the Wayland preview window")
    vision.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = until Ctrl-C)")
    vision.add_argument("--log-every", type=int, default=30,
                        help="print one frame line every N frames (0 = never); voice logs get lost at 30/s")

    voice = parser.add_argument_group("voice (as in falcon)")
    voice.add_argument("--no-voice", action="store_true", help="run the video loop only, no eIQ payload")
    voice.add_argument("--mic", help="capture device, overriding audio.json (alias, substring, or ALSA name)")
    voice.add_argument("--speaker", help="playback device, overriding audio.json")
    voice.add_argument("--range", type=int, help="micfil hardware gain 0-15, overriding audio.json")
    voice.add_argument("--gain", type=float, help="digital playback gain, overriding audio.json")
    voice.add_argument("--capture-gain", type=float,
                       help="digital gain on captured samples, overriding audio.json -- the lever to try "
                            "when the wake word will not fire and mic-check.py says the mic is quiet")
    voice.add_argument("--tts-voice", type=int, default=24, help="TTS speaker id, 1-904")
    voice.add_argument("--wake-timeout", type=float, default=3.0,
                       help="seconds to wait for the user to START talking after the wake word")
    voice.add_argument("--command-seconds", type=float, default=6.0, help="longest command we will listen to")
    voice.add_argument("--no-repeat", action="store_true", help="do not say the transcript back before answering")

    control = parser.add_argument_group("commands, state and the VLM")
    control.add_argument("--command-threads", type=int, default=4,
                         help="how many commands may run at once (kept low so the video keeps its cores)")
    control.add_argument("--db", default="faces.json", help="persistent face database (name -> embedding)")
    control.add_argument("--state", default="state.json", help="persistent alarm state + locked objects")
    control.add_argument("--keyframe", default="scene.jpg", help="where 'describe scene' writes the frame it sent")
    control.add_argument("--snapshot-dir", default=".", help="where the 'snapshot' command writes its JPEGs")
    control.add_argument("--no-vlm", action="store_true", help="disable 'describe scene'")
    control.add_argument("--preload-vlm", action="store_true",
                         help="load SmolVLM at startup (~12 s) instead of on the first 'describe scene'")
    control.add_argument("--prefetch", action="store_true",
                         help="download and decrypt SmolVLM, then exit -- run once on a network you trust")
    control.add_argument("--vlm-tokens", type=int, default=64, help="longest scene description, in tokens")
    control.add_argument("--text-commands", default="",
                         help="debug: also read commands from this file, one per line (e.g. command.txt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prefetch:
        # Booth prep: everything else ships in the payload, SmolVLM comes from Hugging Face. A trade
        # show floor is the wrong place to discover that. No camera, no models, no eIQ - just this.
        SceneDescriber(PAYLOAD, VLM_WEIGHTS, Path(args.keyframe)).load()
        print(f"SmolVLM cached in {VLM_WEIGHTS}. Nothing else reaches the network at run time.")
        return

    # A '..._neutron.tflite' model carries a NeutronGraph op only the delegate can run, so the backend
    # must follow the model - infer it from the name so `./run.sh` just works without remembering --delegate.
    use_neutron = args.delegate or "neutron" in Path(args.model).name
    # CPU path parallelizes convs across all cores; NPU path runs them on Neutron, so extra CPU threads
    # only oversubscribe the camera/preview/face/voice threads. Auto-pick (same reasoning as dolphin).
    num_threads = args.threads if args.threads > 0 else (2 if use_neutron else 6)

    detector = Detector(args.model, use_neutron, num_threads, args.conf, args.iou)
    print(detector.describe())
    sface_model = args.sface_model or ("sface_neutron.tflite" if use_neutron else "sface_int8.tflite")
    print(f"Face embed : {sface_model} ({'Neutron NPU' if use_neutron else 'CPU'})")
    face = FaceRecognizer(args.yunet, args.sface, sface_model, use_neutron=use_neutron)
    registry = Registry(args.db)
    face_worker = FaceWorker(face, registry, args.face_interval)
    tracker = Tracker()
    overlay = Overlay(args.width, args.height, backend="NPU" if use_neutron else "CPU")
    state = SessionState(args.state)
    vocab = vocabulary.load_vocabulary()
    scene = None if args.no_vlm else SceneDescriber(
        PAYLOAD, VLM_WEIGHTS, Path(args.keyframe), max_new_tokens=args.vlm_tokens)
    app = AntiTheftApp(registry, overlay, face_worker, state, vocab, scene, Path(args.snapshot_dir))
    service = CommandService(app.on_command, max_workers=args.command_threads)
    text_commands = TextCommandFile(args.text_commands) if args.text_commands else None

    voice = None
    if not args.no_voice:
        audio_config = resolve_audio_config(args)
        voice = VoiceLoop(
            PAYLOAD, audio_config,
            on_command=lambda text: service.submit(text, source="voice").result().message,
            on_status=overlay.set_voice_status, speaker_id=args.tts_voice,
            wake_timeout_s=args.wake_timeout, command_seconds=args.command_seconds,
            is_repeating=not args.no_repeat,
        )
    else:
        overlay.set_voice_status("voice: off")

    print(f"Known users: {', '.join(registry.user_names) or '(none)'}")
    print(f"Alarm      : {app.alarm_state.label}   locked: {', '.join(state.locked_objects) or '-'}")
    if scene is not None and args.preload_vlm:
        scene.load()
    print("-" * 60)

    camera = Camera(args.device, args.width, args.height, show_preview=not args.no_preview)
    camera.connect_overlay(overlay.draw)
    camera.start()
    if voice is not None:
        voice.start()  # loads eIQ on its own thread; the video does not wait for it
    try:
        run_loop(camera, detector, tracker, face_worker, overlay, app, service,
                 text_commands, args.max_frames, args.log_every)
    except KeyboardInterrupt:
        print("\ninterrupted - stopping")
    finally:
        if voice is not None:
            voice.stop()
        service.stop()
        face_worker.stop()
        camera.stop()


def resolve_audio_config(args) -> audio_devices.AudioConfig:
    """audio.json holds the defaults; anything given on the command line wins."""
    config = audio_devices.load_audio_config()
    if args.mic:
        config.capture_device = audio_devices.resolve_device(args.mic, is_capture=True)
    if args.speaker:
        config.playback_device = audio_devices.resolve_device(args.speaker, is_capture=False)
    if args.range is not None:
        config.micfil_range = args.range
    if args.gain is not None:
        config.playback_gain = args.gain
    if args.capture_gain is not None:
        config.capture_gain = args.capture_gain
    return config


if __name__ == "__main__":
    main()
