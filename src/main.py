#!/usr/bin/env python3
"""The anti-theft demo on the FRDM-IMX95: vision on the Neutron NPU, voice, /IOTCONNECT, the
display streamed over WebRTC, and an LLM on the Ara-240 that answers plain English.

Everything a visitor could ask for is a command - spoken, or pressed on the dashboard - except
`agent`, a sentence answered by Qwen2.5-7B on the Ara-240 DNPU whose tools are the demo's own
command handlers. "Please disarm the alarm" and "who can you see?" both work, and neither is
parsed by us.

Nothing here needs to be told where the hardware is. The camera device in particular is found at
startup rather than written down: `/dev/videoN` moves with the BSP and with whatever else is
plugged in. See `applib/camera_devices.py`.

    ./run.sh                                   # NPU for YOLO and SFace, voice on, cloud on, streaming
    ./run.sh --device /dev/video52             # name the camera yourself; otherwise it is found
    ./run.sh --no-agent                        # everything except the LLM
    ./run.sh --no-webrtc                       # everything except the live video
    ./run.sh --no-iotc                         # no cloud at all - the only way to run without it
    ./run.sh --no-voice                        # just the video loop, no eIQ payload needed
    ./run.sh --no-tts                          # listens, answers on the screen only - no speaker
    ./run.sh --model models/yolo11n_int8.tflite  # full-CPU baseline

Say "Hey NXP", wait for the blip, then one of:

    register user Michael      unregister user Michael     unregister last user
    arm the alarm              disarm the alarm
    lock the laptop            unlock the laptop           describe the scene

The layout, and the rule is one-way -- main.py -> app/ -> applib/ (see README.md):

    main.py       this file: the composition root, and the only place anything is wired
    app/          what the demo *does* - the commands, the watchdog, the LLM's tools. Read this.
    applib/       how it does it: camera, models, faces, audio, cloud, WebRTC
    connector/    NXP's eIQ AAF Connector, in its own venv - the LLM's home. Installed separately.
    config/       audio.json, vocabulary.json
    agenttools/   checks you run once when something is wrong, not part of the demo
    benchmarks/   numbers worth keeping

Threads, and the whole design is about keeping them apart:

    main       camera -> YOLO -> tracking -> overlay, 30 fps, never blocked by anything below
    face       one background thread, ~5 Hz, fills in who each person is        (face_worker.py)
    voice      one thread: loads eIQ, waits for the wake word, transcribes      (voice.py)
    commands   a pool of 4: runs what was said, and answers                     (commands.py)
    iotc       one thread publishing telemetry, plus paho's own MQTT thread     (iotc.py)
    faces      one thread polling the device's S3 folder for photographs        (face_uploads.py)
    webrtc     one thread running asyncio: signalling and every viewer          (webrtc.py)

The LLM gets no thread of its own: an `agent` question occupies one command worker for the ~30 s it
takes, and the model itself is in another process entirely, reached over HTTP (see `app/agent.py`).

The video for the stream never passes through any of them: GStreamer's own threads compose the
overlay, hand the frame to the i.MX95's hardware H.264 encoder and deliver finished packets. See
`camera.py` for the pipeline and `webrtc.py` for why that means no encoding on our CPU at all.

/IOTCONNECT is connected up front, on this thread, before anything is opened: the cloud is required,
so a bad certificate or an unreachable back end stops the demo there rather than being a line on a
HUD nobody reads. `--no-iotc` is the way to run without it. Voice and WebRTC do load and connect on
their own threads, so the picture is live a second after that. Watch the HUD's `voice:`, `cloud:`,
`stream:` and `agent:` lines for when each is ready.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from threading import Event, Timer
from time import perf_counter, sleep

# NXP's library lives inside the demo, but under its own directory: everything in nxp-lib/ is
# theirs and never edited, everything beside it is ours. One sys.path root replaces the nine that
# their `pip install -e .` would have set up. Done before importing voice.py or scene.py.
PAYLOAD = Path(__file__).resolve().parent / "nxp-lib"
sys.path.insert(0, str(PAYLOAD / "src"))

from app import agent  # noqa: E402
from app.agent import AgentService  # noqa: E402
from app.app import AntiTheftApp  # noqa: E402
from app.watchdog import Watchdog  # noqa: E402
from applib import audio_devices, camera_devices, commands, iotc, neutron, vocabulary, webrtc  # noqa: E402
from applib.camera import Camera, find_wayland_socket  # noqa: E402
from applib.commands import Command, CommandResult, CommandService  # noqa: E402
from applib.detector import Detector  # noqa: E402
from applib.eventlog import EventLog  # noqa: E402
from applib.face import FaceRecognizer  # noqa: E402
from applib.face_uploads import FaceUploads  # noqa: E402
from applib.face_worker import FaceWorker  # noqa: E402
from applib.guard import ObjectGuard  # noqa: E402
from applib.iotc import IotcClient  # noqa: E402
from applib.overlay import Overlay  # noqa: E402
from applib.registry import Registry  # noqa: E402
from applib.scene import SceneDescriber  # noqa: E402
from applib.state import SessionState  # noqa: E402
from applib.telemetry import TelemetryState  # noqa: E402
from applib.tracking import Tracker  # noqa: E402
from applib.voice import VoiceLoop  # noqa: E402
from applib.webrtc import WebRtcStreamer  # noqa: E402

MODELS = Path(__file__).resolve().parent / "models"  # converted on the host, copied over with us
VLM_WEIGHTS = MODELS / "vlm"                          # SmolVLM, pulled from Hugging Face by --prefetch
FACES = Path(__file__).resolve().parent / "faces"     # photographs downloaded from the device's S3 folder
VERSION = "2.1.0"       # reported as the 'version' telemetry attribute
FPS_REPORT_FRAMES = 15  # how often the frame loop refreshes the fps it tells the cloud
RESTART_DELAY_S = 3.0   # a restart waits this long, so its C2D ack reaches the cloud first
# How many frames the fps and the frame time on the HUD are averaged over. Two seconds' worth: a
# single frame's total swings by several milliseconds with whatever else the board is doing, and a
# number that flickers between 26 and 31 is one nobody can read - or compare against yesterday's.
TIMING_WINDOW_FRAMES = 60
# Commands whose answer is prose, arriving tens of seconds after anyone asked for it. See show_action.
SLOW_ANSWER_VERBS = {commands.AGENT, commands.DESCRIBE_SCENE}


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
    overlay: Overlay, app: AntiTheftApp, service: CommandService, telemetry: TelemetryState,
    text_commands: TextCommandFile | None, max_frames: int, log_every: int, stop_event: Event,
) -> None:
    steady_state: list[dict[str, float]] = []
    # (inference, end-to-end) for the last TIMING_WINDOW_FRAMES frames - what the HUD and the cloud
    # are told. Separate from `steady_state`, which is every frame of the run and is the exit report.
    recent_ms: deque[tuple[float, float]] = deque(maxlen=TIMING_WINDOW_FRAMES)
    frame_index = 0
    while max_frames == 0 or frame_index < max_frames:
        if stop_event.is_set():  # the restart command, or the --restart-after timer
            print("stop requested - leaving the frame loop")
            break
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
        # Only what this thread ran, in order: the braced figure on the HUD is a *sequential* budget,
        # so it can never be worse than the end-to-end rate beside it. The face pass is deliberately
        # not in it - it is ~66 ms every 200 ms on its own thread and a spare core, concurrent with
        # this loop rather than behind it, so adding it in claimed 47 ms of work per 33 ms frame.
        # If face recognition ever moves onto this thread, it belongs here and nowhere else.
        recent_ms.append((stage_ms["inference"], sum(stage_ms.values())))
        mean_inference = sum(inference for inference, _ in recent_ms) / len(recent_ms)
        mean_total = sum(total for _, total in recent_ms) / len(recent_ms)
        overlay.set_timing(mean_inference, mean_total)
        if frame_index % FPS_REPORT_FRAMES == 0:  # twice a second is plenty for a dashboard
            telemetry.set(fps=round(1000 / max(mean_total, 0.001), 1))
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
    vision = parser.add_argument_group("vision")
    vision.add_argument("--model", default="models/yolo11n_neutron.tflite",
                        help="path to the YOLO .tflite model")
    vision.add_argument("--delegate", action="store_true",
                        help="force the Neutron NPU delegate (auto-on when the model name contains 'neutron')")
    vision.add_argument("--yunet", default="models/face_detection_yunet_2023mar.onnx",
                        help="YuNet ONNX (detect)")
    vision.add_argument("--sface", default="models/face_recognition_sface_2021dec.onnx",
                        help="SFace ONNX (align only)")
    vision.add_argument("--sface-model", default="",
                        help="SFace embedder tflite (default: models/sface_neutron.tflite with "
                             "--delegate, else models/sface_int8.tflite)")
    vision.add_argument("--face-interval", type=float, default=0.2,
                        help="min seconds between async face-recognition passes (0.2 = ~5 Hz)")
    vision.add_argument("--device", default="",
                        help="v4l2 camera device (default: the first USB camera v4l2-ctl reports)")
    vision.add_argument("--width", type=int, default=640, help="camera capture width")
    vision.add_argument("--height", type=int, default=480, help="camera capture height")
    vision.add_argument("--conf", type=float, default=0.25, help="minimum YOLO score to keep a detection")
    vision.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    vision.add_argument("--threads", type=int, default=0, help="tflite CPU threads (0 = auto: 6 CPU / 2 NPU)")
    vision.add_argument("--no-preview", action="store_true", help="skip the Wayland preview window")
    vision.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = until Ctrl-C)")
    # Off by default: a per-frame timing line and a per-cycle face line drown out
    # everything worth reading -- and with the cloud connected, `--iotc-verbose` is the log you
    # actually want. The steady-state summary still prints on exit, so the numbers are not lost.
    vision.add_argument("--log-every", type=int, default=0,
                        help="print one frame line every N frames (0 = never)")

    voice = parser.add_argument_group("voice")
    voice.add_argument("--no-voice", action="store_true", help="run the video loop only, no eIQ payload")
    voice.add_argument("--no-tts", action="store_true",
                       help="listen, but never speak: no synthesis, no playback of answers. With no "
                            "speaker plugged in, the wake word and every command appear delayed by the "
                            "audio nobody can hear; this makes the screen the answer instead")
    voice.add_argument("--mic",
                       help="capture device, overriding audio.json ('auto', alias, substring, or ALSA name)")
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
    control.add_argument("--capture", default="capture.jpg",
                         help="the single JPEG the 'snapshot' command writes and overwrites")
    control.add_argument("--no-vlm", action="store_true", help="disable 'describe scene'")
    control.add_argument("--preload-vlm", action="store_true",
                         help="load SmolVLM at startup (~12 s) instead of on the first 'describe scene'")
    control.add_argument("--prefetch", action="store_true",
                         help="download and decrypt SmolVLM, then exit -- run once on a network you trust")
    control.add_argument("--vlm-tokens", type=int, default=64, help="longest scene description, in tokens")
    control.add_argument("--text-commands", default="",
                         help="debug: also read commands from this file, one per line (e.g. command.txt)")

    cloud = parser.add_argument_group("/IOTCONNECT")
    cloud.add_argument("--no-iotc", action="store_true",
                       help="do not connect to /IOTCONNECT; without this a failure to connect is fatal")
    cloud.add_argument("--iotc-config", default="iotcDeviceConfig.json",
                       help="the device config downloaded from the device's info panel")
    cloud.add_argument("--iotc-cert", default="device-cert.pem",
                       help="device certificate, as downloaded with the duid dropped")
    cloud.add_argument("--iotc-key", default="device-pkey.pem", help="device private key, likewise")
    cloud.add_argument("--iotc-interval", type=float, default=iotc.TELEMETRY_INTERVAL_S,
                       help="seconds between telemetry messages (an event still sends at once)")
    cloud.add_argument("--iotc-verbose", action="store_true",
                       help="log every MQTT packet the SDK sends and receives")
    cloud.add_argument("--restart-after", type=float, default=0.0,
                       help="restart the process after N minutes (0 = never); eIQ stops us at 60")

    stream = parser.add_argument_group("KVS WebRTC")
    stream.add_argument("--no-webrtc", action="store_true",
                        help="do not stream the display (the rest of /IOTCONNECT is unaffected)")

    llm = parser.add_argument_group("the LLM on the Ara-240")
    llm.add_argument("--no-agent", action="store_true",
                     help="do not use the LLM; the 'agent' command then refuses politely")
    llm.add_argument("--ara-url", default=agent.ARA_URL,
                     help="the eIQ AAF Connector's OpenAI endpoint (see connector/README.md)")
    llm.add_argument("--ara-model", default=agent.ARA_MODEL,
                     help="which model the connector should answer with")
    llm.add_argument("--agent-tokens", type=int, default=agent.MAX_TOKENS,
                     help="longest answer, in tokens (~5 tokens/second, so this is also a time limit)")
    return parser.parse_args()


def restart_process() -> None:
    """Replace this process with a fresh copy of itself: same interpreter, arguments and directory.

    eIQ's models stop working an hour after they load and a stand cannot rely on someone noticing;
    everything a visitor set is on disk, so coming back costs the ~15 s of loading. exec rather than
    fork means one pid and no orphan still holding the camera, which is how a restart loses it. The
    sleep gives GStreamer's teardown and the audio device a moment before they are reopened.
    """
    sleep(1.0)
    print(f"[main] restarting: {sys.executable} {' '.join(sys.argv)}", flush=True)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def call_later(seconds: float, action) -> None:
    """Run `action` on a daemon timer thread. Daemon so a pending timer cannot hold up Ctrl-C."""
    timer = Timer(seconds, action)
    timer.daemon = True
    timer.start()


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
    # only oversubscribe the camera/preview/face/voice threads, so the count follows the backend.
    num_threads = args.threads if args.threads > 0 else (2 if use_neutron else 6)

    detector = Detector(args.model, use_neutron, num_threads, args.conf, args.iou)
    print(detector.describe())
    # The face embedder gets the same rule as the detector: the backend follows the *model name*,
    # not the other backend. Naming a CPU embedder with --sface-model used to load it behind the
    # Neutron delegate anyway and then report it as running on the NPU - which delegates nothing
    # (0 of 172 nodes) and so was merely a lie on the startup line, but it is the line somebody
    # reads while working out whether the face path can be trusted.
    sface_model = args.sface_model or str(
        MODELS / ("sface_neutron.tflite" if use_neutron else "sface_int8.tflite"))
    use_neutron_face = "neutron" in Path(sface_model).name
    print(f"Face embed : {sface_model} ({'Neutron NPU' if use_neutron_face else 'CPU'})")
    face = FaceRecognizer(args.yunet, args.sface, sface_model, use_neutron=use_neutron_face)
    registry = Registry(args.db)
    face_worker = FaceWorker(face, registry, args.face_interval)
    tracker = Tracker()
    overlay = Overlay(args.width, args.height)
    state = SessionState(args.state)
    # The watchdog is the alarm's memory: what has happened (the event log) and what has been
    # disturbed (the object guard, which owns the locked objects' anchors). `app.py` reads three
    # flags off it every frame and never the other way round.
    watchdog = Watchdog(registry, state, ObjectGuard(state), EventLog())
    vocab = vocabulary.load_vocabulary()
    telemetry = TelemetryState()
    scene = None if args.no_vlm else SceneDescriber(
        PAYLOAD, VLM_WEIGHTS, Path(args.keyframe), max_new_tokens=args.vlm_tokens)

    # The restart command and the --restart-after timer both end here: raise both flags, let the
    # frame loop finish the frame it is on, shut everything down in order, then exec ourselves
    # again. Two flags rather than one, so that a Ctrl-C during those three seconds still means stop.
    stop_event = Event()
    restart_event = Event()

    def request_restart() -> None:
        print(f"[main] restart requested - stopping in {RESTART_DELAY_S:.0f}s")
        restart_event.set()
        call_later(RESTART_DELAY_S, stop_event.set)

    # The camera has to exist before the app, because the app takes snapshots off the composed
    # display feed (`read_display`) rather than re-drawing the boxes itself - see app._snapshot.
    # It also comes before the streamer, which needs `request_keyframe` to hand to each viewer.
    is_streaming = not args.no_webrtc and webrtc.IS_WEBRTC_AVAILABLE
    camera = Camera(resolve_camera_device(args), args.width, args.height,
                    show_preview=resolve_preview(args), is_streaming=is_streaming)
    streamer = build_streamer(args, overlay, camera) if is_streaming else None
    if streamer is None:
        overlay.set_stream_status("stream: off")
        if not webrtc.IS_WEBRTC_AVAILABLE and not args.no_webrtc:
            print("WebRTC    : disabled (aiortc is not installed)")
    app = AntiTheftApp(registry, overlay, face_worker, state, watchdog, vocab, telemetry, scene,
                       Path(args.capture), on_restart=request_restart,
                       get_display_frame=camera.read_display)
    # The first of two back-edges in the object graph, and the reason it exists is worth a line: the
    # LLM's tools *are* the command handlers, so the agent cannot be built before the app that owns
    # them. (The second is `set_upload_capture` below, for the same kind of reason.)
    app.set_agent(build_agent(args, app, overlay))
    service = CommandService(app.on_command, max_workers=args.command_threads,
                             on_result=lambda result: show_action(overlay, result))
    # A recording's screenshots are ordinary `snapshot` commands, submitted from the video loop and
    # run on a command worker - so the ~100 KB upload never touches the frame loop, and a picture
    # the watchdog took is the same thing as one anybody else asked for.
    watchdog.set_capture(lambda: service.submit_command(
        Command(commands.SNAPSHOT, text="recording"), source="watchdog"))
    text_commands = TextCommandFile(args.text_commands) if args.text_commands else None
    if args.restart_after > 0:
        print(f"Restart    : automatically after {args.restart_after:.0f} minutes")
        call_later(args.restart_after * 60, request_restart)

    voice = None
    if not args.no_voice:
        audio_config = resolve_audio_config(args)
        voice = VoiceLoop(
            PAYLOAD, audio_config,
            on_command=lambda text: service.submit(text, source="voice").result().message,
            on_status=overlay.set_voice_status, speaker_id=args.tts_voice,
            wake_timeout_s=args.wake_timeout, command_seconds=args.command_seconds,
            is_repeating=not args.no_repeat, is_speaking=not args.no_tts,
        )
    else:
        overlay.set_voice_status("voice: off")

    # A photograph dropped into the device's S3 folder registers whoever is named in its file name.
    # It has no bucket yet - `iotc.py` starts it once the cloud names one.
    face_uploads = FaceUploads(app.on_face_image, FACES)
    cloud = build_iotc_client(args, service, telemetry, overlay, streamer, face_uploads)
    if cloud is not None:
        # The other back-edge: a snapshot is saved by the app and uploaded by the cloud, and the
        # cloud cannot exist until the command service does. Handing over one bound method keeps
        # `app.py` free of S3 and keeps "take a screenshot" one command rather than two.
        app.set_upload_capture(cloud.upload_capture)

    print(f"Known users: {', '.join(registry.user_names) or '(none)'}")
    print(f"Alarm      : {app.alarm_state.label}   locked: {', '.join(state.locked_objects) or '-'}")
    # Late, and loud when it fires: converter, delegate and firmware must be one release, and when
    # they are not the models still load and still run - they just return nonsense. See neutron.py.
    mismatch = neutron.check_models(MODELS)
    if mismatch is not None:
        print("=" * 78)
        print(f"WARNING: {mismatch}")
        print("=" * 78)
    if scene is not None and args.preload_vlm:
        scene.load()

    # Before the camera and the models, and on this thread: the cloud is required, so a failure to
    # connect must stop the demo here - with nothing opened yet, and with the SDK's own error as
    # the last thing printed. Run with --no-iotc to skip it on purpose.
    if cloud is not None:
        cloud.connect()
    print("-" * 60)

    camera.connect_overlay(overlay.draw)
    if streamer is not None:
        camera.connect_encoded(streamer.on_encoded)  # where finished H.264 goes; the only back-edge
    camera.start()
    if voice is not None:
        voice.start()  # loads eIQ on its own thread; the video does not wait for it
    if cloud is not None:
        cloud.start()  # already connected above; this is the publisher thread
    try:
        run_loop(camera, detector, tracker, face_worker, overlay, app, service, telemetry,
                 text_commands, args.max_frames, args.log_every, stop_event)
    except KeyboardInterrupt:
        print("\ninterrupted - stopping")
        restart_event.clear()  # Ctrl-C means stop, never come back
    finally:
        if cloud is not None:
            cloud.stop()
        face_uploads.stop()
        if streamer is not None:
            streamer.stop()
        if voice is not None:
            voice.stop()
        service.stop()
        face_worker.stop()
        camera.stop()

    if restart_event.is_set():
        restart_process()


def show_action(overlay: Overlay, result: CommandResult) -> None:
    """Flash a finished command's answer on the screen - and on the stream, which is the same pixels.

    Until now the only acknowledgement a command had was spoken, so a visitor at a noisy booth, or
    anyone driving the demo from the dashboard, had to infer from the HUD whether anything had
    happened. Two seconds of "Registered Nick." or "I cannot see a laptop." is that answer.

    Two kinds of result are deliberately not shown, and deciding that is the whole reason this is a
    function rather than a bound method:

    **The watchdog's own commands.** A recording takes a screenshot every three seconds
    (`watchdog.set_capture` below), and "Snapshot uploaded." blinking on the display every three
    seconds says nothing the REC mark in the corner is not already saying, while covering the
    screen precisely when something interesting is happening in front of it.

    **`agent` and `describe scene`.** Both are prose, and both arrive tens of seconds after anybody
    asked for anything - the LLM at ~5 tokens/second, the VLM not much better. A flash on the
    screen is read as *"this just happened"*, so a paragraph appearing half a minute late attaches
    itself to whatever is in front of the camera by then, and two seconds is not long enough to
    read it anyway. Those two answers already have the channels that suit them: voice speaks them
    in full, and the dashboard gets the whole thing as `answer`. Everything left here is a short
    confirmation of something the demo did.
    """
    if result.source == "watchdog":
        return
    if result.command is not None and result.command.verb in SLOW_ANSWER_VERBS:
        return
    overlay.show_action(result.message, result.is_ok)


def build_streamer(args, overlay: Overlay, camera: Camera) -> WebRtcStreamer:
    """The WebRTC streamer. It has no channel yet - `iotc.py` starts it once the cloud names one.

    `camera.request_keyframe` is the one thing it needs from the video side: aiortc cannot ask a
    hardware encoder for a keyframe by itself, so a viewer that has just connected (or fallen
    behind) gets an IDR because this callback goes back into GStreamer. See `webrtc.py`.
    """
    print("WebRTC    : hardware H.264, waiting for the channel ARN")
    return WebRtcStreamer(on_keyframe_wanted=camera.request_keyframe,
                          on_status=overlay.set_stream_status)


def build_agent(args, app: AntiTheftApp, overlay: Overlay) -> AgentService | None:
    """The LLM on the Ara-240, or None and a reason on the HUD. Never a reason not to run.

    Nothing is checked or connected here: the connector is a separate process with its own venv and
    its own ~4-minute model load, and a demo that waited for it (or refused to start without it)
    would be a worse demo. The first question is what finds out, and it says so if the endpoint is
    not there. `connector/README.md` is how to bring it up.
    """
    if args.no_agent:
        overlay.set_agent_status("agent: off")
        return None
    if not agent.IS_STRANDS_AVAILABLE:
        print("Ara LLM   : disabled (strands-agents is not installed)")
        overlay.set_agent_status("agent: unavailable")
        return None
    print(f"Ara LLM   : {args.ara_model} at {args.ara_url}")
    return AgentService(app.on_command, app.get_status, base_url=args.ara_url,
                        model_id=args.ara_model, max_tokens=args.agent_tokens,
                        on_status=overlay.set_agent_status)


def build_iotc_client(args, service: CommandService, telemetry: TelemetryState, overlay: Overlay,
                      streamer: WebRtcStreamer | None,
                      face_uploads: FaceUploads) -> IotcClient | None:
    """The /IOTCONNECT client, or None when `--no-iotc` says so. Nothing else makes it optional.

    Deliberately unguarded: a missing config, an unreadable key or a device the back end does not
    know all reach `connect()` and stop the demo there, with the SDK's own message. Checking for
    those here would only produce a worse-worded version of the same error. Note that `--no-iotc`
    takes the video stream with it: no cloud means no channel ARN, so there is nothing to stream to.
    """
    if args.no_iotc:
        overlay.set_cloud_status("cloud: off")
        return None
    if not iotc.IS_SDK_AVAILABLE:
        raise SystemExit("/IOTCONNECT: the SDK is not installed. Run ./install.sh, or --no-iotc.")
    client = IotcClient(
        Path(args.iotc_config), service, telemetry, capture_path=Path(args.capture),
        app_version=VERSION, cert_path=Path(args.iotc_cert), key_path=Path(args.iotc_key),
        streaming=streamer, face_uploads=face_uploads,
        on_status=overlay.set_cloud_status, on_stream_status=overlay.set_stream_status,
        interval_s=args.iotc_interval, is_verbose=args.iotc_verbose,
    )
    print(f"/IOTCONNECT: {args.iotc_config}, certificate {client.cert_path.name}, "
          f"telemetry every {args.iotc_interval:.0f}s")
    return client


def resolve_camera_device(args) -> str:
    """`--device` if it was given, otherwise whichever `/dev/videoN` the USB camera is on today.

    The reason it is not a fixed default is in `camera_devices.py`: the board's own IP blocks take fifty
    video nodes before the camera gets one, and how many they take is a property of the BSP. A
    hard-coded default is a demo that fails on the next image with a message about the camera
    returning no frames. No camera at all is fatal here rather than later, and says what it saw.
    """
    if args.device:
        print(f"Camera     : {args.device} (given on the command line)")
        return args.device
    found = camera_devices.find_camera_device()
    if found is None:
        raise SystemExit("No USB camera found. v4l2-ctl --list-devices reports:\n"
                         f"{camera_devices.describe_devices()}\n"
                         "Plug the camera in, or name a node with --device.")
    node, device = found
    print(f"Camera     : {node}  {device}")
    return node


def resolve_preview(args) -> bool:
    """Show the picture on HDMI - unless there is nothing to show it on, which is not fatal.

    A monitor is the one part of this demo that is routinely absent: the board gets carried to a
    booth, a desk or another room, and Weston exits at boot when no HDMI is connected. The preview
    is then the only branch of the pipeline that cannot be built, and it does not fail on its own -
    it stops the *whole* pipeline, so the demo dies reporting `camera returned no frame`. Dropping
    it costs the local screen and nothing else: the OSD is composed upstream, so WebRTC viewers and
    every snapshot still get exactly the picture the monitor would have shown.

    Plugging HDMI in later needs `systemctl restart weston` before the demo starts, because systemd
    gives up after five failed attempts.
    """
    if args.no_preview:
        return False
    if find_wayland_socket() is not None:
        return True
    print("Display    : no Wayland compositor - preview off (HDMI unplugged?). "
          "Streaming and snapshots are unaffected.")
    return False


def resolve_audio_config(args) -> audio_devices.AudioConfig:
    """audio.json holds the defaults; anything given on the command line wins.

    It ships with `"capture_device": "auto"`, which takes the microphone out of whatever is plugged
    into USB - normally the webcam's, right in front of the person talking - and falls back to the
    board's own `micfil` when nothing is. So the line printed here is worth reading: a demo that
    cannot hear anybody and a demo listening to a microphone soldered to the board look identical
    until you know which device it opened. `audio_devices.py` explains the rule.
    """
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
    print(f"Audio      : mic {config.capture_device}, speaker {config.playback_device}")
    return config


if __name__ == "__main__":
    main()
