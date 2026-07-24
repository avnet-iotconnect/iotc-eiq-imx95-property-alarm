#!/usr/bin/env python3
"""elephant pilot - live face recognition + anti-theft alarm on the FRDM-IMX95.

Builds on the dolphin pilot (camera -> YOLO11n on Neutron -> tracking -> overlay) and adds the face
path: each `person` track is run through YuNet + SFace on the CPU to recognize registered users, and a
simple alarm state machine trips when an unrecognized person is on screen.

    python3 main.py --model yolo11n_neutron.tflite --delegate     # NPU for YOLO (faces always on CPU)
    python3 main.py --model yolo11n_int8.tflite                   # full-CPU baseline

Register a user by writing one line into command.txt (default: ./command.txt):
    echo 'register user Joe' > command.txt      # binds Joe to the biggest visible face
    echo 'disarm'            > command.txt       # arm | disarm the alarm

This file is the *composition root*: the only place modules are constructed and wired.
Read `main()` top to bottom to see the whole object graph; read `app.py` to see every reaction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from app import AntiTheftApp
from camera import Camera
from detector import Detector
from face import FaceRecognizer
from overlay import Overlay
from registry import Registry
from tracking import Tracker


class CommandFile:
    """Watches command.txt: returns any new lines, then truncates so each command runs exactly once."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def poll(self) -> list[str]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        lines = [line for line in self.path.read_text().splitlines() if line.strip()]
        self.path.write_text("")  # consume: a command fires once, not every frame until edited
        return lines


def run_loop(
    camera: Camera, detector: Detector, tracker: Tracker, face: FaceRecognizer,
    registry: Registry, overlay: Overlay, app: AntiTheftApp, commands: CommandFile,
    max_frames: int,
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
        face.embed_faces(frame, tracks)   # CPU: YuNet + SFace on each person crop
        registry.identify(tracks)         # embedding -> registered name (fills Track.identity)
        after_face = perf_counter()

        for command in commands.poll():
            app.on_command(command)
        app.on_frame(tracks)

        stage_ms = {
            "capture": (after_capture - start) * 1000,
            "inference": (after_inference - after_capture) * 1000,
            "face+track": (after_face - after_inference) * 1000,
            "app": (perf_counter() - after_face) * 1000,
        }
        overlay.set_timing(stage_ms["inference"], sum(stage_ms.values()))
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
    parser.add_argument("--model", default="yolo11n_neutron.tflite", help="path to the YOLO .tflite model")
    parser.add_argument("--delegate", action="store_true",
                        help="force the Neutron NPU delegate (auto-on when the model name contains 'neutron')")
    parser.add_argument("--yunet", default="face_detection_yunet_2023mar.onnx", help="YuNet ONNX (detect)")
    parser.add_argument("--sface", default="face_recognition_sface_2021dec.onnx", help="SFace ONNX (align only)")
    parser.add_argument("--sface-model", default="",
                        help="SFace embedder tflite (default: sface_neutron.tflite with --delegate, else sface_int8.tflite)")
    parser.add_argument("--db", default="faces.json", help="persistent face database (name -> embedding)")
    parser.add_argument("--command-file", default="command.txt", help="file polled for register/arm/disarm")
    parser.add_argument("--device", default="/dev/video4", help="v4l2 camera device (C920 capture node)")
    parser.add_argument("--width", type=int, default=640, help="camera capture width")
    parser.add_argument("--height", type=int, default=480, help="camera capture height")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum YOLO score to keep a detection")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--threads", type=int, default=0, help="tflite CPU threads (0 = auto: 6 CPU / 2 NPU)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = until Ctrl-C)")
    parser.add_argument("--no-preview", action="store_true", help="skip the Wayland preview window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # A '..._neutron.tflite' model carries a NeutronGraph op only the delegate can run, so the backend
    # must follow the model - infer it from the name so `./run.sh` just works without remembering --delegate.
    use_neutron = args.delegate or "neutron" in Path(args.model).name
    # CPU path parallelizes convs across all cores; NPU path runs them on Neutron, so extra CPU threads
    # only oversubscribe the camera/preview/face threads. Auto-pick (same reasoning as dolphin).
    num_threads = args.threads if args.threads > 0 else (2 if use_neutron else 6)

    detector = Detector(args.model, use_neutron, num_threads, args.conf, args.iou)
    print(detector.describe())
    sface_model = args.sface_model or ("sface_neutron.tflite" if use_neutron else "sface_int8.tflite")
    print(f"Face embed : {sface_model} ({'Neutron NPU' if use_neutron else 'CPU'})")
    face = FaceRecognizer(args.yunet, args.sface, sface_model, use_neutron=use_neutron)
    registry = Registry(args.db)
    tracker = Tracker()
    overlay = Overlay(args.width, args.height, backend="NPU" if use_neutron else "CPU")
    app = AntiTheftApp(registry, overlay)
    commands = CommandFile(args.command_file)
    print(f"Known users: {', '.join(registry.user_names) or '(none - first person will trip the alarm)'}")
    print("-" * 60)

    camera = Camera(args.device, args.width, args.height, show_preview=not args.no_preview)
    camera.connect_overlay(overlay.draw)
    camera.start()
    try:
        run_loop(camera, detector, tracker, face, registry, overlay, app, commands, args.max_frames)
    except KeyboardInterrupt:
        print("\ninterrupted - stopping")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
