#!/usr/bin/env python3
"""Live YOLO11n object detection on the FRDM-IMX95 - CPU vs Neutron NPU, same code path.

The point of this demo is to *see and measure* YOLO running on a live camera two ways:

    python3 detect.py --model yolo11n_int8.tflite                 # runs on the 6x Cortex-A55 CPU
    python3 detect.py --model yolo11n_neutron.tflite --delegate   # runs on the Neutron NPU

Only the two arguments differ; everything below is identical, which is exactly what makes
the CPU-vs-NPU comparison fair. Each frame goes through four stages and we time all four:

    capture  ->  preprocess  ->  inference  ->  postprocess

'inference' is the only stage that moves to the NPU; watch it drop when you add --delegate.
The GStreamer preview window shows the live camera on the HDMI display in parallel.
"""

from __future__ import annotations

import argparse
from time import perf_counter

from tflite_runtime.interpreter import Interpreter, load_delegate

from camera import Camera
from overlay import OverlayState, draw_overlay
from tracking import Track, Tracker
from yolo import (
    COCO_CLASSES,
    decode_detections,
    dequantize_output,
    letterbox,
    map_boxes_to_frame,
    quantize_input,
)

NEUTRON_DELEGATE_PATH = "/usr/lib/libneutron_delegate.so"


def build_interpreter(model_path: str, use_neutron: bool, num_threads: int) -> Interpreter:
    """Load the model, optionally behind the Neutron delegate.

    `num_threads` sets how many CPU cores the tflite runtime may use - it makes the CPU path a
    fair baseline (all 6 Cortex-A55 cores, not one). On the NPU path the delegate runs the heavy
    ops and prints a line like `N nodes delegated out of M ... K partitions` to stderr - that line
    is the proof the NPU actually ran the model (0/M would mean a silent fall back to CPU).
    """
    delegates = [load_delegate(NEUTRON_DELEGATE_PATH)] if use_neutron else []
    interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates, num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter


def describe_model(interpreter: Interpreter, on_neutron: bool) -> None:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    backend = "Neutron NPU (delegate)" if on_neutron else "CPU (tflite reference / XNNPACK)"
    print(f"Backend    : {backend}")
    print(f"Input      : {input_detail['shape']} {input_detail['dtype'].__name__}")
    print(f"Output     : {output_detail['shape']} {output_detail['dtype'].__name__}")
    print("-" * 60)


def report_frame(index: int, tracks: list[Track], stage_ms: dict[str, float]) -> None:
    labels = ", ".join(f"#{t.short_id or '-'} {t.class_name} {t.score:.2f}" for t in tracks) or "(nothing)"
    timing = "  ".join(f"{stage}={ms:5.1f}ms" for stage, ms in stage_ms.items())
    total = sum(stage_ms.values())
    print(f"frame {index:04d} | {timing}  total={total:5.1f}ms ({1000 / total:4.1f} fps) | {labels}")


def report_summary(steady_state: list[dict[str, float]]) -> None:
    if not steady_state:
        return
    print("-" * 60)
    stages = steady_state[0].keys()
    means = {stage: sum(frame[stage] for frame in steady_state) / len(steady_state) for stage in stages}
    total = sum(means.values())
    print(f"Steady-state mean over {len(steady_state)} frames (warm-up frame 0 excluded):")
    for stage, ms in means.items():
        print(f"  {stage:11s}: {ms:6.2f} ms")
    print(f"  {'TOTAL':11s}: {total:6.2f} ms  ->  {1000 / total:.1f} fps end-to-end")


def run_detection_loop(
    interpreter: Interpreter, camera: Camera, tracker: Tracker, overlay_state: OverlayState,
    args: argparse.Namespace,
) -> None:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_size = int(input_detail["shape"][1])  # model input is square: shape is (1, size, size, 3)

    steady_state: list[dict[str, float]] = []
    frame_index = 0
    while args.max_frames == 0 or frame_index < args.max_frames:
        start = perf_counter()
        frame = camera.read()
        if frame is None:
            print("camera returned no frame (timeout/EOS) - stopping")
            break
        after_capture = perf_counter()

        padded, scale, pad_x, pad_y = letterbox(frame, input_size)
        input_tensor = quantize_input(padded, input_detail)
        after_preprocess = perf_counter()

        interpreter.set_tensor(input_detail["index"], input_tensor)
        interpreter.invoke()
        raw_output = dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)
        after_inference = perf_counter()

        boxes, scores, class_ids = decode_detections(raw_output, input_size, args.conf, args.iou)
        boxes = map_boxes_to_frame(boxes, scale, pad_x, pad_y, frame.shape[1], frame.shape[0])
        detections = [
            (COCO_CLASSES[class_id], float(score), [int(v) for v in box])
            for box, score, class_id in zip(boxes, scores, class_ids)
        ]
        tracks = tracker.update(detections)  # stateless detections -> persistent IDs (part of postprocess)
        after_postprocess = perf_counter()

        stage_ms = {
            "capture": (after_capture - start) * 1000,
            "preprocess": (after_preprocess - after_capture) * 1000,
            "inference": (after_inference - after_preprocess) * 1000,
            "postprocess": (after_postprocess - after_inference) * 1000,
        }
        overlay_state.tracks = tracks  # atomic reference swap; the preview draws the latest
        overlay_state.inference_ms = stage_ms["inference"]
        overlay_state.end_to_end_ms = sum(stage_ms.values())

        report_frame(frame_index, tracks, stage_ms)
        if frame_index >= 1:  # frame 0 is warm-up; steady-state numbers start at frame 1
            steady_state.append(stage_ms)
        frame_index += 1

    report_summary(steady_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to the .tflite model")
    parser.add_argument("--delegate", action="store_true", help="run through the Neutron NPU delegate")
    parser.add_argument("--device", default="/dev/video4", help="v4l2 camera device (C920 capture node)")
    parser.add_argument("--width", type=int, default=640, help="camera capture width (native YUY2 size)")
    parser.add_argument("--height", type=int, default=480, help="camera capture height")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum score to keep a detection")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU threads for tflite (0 = auto: 6 for CPU path, 2 for NPU path to avoid oversubscription)",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until Ctrl-C)")
    parser.add_argument("--no-preview", action="store_true", help="skip the Wayland preview window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The CPU path parallelizes convolutions across all cores; the NPU path runs convs on the NPU, so
    # extra CPU threads only oversubscribe the camera/preview threads and add latency spikes. Auto-pick.
    num_threads = args.threads if args.threads > 0 else (2 if args.delegate else 6)
    interpreter = build_interpreter(args.model, args.delegate, num_threads)
    describe_model(interpreter, args.delegate)

    tracker = Tracker()
    overlay_state = OverlayState(backend="NPU" if args.delegate else "CPU")
    camera = Camera(args.device, args.width, args.height, show_preview=not args.no_preview)
    camera.connect_overlay(lambda context: draw_overlay(context, overlay_state))
    camera.start()
    try:
        run_detection_loop(interpreter, camera, tracker, overlay_state, args)
    except KeyboardInterrupt:
        print("\ninterrupted - stopping")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
