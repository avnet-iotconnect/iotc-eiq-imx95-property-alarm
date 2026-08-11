"""The YOLO object detector as one object: frame in, detections out.

This logic would otherwise sit inline in the frame loop. Wrapped in a `Detector` class instead,
`main.py` builds it once and just calls `detector.detect(frame)` - the same shape as
`FaceRecognizer` in `face.py`. Both are the "ML" layer: the interpreter + the
pre/post math (which stays in `yolo.py`) hidden behind one narrow method.

The Neutron NPU runs the conv backbone; the DFL box-decode is done in float on the CPU (see yolo.py).
Nothing here knows about tracking, faces, or the alarm - it only turns pixels into raw detections.
"""

from __future__ import annotations

from tflite_runtime.interpreter import Interpreter, load_delegate

from applib import neutron
from applib.yolo import (
    COCO_CLASSES,
    decode_detections,
    dequantize_output,
    letterbox,
    map_boxes_to_frame,
    quantize_input,
)

Detection = tuple[str, float, list[int]]  # (class_name, score, box_xyxy) - what the tracker consumes


class Detector:
    """Loads a YOLO .tflite (optionally behind the Neutron delegate) and decodes frames to detections."""

    def __init__(
        self, model_path: str, use_neutron: bool, num_threads: int, conf: float = 0.25, iou: float = 0.45
    ) -> None:
        delegates = [load_delegate(neutron.DELEGATE_PATH)] if use_neutron else []
        self.interpreter = Interpreter(
            model_path=model_path, experimental_delegates=delegates, num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        self.on_neutron = use_neutron
        self.conf = conf
        self.iou = iou
        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]
        self.input_size = int(self._input["shape"][1])  # model input is square: (1, size, size, 3)

    def describe(self) -> str:
        backend = "Neutron NPU (delegate)" if self.on_neutron else "CPU (tflite reference / XNNPACK)"
        return (
            f"Backend    : {backend}\n"
            f"Input      : {self._input['shape']} {self._input['dtype'].__name__}\n"
            f"Output     : {self._output['shape']} {self._output['dtype'].__name__}"
        )

    def detect(self, frame) -> list[Detection]:
        """Run one frame through letterbox -> quantize -> NPU -> float decode -> NMS -> frame-space boxes."""
        padded, scale, pad_x, pad_y = letterbox(frame, self.input_size)
        input_tensor = quantize_input(padded, self._input)

        self.interpreter.set_tensor(self._input["index"], input_tensor)
        self.interpreter.invoke()
        raw_output = dequantize_output(self.interpreter.get_tensor(self._output["index"]), self._output)

        boxes, scores, class_ids = decode_detections(raw_output, self.input_size, self.conf, self.iou)
        boxes = map_boxes_to_frame(boxes, scale, pad_x, pad_y, frame.shape[1], frame.shape[0])
        return [
            (COCO_CLASSES[class_id], float(score), [int(v) for v in box])
            for box, score, class_id in zip(boxes, scores, class_ids)
        ]
