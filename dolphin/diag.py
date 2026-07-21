#!/usr/bin/env python3
"""One-shot diagnostic: is the int8 model producing usable scores, or collapsed ones?

Grabs a single live frame, runs the model, and prints the output quantization plus the
raw coordinate and score ranges and the top-5 class scores. If the best score is tiny
(~0), the int8 output scale is dominated by pixel-space box coords and scores collapsed.
"""

from __future__ import annotations

import sys

import numpy as np
from tflite_runtime.interpreter import Interpreter

from camera import Camera
from yolo import COCO_CLASSES, dequantize_output, letterbox, quantize_input

interpreter = Interpreter(model_path=sys.argv[1])
interpreter.allocate_tensors()
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]
print("output dtype/quant:", output_detail["dtype"].__name__, output_detail["quantization"])

camera = Camera(sys.argv[2] if len(sys.argv) > 2 else "/dev/video4", 640, 480, show_preview=False)
camera.start()
frame = camera.read()
camera.stop()

size = int(input_detail["shape"][1])
padded, _, _, _ = letterbox(frame, size)
input_tensor = quantize_input(padded, input_detail)
if list(input_detail["shape"]) == [1, 3, size, size]:  # litert models are NCHW, onnx2tf are NHWC
    input_tensor = input_tensor.transpose(0, 3, 1, 2)
interpreter.set_tensor(input_detail["index"], input_tensor)
interpreter.invoke()
raw = interpreter.get_tensor(output_detail["index"])
output = np.squeeze(dequantize_output(raw, output_detail))
if output.shape[0] < output.shape[1]:
    output = output.T  # (num_boxes, 84)

coords, scores = output[:, :4], output[:, 4:]
print("coord range:", float(coords.min()), "..", float(coords.max()))
print("score range:", float(scores.min()), "..", float(scores.max()))

best_per_box = scores.max(axis=1)
top = best_per_box.argsort()[::-1][:5]
for rank, box_index in enumerate(top):
    class_id = int(scores[box_index].argmax())
    print(f"top{rank}: {COCO_CLASSES[class_id]:15s} score={best_per_box[box_index]:.3f} xywh={coords[box_index]}")
