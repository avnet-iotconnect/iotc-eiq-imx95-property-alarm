#!/usr/bin/env python3
"""Export YOLO11n as a Neutron-convertible int8 tflite with the box decode left OUT of the graph.

Stock `yolo export ... int8` bakes the YOLO box decode (DFL softmax + anchor arithmetic) into the
quantized graph. That decode does not survive int8: the box coordinates collapse to ~0 (while class
scores are fine). This is the well-known YOLO-on-NPU problem, and the standard fix is to NOT quantize
the decode: let the NPU run the conv backbone (which quantizes beautifully) and emit the head's RAW
tensors, then do the cheap DFL/anchor/sigmoid decode in float on the CPU (see src/applib/yolo.py).

So we patch the Detect head to output its raw pre-decode tensors, concatenated as
(batch, 4*reg_max + num_classes, num_boxes) = (1, 64 + 80, 2100) for yolo11n at imgsz 320:
the first 64 channels are the DFL box distribution logits, the last 80 are class logits.
(YOLO11 shares YOLOv8's Detect head, so this patch and the decoder are identical for both.)

Run from repo root:  .venv/bin/python scripts/yolo-export-prep.py
"""

from __future__ import annotations

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

MODEL = "yolo11n"
IMGSZ = 320


def _inference_raw_head(self: Detect, x: dict) -> torch.Tensor:
    """Emit the head's raw conv outputs (box distribution + class logits), undecoded."""
    return torch.cat((x["boxes"], x["scores"]), dim=1)  # (batch, 4*reg_max + num_classes, num_boxes)


Detect._inference = _inference_raw_head

YOLO(f"{MODEL}.pt").export(format="saved_model", int8=True, imgsz=IMGSZ, data="coco8.yaml")
print(f"done -> {MODEL}_saved_model/{MODEL}_full_integer_quant.tflite (raw head, box decode on CPU)")
