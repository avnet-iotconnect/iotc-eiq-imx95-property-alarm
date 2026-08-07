#!/usr/bin/env python3
"""Quantize SFace to int8, the input the Neutron converter needs. Host-side, called by
scripts/package-models.sh.

    .venv/bin/python scripts/sface-quantize.py

ONNX -> saved_model (onnx2tf) -> int8 tflite (TFLiteConverter), calibrated on the aligned face chips
committed in files/sface-calib/. Three details are load-bearing:

- **The representative data is RGB 0-255**, not ImageNet-normalized: SFace's ONNX takes raw pixel
  values, so the int8 input maps straight through (scale 1.0, zero -128) and `applib/face.py` can
  feed it without a normalization step.
- **Real faces are required.** Synthetic calibration was measured and rejected: uniform noise costs
  ~0.08 of agreement with the float model, and blurred noise collapses it to 0.33-0.53 while putting
  two *different* people at 0.525 - past the 0.363 match threshold. Quantization calibration alone
  can manufacture the "everyone matches everyone" bug.
- **Gain/bias variants**, not more identities. Ten chips at three gains and three biases calibrates
  as well as the 81-chip set the first model used (agreement 0.992/0.985/0.985 against float, on
  faces that are not in the calibration set at all).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHIPS = ROOT / "files" / "sface-calib"
MODELS = ROOT / "work" / "models"
WORK_DIR = ROOT / "work" / "sface_onnx2tf"
GAINS, BIASES = (0.7, 1.0, 1.3), (-30, 0, 30)


def representative_chips() -> np.ndarray:
    chips = [cv2.imread(str(path)) for path in sorted(CHIPS.glob("*.png"))]
    if not chips:
        sys.exit(f"No calibration chips in {CHIPS} - see its README.")
    return np.stack([np.clip(chip[:, :, ::-1].astype(np.float32) * gain + bias, 0, 255)
                     for chip in chips for gain in GAINS for bias in BIASES])


def main() -> None:
    import onnx2tf

    samples = representative_chips()
    print(f"calibrating on {len(samples)} samples from {len(list(CHIPS.glob('*.png')))} chips")
    import tensorflow as tf

    # `output_signaturedefs` is what makes onnx2tf leave a saved_model TFLiteConverter can read; by
    # default it writes float tflite only. Its *own* int8 path is not used - its strict validator
    # aborts the whole run over an int16-activation variant nobody asked for.
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    onnx2tf.convert(
        input_onnx_file_path=str(MODELS / "face_recognition_sface_2021dec.onnx"),
        output_folder_path=str(WORK_DIR),
        output_signaturedefs=True,
        disable_strict_mode=True,
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(str(WORK_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [sample[np.newaxis, ...].astype(np.float32)] for sample in samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    out = MODELS / "sface_int8.tflite"
    out.write_bytes(converter.convert())
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
