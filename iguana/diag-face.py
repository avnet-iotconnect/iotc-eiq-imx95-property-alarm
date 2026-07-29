#!/usr/bin/env python3
"""One-shot sanity check for the face path - run this FIRST on the board, before main.py.

The whole YuNet + SFace choice rests on one assumption: the board's system `cv2` is new enough to ship
`FaceDetectorYN` and `FaceRecognizerSF` (OpenCV >= 4.5.4). This script proves that assumption on-device
and that both ONNX models load and produce sane output, so a failure here is a clear message instead of
a confusing crash deep inside the frame loop.

    python3 diag-face.py face_detection_yunet_2023mar.onnx face_recognition_sface_2021dec.onnx
"""

from __future__ import annotations

import sys

import cv2
import numpy as np


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    yunet_path, sface_path = sys.argv[1], sys.argv[2]

    print(f"cv2 version : {cv2.__version__}")
    for attribute in ("FaceDetectorYN", "FaceRecognizerSF"):
        if not hasattr(cv2, attribute):
            print(f"FAIL: this cv2 build has no cv2.{attribute} - need OpenCV >= 4.5.4 in the BSP.")
            return 1
    print("cv2 has FaceDetectorYN + FaceRecognizerSF : OK")

    detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.7, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    print("both models loaded : OK")

    # Detector must run without error on a real-sized frame (a blank frame simply finds no faces).
    detector.setInputSize((320, 320))
    _, faces = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    print(f"detect() ran, faces found on blank frame: {0 if faces is None else len(faces)} (expected 0)")

    # Embedding shape + cosine self-similarity sanity (a vector must be identical to itself = 1.0).
    aligned = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    feature = recognizer.feature(aligned).flatten()
    self_cosine = float(np.dot(feature, feature) / (np.linalg.norm(feature) ** 2))
    print(f"embedding shape : {feature.shape} (expected (128,))")
    print(f"cosine(self,self): {self_cosine:.4f} (expected ~1.0)")

    ok = feature.shape == (128,) and abs(self_cosine - 1.0) < 1e-3
    print("RESULT:", "face path is ready" if ok else "unexpected output - investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
