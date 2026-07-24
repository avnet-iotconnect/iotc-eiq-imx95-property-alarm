"""Face recognition: turn each `person` track into a 128-d face embedding (the "ML.Face" lineage).

Two OpenCV models, both shipped inside the board's system `cv2` - no pip installs, matching the
project's hard dependency constraint:

    YuNet  (cv2.FaceDetectorYN)   - finds a face box + 5 landmarks inside a person crop  (~340 KB ONNX)
    SFace  (cv2.FaceRecognizerSF) - aligns via those landmarks, embeds the face to 128 floats  (few ms)

They are designed to pair: YuNet's landmarks are exactly what SFace's `alignCrop` wants. Both run on
the CPU, off the *serial* Neutron budget that YOLO owns - the board has ample CPU headroom at 30 fps.

This module only produces embeddings; turning an embedding into a *name* is the registry's job
(`registry.identify`). Keeping "who is this vector" out of here keeps ML and DB cleanly split.
"""

from __future__ import annotations

import cv2
import numpy as np

from tracking import Track

# How much to grow a person box before looking for a face - heads often sit just above the YOLO box.
_CROP_MARGIN = 0.15
_MIN_CROP_SIDE = 32  # YuNet needs a non-trivial input; skip slivers


class FaceRecognizer:
    """Detects + embeds faces on person tracks using OpenCV's YuNet + SFace (CPU)."""

    def __init__(self, detector_model_path: str, recognizer_model_path: str, score_threshold: float = 0.7) -> None:
        # Input size is set per-crop in embed_faces; (320, 320) is just the initial placeholder.
        self.detector = cv2.FaceDetectorYN.create(
            detector_model_path, "", (320, 320), score_threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_model_path, "")

    def embed_faces(self, frame_rgb: np.ndarray, tracks: list[Track]) -> None:
        """For every `person` track, write its current face vector into `track.embedding` (or None)."""
        for track in tracks:
            if track.class_name != "person":
                continue
            track.embedding = self._embed_person(frame_rgb, track.box)

    def _embed_person(self, frame_rgb: np.ndarray, box: list[int]) -> np.ndarray | None:
        crop_bgr = self._crop_bgr(frame_rgb, box)
        if crop_bgr is None:
            return None
        face = self._largest_face(crop_bgr)
        if face is None:
            return None
        aligned = self.recognizer.alignCrop(crop_bgr, face)
        embedding = self.recognizer.feature(aligned)  # (1, 128) float32
        return embedding.flatten()

    def _crop_bgr(self, frame_rgb: np.ndarray, box: list[int]) -> np.ndarray | None:
        """Person box -> a margin-padded BGR crop (cv2 face models are trained on BGR, our frame is RGB)."""
        height, width = frame_rgb.shape[:2]
        x1, y1, x2, y2 = box
        margin_x = int((x2 - x1) * _CROP_MARGIN)
        margin_y = int((y2 - y1) * _CROP_MARGIN)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)
        if x2 - x1 < _MIN_CROP_SIDE or y2 - y1 < _MIN_CROP_SIDE:
            return None
        crop_rgb = frame_rgb[y1:y2, x1:x2]
        return np.ascontiguousarray(crop_rgb[:, :, ::-1])  # RGB -> BGR

    def _largest_face(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Detect faces in the crop; return the biggest one's YuNet row (box + 5 landmarks + score)."""
        crop_height, crop_width = crop_bgr.shape[:2]
        self.detector.setInputSize((crop_width, crop_height))
        _, faces = self.detector.detect(crop_bgr)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda face: face[2] * face[3])  # face[2],[3] = width,height
