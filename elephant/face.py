"""Face recognition: turn each `person` track into a 128-d face embedding (the "ML.Face" lineage).

Three pieces, split by what they run on:
  YuNet  (cv2.FaceDetectorYN)   - find a face box + 5 landmarks inside a person crop        [CPU, cv2]
  align  (cv2.FaceRecognizerSF) - warp the face to a canonical 112x112 using those landmarks [CPU, cv2]
  SFace  (tflite Interpreter)   - embed the aligned 112x112 face to 128 floats              [NEUTRON NPU]

The embedder is the int8 SFace we converted for Neutron (scripts/sface-convert-neutron.sh): ~5 ms on the
NPU vs ~58 ms for cv2's CPU path, embedding cosine ~0.99 vs the float model. Detection + alignment stay on
the CPU (YuNet scales with crop size; alignment is just a landmark warp). SFace wants RGB 0-255 - cv2's
alignCrop returns BGR, so we swap - and the int8 input maps straight through (scale 1.0, zero -128).

This module only produces embeddings + the face box (for the overlay); turning an embedding into a *name*
is the registry's job (`registry.resolve_identities`). Keeping "who is this vector" out of here keeps the
ML/DB split. It's called off the frame loop by `face_worker.py` on one person crop at a time.
"""

from __future__ import annotations

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

NEUTRON_DELEGATE_PATH = "/usr/lib/libneutron_delegate.so"

# How much to grow a person box before looking for a face - heads often sit just above the YOLO box.
_CROP_MARGIN = 0.15
_MIN_CROP_SIDE = 32  # YuNet needs a non-trivial input; skip slivers


class FaceRecognizer:
    """YuNet detect + cv2 align + Neutron SFace embed, run over person tracks."""

    def __init__(
        self, detector_model_path: str, aligner_model_path: str, embed_model_path: str,
        use_neutron: bool, score_threshold: float = 0.7,
    ) -> None:
        # Input size is set per-crop in embed_person; (320, 320) is just the initial placeholder.
        self.detector = cv2.FaceDetectorYN.create(
            detector_model_path, "", (320, 320), score_threshold, 0.3, 5000
        )
        self.aligner = cv2.FaceRecognizerSF.create(aligner_model_path, "")  # used only for alignCrop
        delegates = [load_delegate(NEUTRON_DELEGATE_PATH)] if use_neutron else []
        self.embedder = Interpreter(
            model_path=embed_model_path, experimental_delegates=delegates, num_threads=2
        )
        self.embedder.allocate_tensors()
        self._embed_in = self.embedder.get_input_details()[0]
        self._embed_out = self.embedder.get_output_details()[0]

    def embed_person(self, frame_rgb: np.ndarray, box: list[int]) -> tuple[np.ndarray, list[int]] | None:
        """One person crop -> (128-d embedding, face box xyxy in frame px), or None if no face found."""
        crop_bgr, offset_x, offset_y = self._crop_bgr(frame_rgb, box)
        if crop_bgr is None:
            return None
        face = self._largest_face(crop_bgr)
        if face is None:
            return None
        aligned_bgr = self.aligner.alignCrop(crop_bgr, face)  # 112x112 BGR uint8
        embedding = self._embed_chip(aligned_bgr)
        fx, fy, fw, fh = face[:4]
        face_box = [int(offset_x + fx), int(offset_y + fy), int(offset_x + fx + fw), int(offset_y + fy + fh)]
        return embedding, face_box

    def _embed_chip(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Run the aligned face through Neutron SFace -> 128-d embedding (SFace input is RGB 0-255)."""
        rgb = aligned_bgr[:, :, ::-1].astype(np.float32)
        scale, zero_point = self._embed_in["quantization"]
        dtype = self._embed_in["dtype"]
        tensor = np.clip(np.round(rgb / scale + zero_point), np.iinfo(dtype).min, np.iinfo(dtype).max)
        self.embedder.set_tensor(self._embed_in["index"], tensor.astype(dtype)[np.newaxis, ...])
        self.embedder.invoke()
        out_scale, out_zero = self._embed_out["quantization"]
        raw = self.embedder.get_tensor(self._embed_out["index"]).astype(np.float32).flatten()
        return (raw - out_zero) * out_scale

    def _crop_bgr(self, frame_rgb: np.ndarray, box: list[int]) -> tuple[np.ndarray | None, int, int]:
        """Person box -> a margin-padded BGR crop + its top-left offset in the frame (for face-box mapping)."""
        height, width = frame_rgb.shape[:2]
        x1, y1, x2, y2 = box
        margin_x = int((x2 - x1) * _CROP_MARGIN)
        margin_y = int((y2 - y1) * _CROP_MARGIN)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)
        if x2 - x1 < _MIN_CROP_SIDE or y2 - y1 < _MIN_CROP_SIDE:
            return None, 0, 0
        crop_rgb = frame_rgb[y1:y2, x1:x2]
        return np.ascontiguousarray(crop_rgb[:, :, ::-1]), x1, y1  # RGB -> BGR, plus offset

    def _largest_face(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Detect faces in the crop; return the biggest one's YuNet row (box + 5 landmarks + score)."""
        crop_height, crop_width = crop_bgr.shape[:2]
        self.detector.setInputSize((crop_width, crop_height))
        _, faces = self.detector.detect(crop_bgr)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda face: face[2] * face[3])  # face[2],[3] = width,height
