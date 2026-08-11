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

Each look also comes back measured - how big, how sharp, how square-on the face was - because
registration has to be able to refuse a bad one. See `FaceSample` and `find_quality_problem` below.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

from applib import neutron

# How much to grow a person box before looking for a face - heads often sit just above the YOLO box.
_CROP_MARGIN = 0.15
_MIN_CROP_SIDE = 32  # YuNet needs a non-trivial input; skip slivers

# The registration gate. These are starting points, measured off photographed faces scaled to the
# size this camera sees rather than off a hundred people at a booth - so expect to move them. Every
# registration attempt prints what it measured beside what it wanted, which is the whole tuning
# procedure: register somebody a few times and read the console.
MIN_FACE_PIXELS = 60        # shorter side of the face box, in frame pixels
MIN_STRAIGHTNESS = 0.50     # 0..1 from the landmarks; a clearly turned head measures ~0.3
MIN_DETECTION_SCORE = 0.90  # YuNet's own confidence (it is already told not to report below 0.7)
MIN_SHARPNESS = 60.0        # Laplacian variance of the aligned chip - the blur/motion detector
MIN_CONSISTENCY = 0.75      # cosine between this look at the face and the one 200 ms before it

# The nose offset, in eye-spacings, at which we call a face fully turned away. Only used to turn
# the raw offset into the 0..1 `straightness` above, so the two constants move together.
_TURNED_AWAY_OFFSET = 0.35


@dataclass
class FaceSample:
    """One look at one person's face: the vector we would store, and how good a look it was.

    The four measurements are all cheap by-products of work already done - YuNet hands back the
    score and the landmarks, and the aligned chip is sitting in memory anyway.
    """

    embedding: np.ndarray
    face_box: list[int]        # xyxy in frame pixels, for the overlay
    detection_score: float     # how sure YuNet was that this is a face at all
    face_pixels: int           # shorter side of the face box: how much of a face the embedder saw
    sharpness: float           # Laplacian variance of the aligned chip: blur, or motion
    straightness: float        # 0..1 from the eye/nose landmarks: 1.0 is looking right at us

    def describe(self) -> str:
        """One line of numbers for the console - the raw material for tuning the thresholds."""
        return (f"size {self.face_pixels}px  straight {self.straightness:.2f}  "
                f"score {self.detection_score:.2f}  sharpness {self.sharpness:.0f}")


def find_quality_problem(sample: FaceSample, consistency: float) -> str | None:
    """What to ask the person to change before we store this face, or None when it will do.

    Registration is the one moment where a bad look is expensive, and it is worth seeing by how
    much. Embedding one face twice - once held still, once smeared by movement - gives two vectors
    only 0.55 alike, while a *different* person sits at 0.06 against the still one and 0.14 against
    the smeared one. So a bad registration does not merely fail to match: it drifts off its owner
    and towards everybody else at the same time, and the gap identity is decided on falls from 0.94
    to 0.41 with `registry.SFACE_COSINE_THRESHOLD` at 0.363. That is what "it mixes people up"
    looks like from underneath. Matching itself is deliberately left ungated: it gets a fresh look
    five times a second and can afford a poor one.

    `consistency` is the cosine between this look and the previous one, which the caller has and
    this module does not. Two looks 200 ms apart that disagree mean the face was moving, or that
    the largest person on screen changed halfway through - either way, not a face to store.

    The order is the order the messages are worth hearing: being too far away makes every other
    number bad, so say that first and let the rest be measured on a face that is actually there.
    """
    if sample.face_pixels < MIN_FACE_PIXELS:
        return "Please come closer to the camera"
    if sample.straightness < MIN_STRAIGHTNESS:
        return "Please look straight at the camera"
    if sample.detection_score < MIN_DETECTION_SCORE:
        return "Let me see your whole face"
    if sample.sharpness < MIN_SHARPNESS:
        return "Please hold still"
    if consistency < MIN_CONSISTENCY:
        return "Please hold still and keep looking at the camera"
    return None


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
        delegates = [load_delegate(neutron.DELEGATE_PATH)] if use_neutron else []
        self.embedder = Interpreter(
            model_path=embed_model_path, experimental_delegates=delegates, num_threads=2
        )
        self.embedder.allocate_tensors()
        self._embed_in = self.embedder.get_input_details()[0]
        self._embed_out = self.embedder.get_output_details()[0]

    def embed_person(self, frame_rgb: np.ndarray, box: list[int]) -> FaceSample | None:
        """One person crop -> a measured `FaceSample`, or None when there is no face to be found."""
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
        return FaceSample(
            embedding, face_box,
            detection_score=float(face[-1]),
            face_pixels=int(min(fw, fh)),
            sharpness=sharpness(aligned_bgr),
            straightness=straightness(face),
        )

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
        if faces is None:
            return None
        # YuNet can emit inf/nan rows; the whole row is checked because the landmarks are used too.
        valid = [face for face in faces if np.all(np.isfinite(face))]
        if not valid:
            return None
        return max(valid, key=lambda face: face[2] * face[3])  # face[2],[3] = width,height


def sharpness(aligned_bgr: np.ndarray) -> float:
    """Laplacian variance of the aligned chip: high on a crisp face, near zero on a smeared one.

    Measured on the 112x112 the embedder actually sees rather than on the frame, so a face that was
    upscaled from far away scores low for the same reason it embeds badly - there is no detail in it.
    """
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def straightness(face: np.ndarray) -> float:
    """How square-on the face is, 0..1, from YuNet's eye and nose landmarks. 1.0 is looking at us.

    The nose tip of a face turned to one side slides towards the near eye. Measuring that offset
    *along the eye line* means a tilted head does not read as a turned one, and dividing by the eye
    spacing means the answer does not depend on how far away the person is standing.

    YuNet's row is [x, y, w, h, right eye, left eye, nose, right mouth, left mouth, score].
    """
    right_eye, left_eye, nose = face[4:6], face[6:8], face[8:10]
    eye_span = float(np.linalg.norm(left_eye - right_eye))
    if eye_span <= 0:
        return 0.0
    along_eyes = (left_eye - right_eye) / eye_span
    offset = abs(float(np.dot(nose - (right_eye + left_eye) / 2, along_eyes))) / eye_span
    return max(0.0, 1.0 - offset / _TURNED_AWAY_OFFSET)
