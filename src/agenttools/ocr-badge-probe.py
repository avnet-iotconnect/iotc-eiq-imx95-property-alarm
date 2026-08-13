#!/usr/bin/env python3
"""Name-tag OCR feasibility probe: PP-OCRv3 detection + CRNN recognition, all through cv2.dnn.

Answers three questions before any of this is written into the demo:
  1. can the badge be read at all, at the two distances the interaction offers?
  2. how tall does the text have to be, in pixels, before it becomes readable?
  3. does a confidence number separate a real read from noise?

Everything here uses only cv2 -- the same OpenCV build the board ships (4.12.0), no pip runtime.
Run:  .venv-ocr/bin/python agenttools/ocr-badge-probe.py <image> [<image> ...]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

MODELS = Path(__file__).parent / "ocr-models"
DETECTOR_MODEL = MODELS / "text_detection_en_ppocrv3.onnx"
RECOGNIZER_MODEL = MODELS / "text_recognition_CRNN_EN_2021sep.onnx"
_YUNET = "face_detection_yunet_2023mar.onnx"
_LOCAL_YUNET = Path(__file__).parent / "models" / _YUNET  # how it lands when copied to the board
FACE_MODEL = (_LOCAL_YUNET if _LOCAL_YUNET.exists()
              else Path(__file__).parents[2] / "work/models" / _YUNET)

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"  # CRNN_EN: no case, no punctuation
RECOGNIZER_INPUT = (100, 32)

# PP-OCRv3 detection, opencv_zoo's own defaults.
DETECT_BINARY_THRESHOLD = 0.3
DETECT_POLYGON_THRESHOLD = 0.5
DETECT_UNCLIP_RATIO = 2.0
DETECT_MAX_CANDIDATES = 200
DETECT_MEAN = (122.67891434, 116.66876762, 104.00698793)
# DB allocates its feature maps at full input resolution and *segfaults* past roughly two
# megapixels on this host. The board has far less to give, so the cap is a real limit, not a
# probe convenience: anything larger has to be tiled or downscaled before it reaches the net.
DETECT_MAX_PIXELS = 1_600_000


@dataclass
class TextLine:
    """One line of text the detector found, as read back by the recogniser."""

    text: str
    quad: np.ndarray  # 4x2, clockwise from bottom-left
    detection_score: float
    char_confidence: float  # lowest per-character CTC probability -- the weakest link
    height_px: float

    @property
    def centre_y(self) -> float:
        return float(self.quad[:, 1].mean())

    def __str__(self) -> str:
        return (f"h={self.height_px:5.1f}px  det={self.detection_score:.2f}  "
                f"conf={self.char_confidence:.2f}  y={self.centre_y:4.0f}  {self.text!r}")


def _round_up_to_32(value: int) -> int:
    return ((value + 31) // 32) * 32


def detect_lines(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Text quads and their scores. Input is padded to a multiple of 32, which DB requires."""
    height, width = image.shape[:2]
    size = (_round_up_to_32(width), _round_up_to_32(height))
    detector = cv2.dnn.TextDetectionModel_DB(str(DETECTOR_MODEL))
    detector.setBinaryThreshold(DETECT_BINARY_THRESHOLD)
    detector.setPolygonThreshold(DETECT_POLYGON_THRESHOLD)
    detector.setUnclipRatio(DETECT_UNCLIP_RATIO)
    detector.setMaxCandidates(DETECT_MAX_CANDIDATES)
    detector.setInputParams(1 / 255.0, size, DETECT_MEAN)
    padded = cv2.copyMakeBorder(image, 0, size[1] - height, 0, size[0] - width,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    quads, scores = detector.detect(padded)
    return quads, scores


def recognize(image: np.ndarray, quad: np.ndarray, net: cv2.dnn.Net) -> tuple[str, float]:
    """CTC-greedy decode done by hand, because cv2's wrapper returns the string and no confidence.

    Returns the text and the *lowest* per-character probability -- a name is only as trustworthy
    as its weakest letter, and averaging hides one hallucinated character in a long line.
    """
    target = np.array([[0, RECOGNIZER_INPUT[1] - 1], [0, 0],
                       [RECOGNIZER_INPUT[0] - 1, 0],
                       [RECOGNIZER_INPUT[0] - 1, RECOGNIZER_INPUT[1] - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(quad.reshape(4, 2).astype(np.float32), target)
    cropped = cv2.warpPerspective(image, transform, RECOGNIZER_INPUT)
    grey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    net.setInput(cv2.dnn.blobFromImage(grey, 1 / 127.5, RECOGNIZER_INPUT, 127.5))
    output = net.forward()  # (T, 1, 37); class 0 is the CTC blank

    probabilities = np.exp(output[:, 0, :] - output[:, 0, :].max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    indices = probabilities.argmax(axis=1)

    text, confidences, previous = "", [], -1
    for step, index in enumerate(indices):
        if index != 0 and index != previous:
            text += CHARSET[index - 1]
            confidences.append(float(probabilities[step, index]))
        previous = index
    return text, min(confidences) if confidences else 0.0


def read_text(image: np.ndarray, net: cv2.dnn.Net) -> list[TextLine]:
    """Every line of text in the image, top to bottom."""
    quads, scores = detect_lines(image)
    lines = []
    for quad, score in zip(quads, scores):
        text, confidence = recognize(image, quad, net)
        if not text:
            continue
        side_a = np.linalg.norm(quad[0] - quad[1])
        side_b = np.linalg.norm(quad[1] - quad[2])
        lines.append(TextLine(text, quad, float(score), confidence, float(min(side_a, side_b))))
    return sorted(lines, key=lambda line: line.centre_y)


def find_badge_roi(image: np.ndarray) -> tuple[np.ndarray, str]:
    """Below the chin of the largest face, down to the bottom edge -- the brief's rule.

    The ROI is not about resolution; it is what stops the detector reading the room behind
    the person. Falls back to the whole frame when there is no face (badge held up close).
    """
    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (width, height),
                                         score_threshold=0.7)
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return image, "whole frame (no face)"
    face = max(faces, key=lambda f: f[2] * f[3])
    face_x, face_y, face_w, face_h = (float(v) for v in face[:4])
    left = max(0, int(face_x - face_w * 0.6))
    right = min(width, int(face_x + face_w * 1.6))
    top = min(height - 1, int(face_y + face_h))
    return image[top:height, left:right], f"below a {face_w:.0f}px face, {right - left}x{height - top}"


MIN_LINE_HEIGHT_PX = 12.0
MIN_CHAR_CONFIDENCE = 0.45


def pick_name(lines: list[TextLine]) -> tuple[str | None, str]:
    """Which line is the person's name.

    CRNN emits no spaces, so "Nik Markovic" arrives as 'nikmarkovic' and every word-based rule
    we sketched is dead. What replaces it is better: the badge prints the first name twice --
    huge for across-the-room, then again inside the full name -- so the short line is a *prefix*
    of the long one, and it hands us back the word boundary the recogniser threw away.
    """
    usable = [ln for ln in lines
              if ln.text.isalpha() and 2 <= len(ln.text) <= 24
              and ln.height_px >= MIN_LINE_HEIGHT_PX
              and ln.char_confidence >= MIN_CHAR_CONFIDENCE]
    if not usable:
        return None, "nothing read clearly enough to be a name"

    tallest = max(usable, key=lambda ln: ln.height_px)
    for line in usable:
        if line is not tallest and line.text.startswith(tallest.text) and len(line.text) > len(tallest.text):
            first, last = tallest.text, line.text[len(tallest.text):]
            return f"{first.title()} {last.title()}", f"{line.text!r} splits on the tallest line {first!r}"

    return tallest.text.title(), "tallest line read clearly (no full name found)"


def probe(path: Path, net: cv2.dnn.Net) -> None:
    image = cv2.imread(str(path))
    print(f"\n{'=' * 78}\n{path.name}  {image.shape[1]}x{image.shape[0]}\n{'=' * 78}")

    roi, description = find_badge_roi(image)
    for scale in (1, 2, 3, 4):
        scaled = roi if scale == 1 else cv2.resize(roi, None, fx=scale, fy=scale,
                                                   interpolation=cv2.INTER_CUBIC)
        if scaled.shape[0] * scaled.shape[1] > DETECT_MAX_PIXELS:
            print(f"\n--- ROI {description}, upscaled {scale}x -> "
                  f"{scaled.shape[1]}x{scaled.shape[0]}   SKIPPED, over the detector's limit")
            continue
        started = perf_counter()
        lines = read_text(scaled, net)
        elapsed = (perf_counter() - started) * 1000
        name, why = pick_name(lines)
        print(f"\n--- ROI {description}, upscaled {scale}x -> {scaled.shape[1]}x{scaled.shape[0]}"
              f"   [{elapsed:.0f} ms, {len(lines)} lines]")
        for line in lines:
            print(f"      {line}")
        print(f"    NAME -> {name!r}   ({why})")


def main() -> None:
    net = cv2.dnn.readNet(str(RECOGNIZER_MODEL))
    for argument in sys.argv[1:]:
        probe(Path(argument), net)


if __name__ == "__main__":
    main()
