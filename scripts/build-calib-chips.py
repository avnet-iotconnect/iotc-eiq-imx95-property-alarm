#!/usr/bin/env python3
"""Build the committed SFace calibration chips from public-domain portraits. Run once; the chips
are committed, so this exists as provenance and for changing the set.

Writes files/sface-calib/*.png plus a README naming every source. Two things matter:

- **Licence.** US government works are public domain by statute (17 U.S.C. sec.105), so NASA's image
  library can be redistributed in a public repo without a per-image negotiation. Faces of real
  people who were photographed officially, in their professional capacity.
- **Resolution.** The portraits are studio-quality, with far more facial detail than a 640x480
  webcam ever delivers. Each source is downscaled until the detected face is about the size this
  demo's camera sees (~90 px) *before* alignment, so the calibration chips carry camera-like detail
  rather than magazine detail. Quantization ranges should come from the data the model will meet.

    .venv/bin/python scripts/build-calib-chips.py
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "work" / "models"
OUT = ROOT / "files" / "sface-calib"
TARGET_FACE_PX = 90  # what a person standing at this booth measures on a 640x480 frame

# Deliberately varied: suits, flight suits, shirtsleeves, a range of ages and skin tones. All from
# NASA's public library, which is the largest cleanly-licensed source of official portraits.
QUERIES = [
    "official portrait astronaut candidate",
    "official portrait administrator NASA",
    "NASA engineer portrait",
    "NASA scientist portrait",
    "NASA director official portrait",
]


def search(query: str, limit: int = 6) -> list[tuple[str, str]]:
    url = ("https://images-api.nasa.gov/search?media_type=image&q="
           + urllib.parse.quote(query))
    data = json.load(urllib.request.urlopen(url, timeout=60))
    found = []
    for item in data["collection"]["items"][:limit]:
        title = item["data"][0]["title"]
        try:
            assets = json.load(urllib.request.urlopen(item["href"], timeout=60))
        except Exception:
            continue
        jpgs = [a for a in assets if a.endswith(".jpg")]
        medium = [a for a in jpgs if "medium" in a] or jpgs
        if medium:
            found.append((title, medium[0]))
    return found


def chip_from(url: str, detector, aligner) -> np.ndarray | None:
    """Download, shrink until the face is camera-sized, then align exactly as the demo does."""
    raw = np.frombuffer(urllib.request.urlopen(url, timeout=60).read(), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        return None
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    if faces is None:
        return None
    row = max(faces, key=lambda r: r[2] * r[3])
    face_px = min(row[2], row[3])
    if face_px < TARGET_FACE_PX:
        return None                                   # already smaller than the camera would see
    scale = TARGET_FACE_PX / face_px
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)                 # re-detect: landmarks must match the pixels
    if faces is None:
        return None
    return aligner.alignCrop(small, max(faces, key=lambda r: r[2] * r[3]))


def main() -> None:
    detector = cv2.FaceDetectorYN.create(
        str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.85, 0.3, 5000)
    aligner = cv2.FaceRecognizerSF.create(
        str(MODELS / "face_recognition_sface_2021dec.onnx"), "")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()

    credits, seen = [], set()
    for query in QUERIES:
        for title, url in search(query):
            if title in seen or len(credits) >= 10:
                continue
            chip = chip_from(url, detector, aligner)
            if chip is None:
                continue
            seen.add(title)
            name = f"face-{len(credits) + 1:02d}.png"
            cv2.imwrite(str(OUT / name), chip)
            credits.append((name, title, url))
            print(f"{name}  {title[:58]}")

    lines = ["# SFace int8 calibration chips", "",
             "112x112 aligned face chips, used only to calibrate the int8 quantization of SFace",
             "(`scripts/package-models.sh`). Downscaled to roughly the face size this demo's camera",
             "sees before alignment, so the quantization ranges match real input.", "",
             "All sources are **NASA** images: works of the US federal government, in the public",
             "domain under 17 U.S.C. 105.", ""]
    lines += [f"- `{name}` — {title} — <{url}>" for name, title, url in credits]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\n{len(credits)} chips + README in {OUT}")


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  (only needed by search())

    main()
