"""Multi-object tracking + readable short IDs for the live overlay.

Detection is stateless - each frame is decoded on its own. This module adds the memory: it associates
this frame's boxes with persistent tracks (greedy IoU), so an object keeps one identity across frames.
That lets us SEE on screen when tracking is lost (an ID disappears) or confused (an ID's class flips -
it "became another thing"), which is the whole point of showing IDs.

Two layers, both here so mapping logic lives in one place:
  Tracker      - association + lifecycle. Internal track ids are ever-growing integers (never reused).
  _ShortIdPool - maps those to short, screen-friendly numbers 1..9. Freed numbers go to the BACK of the
                 queue and new tracks take from the FRONT, so a number that just vanished is not reused
                 immediately - a genuinely new object looks new rather than inheriting the departed id.

Dolphin left `Track.identity` as a seam; the elephant pilot fills it. `face.py` runs a face model on
each `person` track and writes the recognized name into `identity` (and the raw `embedding` used to get
it, so a "register user" command can grab a live face). Everything downstream reads `identity`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

MAX_LABELS = 9  # short ids 1..9, each with its own color (see color_for)

# Nine visually distinct colors, 1-based (index by short_id). RGB floats 0..1 for Cairo.
_PALETTE: list[tuple[float, float, float]] = [
    (0.20, 0.85, 0.30),  # 1 green
    (0.25, 0.60, 1.00),  # 2 blue
    (1.00, 0.60, 0.00),  # 3 orange
    (0.95, 0.25, 0.90),  # 4 magenta
    (0.15, 0.90, 0.90),  # 5 cyan
    (1.00, 0.85, 0.10),  # 6 yellow
    (1.00, 0.35, 0.35),  # 7 red
    (0.65, 0.45, 1.00),  # 8 purple
    (0.70, 0.85, 0.25),  # 9 lime
]


def color_for(short_id: int | None) -> tuple[float, float, float]:
    """Stable color for a short id (1..9); gray when a track ran out of ids."""
    if short_id is None:
        return (0.70, 0.70, 0.70)
    return _PALETTE[(short_id - 1) % len(_PALETTE)]


@dataclass
class Track:
    """One tracked object. `short_id` is what the screen shows; `identity` is the recognized face name."""

    track_id: int  # internal, ever-increasing, never reused
    short_id: int | None  # readable 1..9 shown on screen, or None if the pool was exhausted
    class_name: str
    score: float
    box: list[int]  # xyxy in frame pixels
    misses: int = 0  # consecutive frames this track went unmatched (coasting)
    identity: str | None = None  # a recognized user's name, filled by face.py when a match is found
    embedding: np.ndarray | None = field(default=None, repr=False)  # this track's latest face vector


class Tracker:
    """Greedy-IoU tracker: keeps object identities across frames so the overlay can show stable IDs."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age  # keep a missing track this many frames before dropping it (occlusion)
        self._tracks: list[Track] = []
        self._short_ids = _ShortIdPool(MAX_LABELS)
        self._next_track_id = 0

    def update(self, detections: list[tuple[str, float, list[int]]]) -> list[Track]:
        """Feed this frame's (class_name, score, box_xyxy) detections; get back the visible tracks."""
        matched_track_to_det = self._match(detections)
        matched_dets = set(matched_track_to_det.values())

        for track_index, det_index in matched_track_to_det.items():
            self._apply_detection(self._tracks[track_index], detections[det_index])

        self._age_and_drop_unmatched(set(matched_track_to_det.keys()))

        for det_index, detection in enumerate(detections):
            if det_index not in matched_dets:
                self._spawn_track(detection)

        # Show only tracks seen this frame; a coasting (missing) track stays alive to keep its id if it
        # reappears within max_age, but is not drawn - so a real loss of tracking is visible immediately.
        return [track for track in self._tracks if track.misses == 0]

    def _match(self, detections: list[tuple[str, float, list[int]]]) -> dict[int, int]:
        """Greedily pair existing tracks to detections by best IoU above threshold (position, any class)."""
        available_dets = set(range(len(detections)))
        pairs: dict[int, int] = {}
        for track_index, track in enumerate(self._tracks):
            best_det, best_iou = None, self.iou_threshold
            for det_index in available_dets:
                iou = _iou(track.box, detections[det_index][2])
                if iou >= best_iou:
                    best_det, best_iou = det_index, iou
            if best_det is not None:
                pairs[track_index] = best_det
                available_dets.discard(best_det)
        return pairs

    def _apply_detection(self, track: Track, detection: tuple[str, float, list[int]]) -> None:
        # Keep identity/embedding across the update; face.py refreshes them after tracking runs.
        track.class_name, track.score, track.box = detection
        track.misses = 0

    def _age_and_drop_unmatched(self, matched_track_indices: set[int]) -> None:
        survivors: list[Track] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in matched_track_indices:
                survivors.append(track)
                continue
            track.misses += 1
            if track.misses <= self.max_age:
                survivors.append(track)  # coast: keep identity through a brief miss/occlusion
            elif track.short_id is not None:
                self._short_ids.release(track.short_id)  # gone for good: free its number (LRU)
        self._tracks = survivors

    def _spawn_track(self, detection: tuple[str, float, list[int]]) -> None:
        class_name, score, box = detection
        self._next_track_id += 1
        self._tracks.append(
            Track(self._next_track_id, self._short_ids.acquire(), class_name, score, box)
        )


class _ShortIdPool:
    """Hands out short ids 1..size, reusing freed ones least-recently-freed first (avoids instant reuse)."""

    def __init__(self, size: int) -> None:
        self._available: deque[int] = deque(range(1, size + 1))

    def acquire(self) -> int | None:
        return self._available.popleft() if self._available else None  # None when more objects than ids

    def release(self, short_id: int) -> None:
        self._available.append(short_id)  # back of the queue: last to be handed out again


def _iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)
