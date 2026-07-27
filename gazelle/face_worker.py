"""Runs face recognition OFF the frame loop, on its own cadence (the async part of the "db" seam).

Face detection (YuNet) is ~50 ms on the CPU - too slow to do every frame without dropping YOLO to ~12 fps.
But identity barely changes frame to frame, so we don't need it every frame. This worker fires at most
every `interval_s` (default 200 ms) on a single background thread: it takes the *current* frame + the
person track boxes, embeds each face, asks the registry to assign registered users to tracks, and stashes
the answers in `_identities` (sticky, keyed by track_id) for the main loop to read.

Why this stays simple and safe:
- cv2/tflite release the GIL, so the ~50 ms detect actually runs on a spare core while YOLO keeps 30 fps.
- The main loop hands the worker a frame *reference* - camera.read() already returns a fresh array per
  frame, so there's no aliasing and no extra copy; the worker just keeps that one frame alive until done.
- The only shared state is three small dicts under one short-held lock. `main` never blocks on the ~50 ms.
- The frame loop never queues work: if a cycle is still running when the timer is up, we skip the tick.

`apply()` writes the last cycle's answers onto the live Tracks (name, score, face box, embedding). `app.py`
still just reads `track.identity`; it never sees this thread.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from time import perf_counter

import numpy as np

from face import FaceRecognizer
from registry import Identity, Registry
from tracking import Track


class FaceWorker:
    """Dispatches face recognition on a timer to a background thread and merges results into tracks."""

    def __init__(self, face: FaceRecognizer, registry: Registry, interval_s: float = 0.2) -> None:
        self.face = face
        self.registry = registry
        self.interval_s = interval_s
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._last_dispatch = 0.0
        self._lock = Lock()
        self._identities: dict[int, Identity] = {}       # track_id -> (name, score); sticky across cycles
        self._face_boxes: dict[int, list[int]] = {}      # track_id -> face box; only this cycle's detections
        self._embeddings: dict[int, np.ndarray] = {}     # track_id -> embedding; last cycle, for registration

    def maybe_dispatch(self, frame: np.ndarray, tracks: list[Track]) -> None:
        """If the previous cycle is done and the interval elapsed, launch a new face pass (non-blocking)."""
        if self._future is not None and not self._future.done():
            return
        if self._future is not None:
            error = self._future.exception()
            if error is not None:
                print(f"[face] worker error: {error!r}")
            self._future = None
        now = perf_counter()
        if now - self._last_dispatch < self.interval_s:
            return
        snapshot = [(t.track_id, list(t.box)) for t in tracks if t.class_name == "person"]
        if not snapshot:
            return
        self._last_dispatch = now
        self._future = self._executor.submit(self._run, frame, snapshot)

    def apply(self, tracks: list[Track]) -> None:
        """Write the latest cycle's results onto the live tracks (fast, main thread, every frame)."""
        with self._lock:
            identities, face_boxes, embeddings = self._identities, self._face_boxes, self._embeddings
        for track in tracks:
            track.identity, track.match_score = identities.get(track.track_id, (None, None))
            track.face_box = face_boxes.get(track.track_id)
            track.embedding = embeddings.get(track.track_id)

    def register_user(self, name: str, tracks: list[Track]) -> str | None:
        """Bind `name` to the largest person whose face we embedded last cycle; reflect it immediately."""
        with self._lock:
            candidates = [t for t in tracks if t.class_name == "person" and t.track_id in self._embeddings]
            if not candidates:
                return None
            primary = max(candidates, key=lambda t: _box_area(t.box))
            embedding = self._embeddings[primary.track_id]
            self.registry.register_user(name, embedding)
            self._identities = {**self._identities, primary.track_id: (name, 1.0)}  # seed: no register flicker
        return name

    def forget_user(self, name: str) -> None:
        """Drop a name off the live tracks the instant it is unregistered, so the label goes away.

        Without this the sticky identity would keep showing the name until the next face cycle
        reassigned it - and to the person watching the screen, "unregister" would look ignored.
        """
        with self._lock:
            self._identities = {
                track_id: identity for track_id, identity in self._identities.items()
                if identity[0] != name
            }

    def _run(self, frame: np.ndarray, snapshot: list[tuple[int, list[int]]]) -> None:
        """Background: embed each person's face, resolve identities, publish results. Never touches Tracks."""
        started = perf_counter()
        embeddings: dict[int, np.ndarray] = {}
        face_boxes: dict[int, list[int]] = {}
        for track_id, box in snapshot:
            result = self.face.embed_person(frame, box)
            if result is not None:
                embeddings[track_id], face_boxes[track_id] = result
        current_ids = [track_id for track_id, _ in snapshot]
        with self._lock:
            identities = self.registry.resolve_identities(embeddings, current_ids, self._identities)
            self._identities, self._face_boxes, self._embeddings = identities, face_boxes, embeddings
        self._log(started, len(snapshot), embeddings, identities)

    def _log(self, started: float, people: int, embeddings: dict, identities: dict[int, Identity]) -> None:
        named = ", ".join(f"{n} {s:.2f}" for n, s in identities.values() if n) or "-"
        cycle_ms = (perf_counter() - started) * 1000
        print(f"[face] {cycle_ms:5.1f} ms | {people} person, {len(embeddings)} face | {named}")

    def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _box_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)
