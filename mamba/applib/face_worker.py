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

Registration goes through here too (`try_register_user`), because this is where the live looks at a
face are - both the latest one and the one before it, which is what "was the face holding still"
means. The thresholds it judges them by are `face.py`'s.

`look_at_photo` is the one thing here that has nothing to do with the camera: a picture that arrived
from the cloud, measured on this same thread because **the models are not re-entrant** - one YuNet
detector, one SFace interpreter, so every call into `face.py` happens on this one worker thread.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import numpy as np

from applib.face import FaceRecognizer, FaceSample, find_photo_problem, find_quality_problem
from applib.registry import Identity, Registry
from applib.tracking import Track

# A photograph is detected at this size at most. YuNet is handed the whole picture, and finding one
# face in a phone camera's twelve megapixels costs seconds of CPU; it also keeps `MIN_FACE_PIXELS`
# meaning roughly what it means on a 640x480 camera frame instead of passing everything.
MAX_PHOTO_SIDE = 1280


@dataclass
class Registration:
    """The outcome of one registration attempt. The face was stored exactly when `problem` is None.

    `report` is the measured quality either way - it is worth printing on success too, because a
    registration that *just* passed is the one to look at when a name starts landing on the wrong
    person.
    """

    problem: str | None
    report: str = "no face"
    nearest: str = "-"  # the registered user this face is most like: the mix-up, made visible
    duplicate: str | None = None  # ... and that user's name when the demo would call this face them

    @property
    def is_registered(self) -> bool:
        return self.problem is None


@dataclass
class PhotoLook:
    """What one photograph turned out to hold. `embedding` is set exactly when `problem` is None.

    Nothing has been written when this comes back - unlike `Registration`, which reports one that
    already happened. The name belongs to the caller, so the decision does too: `app.on_face_image`.
    """

    problem: str | None
    report: str = "no face"
    nearest: str = "-"                        # the registered user this face is most like
    embedding: np.ndarray | None = None       # what to store, when there is nothing wrong with it
    duplicate: str | None = None              # ... and that user's name when this *is* them


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
        self._samples: dict[int, FaceSample] = {}        # track_id -> last cycle's look at that face
        self._previous: dict[int, FaceSample] = {}       # ... and the cycle before it, for "hold still"

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
        """Write the latest cycle's results onto the live tracks (fast, main thread, every frame).

        **Only a person carries a name.** Tracking matches by position and ignores class, so a
        person who steps out of shot can leave their track on the chair behind them - and a cycle
        only runs while at least one person track exists, so nothing would ever come back to correct
        it. The name is dropped here, and forgotten, the moment the track stops being a person.
        """
        with self._lock:
            identities, samples = self._identities, self._samples
        departed = {track.track_id for track in tracks if track.class_name != "person"} & set(identities)
        for track in tracks:
            if track.class_name != "person":
                track.identity, track.match_score, track.face_box, track.embedding = None, None, None, None
                continue
            track.identity, track.match_score = identities.get(track.track_id, (None, None))
            sample = samples.get(track.track_id)
            track.face_box = sample.face_box if sample else None
            track.embedding = sample.embedding if sample else None
        if departed:
            with self._lock:
                self._identities = {track_id: identity for track_id, identity in self._identities.items()
                                    if track_id not in departed}

    def try_register_user(self, name: str, tracks: list[Track]) -> Registration:
        """Bind `name` to the largest person on screen - but only if this was a good look at them.

        The look being judged is the last cycle's, at most ~200 ms old, which is why `app.py` counts
        the person down to this moment instead of registering the instant it hears the command: the
        sample this reads is whatever the camera had while they were being told to look at it.

        Nothing is written when the look is not good enough. The caller counts down again and asks
        for another one - see `face.find_quality_problem` for what "good enough" means and why the
        gate is here rather than at match time.

        `duplicate` is measured here and acted on in `app.py`: it names the already-registered user
        this face scores *above the match threshold* against, which is to say the person the demo
        would put a label on if they walked into shot. Registering that face under a second name is
        how one person ends up in the database twice, and after that which name appears is a
        coin-toss between two vectors of the same face. This file only reports it - what to do about
        it is the owner's decision, and it is in `app._register_user`.
        """
        with self._lock:
            candidates = [t for t in tracks if t.class_name == "person" and t.track_id in self._samples]
            if not candidates:
                return Registration("I cannot see your face")
            primary = max(candidates, key=lambda t: _box_area(t.box))
            sample = self._samples[primary.track_id]
            previous = self._previous.get(primary.track_id)
            # No previous look is not "steady enough yet" - it is nothing to compare against, which
            # scores zero and asks the person to hold still for one more cycle. Two consecutive
            # readable looks is itself part of the bar.
            consistency = (self.registry.compare(sample.embedding, previous.embedding)
                           if previous is not None else 0.0)
            report = f"{sample.describe()}  steady {consistency:.2f}"
            near_name, near_score = self.registry.find_nearest_user(sample.embedding)
            nearest = f"{near_name} {near_score:.2f}" if near_name else "nobody registered yet"
            # The caller has already refused a name that is registered, so this can only be somebody
            # else's name - the same face wearing two of them.
            is_duplicate = near_score is not None and near_score >= self.registry.match_threshold
            problem = find_quality_problem(sample, consistency)
            if problem is not None:
                return Registration(problem, report, nearest)
            self.registry.register_user(name, sample.embedding)
            self._identities = {**self._identities, primary.track_id: (name, 1.0)}  # seed: no register flicker
            return Registration(None, report, nearest, near_name if is_duplicate else None)

    def look_at_photo(self, image_path: Path) -> PhotoLook:
        """Measure the face in a photograph the way a live look is measured. Writes nothing.

        The only method here that **blocks on the worker thread**, and it has to: the models are
        shared and not re-entrant, so a cloud thread embedding a file while a face cycle is
        embedding the camera would interleave inside both YuNet and SFace. Queuing it behind the
        cycle costs whoever asked ~50 ms.
        """
        return self._executor.submit(self._look_at_photo, image_path).result()

    def _look_at_photo(self, image_path: Path) -> PhotoLook:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return PhotoLook("it is not an image I can read")
        image_bgr = _fit_within(image_bgr, MAX_PHOTO_SIDE)
        height, width = image_bgr.shape[:2]
        image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        # The whole picture stands in for a person box: a portrait is a person crop already, so
        # embed_person does what it does for a track - largest face, aligned, embedded, measured.
        sample = self.face.embed_person(image_rgb, [0, 0, width, height])
        if sample is None:
            return PhotoLook("I cannot find a face in it")
        near_name, near_score = self.registry.find_nearest_user(sample.embedding)
        nearest = f"{near_name} {near_score:.2f}" if near_name else "nobody registered yet"
        is_duplicate = near_score is not None and near_score >= self.registry.match_threshold
        problem = find_photo_problem(sample)
        if problem is not None:
            return PhotoLook(problem, sample.describe(), nearest)
        return PhotoLook(None, sample.describe(), nearest, sample.embedding,
                         near_name if is_duplicate else None)

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
        samples: dict[int, FaceSample] = {}
        for track_id, box in snapshot:
            sample = self.face.embed_person(frame, box)
            if sample is not None:
                samples[track_id] = sample
        embeddings = {track_id: sample.embedding for track_id, sample in samples.items()}
        current_ids = [track_id for track_id, _ in snapshot]
        with self._lock:
            identities = self.registry.resolve_identities(embeddings, current_ids, self._identities)
            self._identities = identities
            self._previous, self._samples = self._samples, samples

    def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _box_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def _fit_within(image: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink an image until its longer side fits, keeping its aspect. Smaller images are untouched."""
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)),
                      interpolation=cv2.INTER_AREA)
