"""Watching a locked object: has it moved, and is it still there? (the "db" lineage)

A lock is an **anchor**: the box the object occupied when the user locked it, kept in `state.json`
so it survives the hourly restart. Every frame this file asks two questions of each locked class,
and answers them in the only terms the demo has - YOLO boxes:

- **Moved?** The object's centre has drifted further than `MOVE_FRACTION` of the object's *own*
  size from the anchor's centre. Measuring in fractions of the object is what makes one threshold
  work for a laptop and for a phone; measuring the centre rather than the corners is what stops a
  box that merely breathes by a few pixels from reading as a move.
- **Gone?** No detection of that class for `MISSING_S`. And *which* object it was matters, because
  once it is gone there is nothing on screen to point at - the name is all that is left to report.

Two things keep this from crying wolf, which is the whole difficulty with locking:

- **A debounce.** YOLO drops and reacquires an object freely, and a reacquired box can land a long
  way off for one frame. A displacement has to hold for `MOVE_DEBOUNCE_S` before it counts.
- **A first sighting.** Nothing is reported about an object until the camera has seen it once in
  *this* run. After the hourly restart the anchor on disk is only a memory, and that first sighting
  is what brings it up to date.
- **Re-anchoring.** After a confirmed move the old anchor is *dropped* and the object is re-anchored
  where it now stands. Not remembering the old place is deliberate: the object is not going back by
  itself, and an anchor it can never satisfy again would trigger on every frame for the rest of the
  day. Whoever locked it has to lock it again - which is the honest thing to ask.

Who moved it decides what happens, and that is the caller's fact to supply: with a registered user
present a move is somebody tidying up, so it is absorbed silently (the anchor follows). With nobody
recognised on screen the same move is an event. `QUIET_S` then holds off a second event, so an
object being carried across the room reports every few seconds rather than every few frames - long
enough to be readable, short enough to keep a recording alive (see `watchdog.py`).

This file decides *whether something happened*, never what to do about it. It returns
`(kind, description)` pairs from `eventlog.py`'s vocabulary and the watchdog decides the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from applib import eventlog
from applib.state import SessionState
from applib.tracking import Track

MOVE_FRACTION = 0.5    # how far the centre may drift, as a fraction of the object's own size
MOVE_DEBOUNCE_S = 0.5  # ... and for how long, before a wobbling box counts as a move
MISSING_S = 2.0        # gone this long is gone - USER_GOAL's 2-second rule
QUIET_S = 3.0          # after reporting a move, say nothing about that object for this long


@dataclass
class _Watch:
    """What we know about one locked object right now. RAM only - the anchor is in `state.json`."""

    last_seen_at: float
    is_confirmed: bool = False            # seen at least once since we started: see `_check_missing`
    displaced_since: float | None = None  # when the centre first drifted too far, if it has
    quiet_until: float = 0.0              # a move was reported; do not report another until then
    is_missing: bool = False


class ObjectGuard:
    """Anchors and disturbance, for every object the user has locked."""

    def __init__(self, state: SessionState) -> None:
        self.state = state           # the anchors live here, because they must survive a restart
        self._watches: dict[str, _Watch] = {}

    def check(self, now: float, tracks: list[Track], is_user_present: bool) -> list[tuple[str, str]]:
        """Look at this frame and report what happened to the locked objects, if anything."""
        events: list[tuple[str, str]] = []
        for class_name in self.state.locked_objects:  # a copy, so a command may lock while we look
            watch = self._watches.setdefault(class_name, _Watch(last_seen_at=now))
            anchor = self.state.get_anchor(class_name)
            box = _nearest_box(tracks, class_name, anchor)
            if box is None:
                events += self._check_missing(now, class_name, watch, is_user_present)
            else:
                events += self._check_moved(now, class_name, watch, anchor, box, is_user_present)
        return events

    def _check_missing(self, now: float, class_name: str, watch: _Watch,
                       is_user_present: bool) -> list[tuple[str, str]]:
        """Nothing of this class on screen. After `MISSING_S`, that is a theft - once.

        An object that has not been seen *in this run* cannot go missing in it. The demo restarts
        itself every hour, and after a restart the anchor on disk is only a memory: shouting
        "stolen" at a laptop that was put away before the demo started would be a false alarm
        nobody could clear.
        """
        if not watch.is_confirmed or watch.is_missing or now - watch.last_seen_at < MISSING_S:
            return []
        watch.is_missing = True
        if is_user_present:
            return []  # a registered user was here: they took it, and that is allowed
        return [(eventlog.OBJECT_GONE, f"the {class_name} disappeared from view")]

    def _check_moved(self, now: float, class_name: str, watch: _Watch, anchor: list[int] | None,
                     box: list[int], is_user_present: bool) -> list[tuple[str, str]]:
        """The object is on screen. Anchor it, or decide whether it has been moved."""
        watch.last_seen_at = now
        if not watch.is_confirmed or watch.is_missing or anchor is None:
            # The first sighting of this run - which is also how an object that was moved while the
            # demo was restarting gets its anchor brought up to date - or one that has come back.
            watch.is_confirmed, watch.is_missing = True, False
            self.state.set_anchor(class_name, box)
            return []
        if _displacement(anchor, box) <= MOVE_FRACTION:
            watch.displaced_since = None
            return []
        if watch.displaced_since is None:
            watch.displaced_since = now
        if now - watch.displaced_since < MOVE_DEBOUNCE_S:
            return []

        # A real move. The old anchor is gone either way; only the reason differs.
        watch.displaced_since = None
        self.state.set_anchor(class_name, box)
        if is_user_present or now < watch.quiet_until:
            return []
        watch.quiet_until = now + QUIET_S
        return [(eventlog.OBJECT_MOVED, f"the {class_name} was moved")]

    def get_missing(self) -> list[str]:
        """The locked objects that are not on screen and have not been for a while."""
        return [name for name in self.state.locked_objects
                if name in self._watches and self._watches[name].is_missing]

    def describe(self) -> list[str]:
        """The locked objects for the HUD, the missing ones marked: ['laptop', 'cell phone (gone)']."""
        missing = set(self.get_missing())
        return [f"{name} (gone)" if name in missing else name for name in self.state.locked_objects]

    def forget(self, class_name: str) -> None:
        """Drop everything remembered about one object - what unlocking and re-locking both mean."""
        self._watches.pop(class_name, None)

    def reset(self) -> None:
        """Forget every disturbance and every anchor, keeping the locks themselves.

        Disarming is the reset button (see `watchdog.disarm`). The anchors go because the user is
        evidently standing there sorting things out, and each object re-anchors where it is on the
        next frame that sees it.
        """
        self._watches.clear()
        for class_name in self.state.locked_objects:
            self.state.set_anchor(class_name, None)


def _nearest_box(tracks: list[Track], class_name: str, anchor: list[int] | None) -> list[int] | None:
    """This frame's box for a locked class: the one nearest the anchor, or the biggest one.

    Nearest rather than largest once there is an anchor, because two laptops on a table are one
    class to YOLO and the guarded one is the one where it was left.
    """
    boxes = [track.box for track in tracks if track.class_name == class_name]
    if not boxes:
        return None
    if anchor is None:
        return max(boxes, key=_area)
    return min(boxes, key=lambda box: hypot(*_offset(anchor, box)))


def _displacement(anchor: list[int], box: list[int]) -> float:
    """How far the object has moved, measured in its own widths - 0.0 is exactly where it was left."""
    return hypot(*_offset(anchor, box)) / max(_size(anchor), 1.0)


def _offset(anchor: list[int], box: list[int]) -> tuple[float, float]:
    (ax, ay), (bx, by) = _centre(anchor), _centre(box)
    return bx - ax, by - ay


def _centre(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _size(box: list[int]) -> float:
    """One number for how big the object is: the mean of its sides."""
    return ((box[2] - box[0]) + (box[3] - box[1])) / 2


def _area(box: list[int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])
