"""The handful of facts that have to survive a restart (the "db" lineage, alongside registry.py).

Why this exists at all: eIQ kills the process after **60 minutes** - the timeout lives inside NXP's
compiled modules and cannot be separated from the models, so at a trade show the demo *will* restart
between visitors. Anything a visitor set by voice must still be true afterwards, or the booth staff
re-arms the alarm by hand every hour. Registered faces already persist in `faces.json`; this is the
rest of it, in `state.json`.

Only *settings* live here, never transient facts. `is_armed` persists, and so does each locked
object's **anchor** - the box it occupied when it was locked, which is the whole content of a lock
(see `guard.py`). The alert flag, the recording and the event log do not: they describe a moment,
and after a restart the honest answer about them is "I do not know", which is what an empty
`watchdog.py` says by itself.
"""

from __future__ import annotations

import json
from pathlib import Path

Box = list[int]  # xyxy in frame pixels, as YOLO reports it


class SessionState:
    """Armed flag + locked objects and their anchors, written to disk on every change.

    Written *immediately* rather than on a timer, because the process is killed rather than asked to
    stop. The writes are rare by construction - arming happens twice a minute and an anchor only
    moves when the guard has confirmed a real move - and the file is a few hundred bytes.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        stored = json.loads(self.path.read_text()) if self.path.exists() else {}
        self._is_armed: bool = bool(stored.get("is_armed", False))  # GUIDELINES: disarmed is the default
        self._locks: dict[str, Box | None] = _read_locks(stored.get("locked_objects", {}))

    @property
    def is_armed(self) -> bool:
        return self._is_armed

    def set_armed(self, is_armed: bool) -> None:
        self._is_armed = is_armed
        self._save()

    @property
    def locked_objects(self) -> list[str]:
        """YOLO class names currently under guard, e.g. ['laptop', 'cell phone']. A copy: the video
        loop walks this list while a command thread may be adding to it.
        """
        return list(self._locks)

    def get_anchor(self, class_name: str) -> Box | None:
        """Where the object was locked, or None if that is not known yet - see `guard.py`."""
        return self._locks.get(class_name)

    def set_anchor(self, class_name: str, box: Box | None) -> None:
        """Move the anchor (or forget it). Silently ignored for something that is not locked."""
        if class_name in self._locks:
            self._locks[class_name] = box
            self._save()

    def lock_object(self, class_name: str, anchor: Box | None = None) -> bool:
        """True if it was added, False if it was already locked (the caller decides what that means)."""
        if class_name in self._locks:
            return False
        self._locks[class_name] = anchor
        self._save()
        return True

    def unlock_object(self, class_name: str) -> bool:
        """True if it was removed, False if it was not locked in the first place."""
        if class_name not in self._locks:
            return False
        del self._locks[class_name]
        self._save()
        return True

    def _save(self) -> None:
        self.path.write_text(json.dumps(
            {"is_armed": self._is_armed, "locked_objects": self._locks}, indent=2))


def _read_locks(stored) -> dict[str, Box | None]:
    """Read the locks, accepting the older `["laptop"]` form as locks with no anchor yet."""
    if isinstance(stored, list):
        return {class_name: None for class_name in stored}
    return {class_name: anchor for class_name, anchor in stored.items()}
