"""The handful of facts that have to survive a restart (the "db" lineage, alongside registry.py).

Why this exists at all: eIQ kills the process after **60 minutes** - the timeout lives inside NXP's
compiled modules and cannot be separated from the models, so at a trade show the demo *will* restart
between visitors. Anything a visitor set by voice must still be true afterwards, or the booth staff
re-arms the alarm by hand every hour. Registered faces already persist in `faces.json`; this is the
rest of it, in `state.json`.

Only *settings* live here, never transient facts. `is_armed` persists; the ALARM state itself does
not, because it is re-derived from what the camera sees on the very next frame (see `app.py`).
"""

from __future__ import annotations

import json
from pathlib import Path


class SessionState:
    """Armed flag + locked objects, written to disk on every change (they change perhaps twice a minute)."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        stored = json.loads(self.path.read_text()) if self.path.exists() else {}
        self._is_armed: bool = bool(stored.get("is_armed", False))  # GUIDELINES: disarmed is the default
        self._locked_objects: list[str] = list(stored.get("locked_objects", []))

    @property
    def is_armed(self) -> bool:
        return self._is_armed

    def set_armed(self, is_armed: bool) -> None:
        self._is_armed = is_armed
        self._save()

    @property
    def locked_objects(self) -> list[str]:
        """YOLO class names currently under guard, e.g. ['laptop', 'cell phone']."""
        return list(self._locked_objects)

    def lock_object(self, class_name: str) -> bool:
        """True if it was added, False if it was already locked."""
        if class_name in self._locked_objects:
            return False
        self._locked_objects.append(class_name)
        self._save()
        return True

    def unlock_object(self, class_name: str) -> bool:
        """True if it was removed, False if it was not locked in the first place."""
        if class_name not in self._locked_objects:
            return False
        self._locked_objects.remove(class_name)
        self._save()
        return True

    def _save(self) -> None:
        self.path.write_text(json.dumps(
            {"is_armed": self._is_armed, "locked_objects": self._locked_objects}, indent=2))
