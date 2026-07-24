"""AntiTheftApp: the one place decisions get made. Everything else just feeds it or is called by it.

This is the composition-root pattern: the app takes its collaborators (`registry`,
`overlay`) as plain constructor arguments and calls out to them. It never imports `main`, never
constructs its own modules, and nothing calls back *into* it except `main`'s two event methods:

    on_frame(tracks)   - what happens each camera frame: run the alarm state machine
    on_command(text)   - what happens when a line appears in command.txt: register / arm / disarm

Reading these two methods top to bottom is the whole behavior of the demo. The alarm rule for this
pilot is intentionally the single face-recognition question: armed + a person on screen + nobody
recognized => ALARM. (Guarded-object / theft-timing states come in a later pilot.)
"""

from __future__ import annotations

from face_worker import FaceWorker
from overlay import AlarmState, Overlay
from registry import Registry
from tracking import Track

_REGISTER_PREFIX = "register user "


class AntiTheftApp:
    """Alarm state machine + registration, driven entirely by on_frame / on_command."""

    def __init__(self, registry: Registry, overlay: Overlay, face_worker: FaceWorker) -> None:
        self.registry = registry
        self.overlay = overlay
        self.face_worker = face_worker  # registration reads the latest face embedding from here
        self.state = AlarmState.ARMED  # start armed: the first unrecognized person trips the alarm
        self._last_tracks: list[Track] = []

    def on_frame(self, tracks: list[Track]) -> None:
        """Each frame: decide the alarm state from who/what is on screen, then update the overlay."""
        self._last_tracks = tracks
        if self.state is not AlarmState.DISARMED:
            person_present = self.registry.is_person_present(tracks)
            user_present = self.registry.is_registered_user_present(tracks)
            self._set_state(AlarmState.ALARM if person_present and not user_present else AlarmState.ARMED)
        self.overlay.set_tracks(tracks)
        self.overlay.set_alarm_state(self.state)

    def on_command(self, text: str) -> None:
        """One line from command.txt. Kept to a few obvious verbs - this is the demo's control surface."""
        command = text.strip().lower()
        if command.startswith(_REGISTER_PREFIX):
            self._register_user(text.strip()[len(_REGISTER_PREFIX):])
        elif command in ("arm", "arm alarm"):
            self._set_state(AlarmState.ARMED)
        elif command in ("disarm", "disarm alarm"):
            self._set_state(AlarmState.DISARMED)
        else:
            print(f"[app] ignoring unknown command: {text.strip()!r}")

    def _register_user(self, name: str) -> None:
        """Bind `name` to the most prominent person on screen (the worker holds the live embeddings)."""
        if not name:
            print("[app] register: no name given")
        elif self.face_worker.register_user(name, self._last_tracks) is None:
            print(f"[app] register {name!r}: no person with a visible face on screen right now")
        else:
            print(f"[app] registered {name!r} (now knows: {', '.join(self.registry.user_names)})")

    def _set_state(self, new_state: AlarmState) -> None:
        """The single place alarm state changes - so every transition has one spot that announces it."""
        if new_state is self.state:
            return
        self.state = new_state
        print(f"[app] alarm state -> {new_state.label}")  # later: also IoTConnect telemetry
