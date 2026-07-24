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

from overlay import AlarmState, Overlay
from registry import Registry
from tracking import Track

_REGISTER_PREFIX = "register user "


class AntiTheftApp:
    """Alarm state machine + registration, driven entirely by on_frame / on_command."""

    def __init__(self, registry: Registry, overlay: Overlay) -> None:
        self.registry = registry
        self.overlay = overlay
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
        """Bind `name` to the face of the most prominent person currently on screen."""
        track = self._primary_person_with_face()
        if not name:
            print("[app] register: no name given")
        elif track is None:
            print(f"[app] register {name!r}: no person with a visible face on screen right now")
        else:
            self.registry.register_user(name, track.embedding)
            track.identity = name  # reflect it immediately, before the next frame re-identifies
            print(f"[app] registered {name!r} (now knows: {', '.join(self.registry.user_names)})")

    def _primary_person_with_face(self) -> Track | None:
        """The largest person track that produced a face embedding this frame - the one being registered."""
        candidates = [t for t in self._last_tracks if t.class_name == "person" and t.embedding is not None]
        return max(candidates, key=lambda t: _box_area(t.box), default=None)

    def _set_state(self, new_state: AlarmState) -> None:
        """The single place alarm state changes - so every transition has one spot that announces it."""
        if new_state is self.state:
            return
        self.state = new_state
        print(f"[app] alarm state -> {new_state.label}")  # later: also IoTConnect telemetry


def _box_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)
