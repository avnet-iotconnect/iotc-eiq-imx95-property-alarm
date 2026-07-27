"""AntiTheftApp: the one place decisions get made. Everything else just feeds it or is called by it.

This is the composition-root pattern: the app takes its collaborators (`registry`, `overlay`,
`face_worker`, `state`, `scene`) as plain constructor arguments and calls out to them. It never
imports `main`, never constructs its own modules, and nothing calls back *into* it except two
methods:

    on_frame(frame, tracks)   - every camera frame: run the alarm state machine   [video thread]
    on_command(command)       - one parsed command, from voice or later /IOTCONNECT [command thread]

Reading those two methods top to bottom is the whole behaviour of the demo.

**Handlers return the sentence to say, or raise `CommandError`.** That is the entire contract with
`commands.py`, and it is what makes one implementation serve both voice (which speaks the text) and
a future C2D message (which acks with it). Handlers are the layer that knows what *exists* - which
users are registered, what is on screen - so this is where a heard word ("money", "computer") is
matched to a real thing (Marija, laptop); `commands.parse` deliberately does none of that.

**Threading.** `on_command` runs on the command pool, `on_frame` on the video loop. One lock
serialises commands against each other, so no two handlers mutate the registry at once. `on_frame`
takes no lock: it only reads, and the values it reads (an armed flag, a list of names) are
slow-changing enough that seeing one frame's stale value is harmless - the same reasoning
`face_worker.py` uses.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import monotonic, strftime

import numpy as np

import commands
import overlay
import vocabulary
from commands import Command, CommandError
from face_worker import FaceWorker
from overlay import AlarmState, Overlay
from registry import Registry
from scene import SceneDescriber
from state import SessionState
from tracking import Track
from vocabulary import Vocabulary

# How long a sighting still counts for. The tracker reports only tracks seen *this* frame (dolphin
# made that visible on purpose), and YOLO drops a person for the odd frame, so asking "is a person
# on screen right now" makes the ALARM banner flicker several times a second in an empty room.
# Remembering the last sighting for a moment fixes it with no state machine to speak of. The real
# 2s/5s guarded-object timing from GUIDELINES.md is a later pilot; this is only what the banner needs.
PERSON_MEMORY_S = 1.0
USER_MEMORY_S = 2.0


class AntiTheftApp:
    """Alarm state machine + the command handlers, driven entirely by on_frame / on_command."""

    def __init__(
        self, registry: Registry, overlay: Overlay, face_worker: FaceWorker, state: SessionState,
        vocab: Vocabulary, scene: SceneDescriber | None = None, snapshot_dir: Path = Path("."),
    ) -> None:
        self.registry = registry
        self.overlay = overlay
        self.face_worker = face_worker  # registration reads the latest face embedding from here
        self.state = state              # armed flag + locked objects, persisted across restarts
        self.vocabulary = vocab         # turns what was heard into a name or a YOLO class
        self.scene = scene              # None when the VLM is unavailable
        self.snapshot_dir = snapshot_dir
        self.alarm_state = AlarmState.ARMED if state.is_armed else AlarmState.DISARMED
        self._last_tracks: list[Track] = []
        self._last_frame: np.ndarray | None = None
        self._person_seen_at = float("-inf")
        self._user_seen_at = float("-inf")
        self._lock = Lock()

        self._handlers = {
            commands.REGISTER_USER: self._register_user,
            commands.UNREGISTER_USER: self._unregister_user,
            commands.UNREGISTER_LAST: self._unregister_last,
            commands.ARM: self._arm,
            commands.DISARM: self._disarm,
            commands.LOCK_OBJECT: self._lock_object,
            commands.UNLOCK_OBJECT: self._unlock_object,
            commands.DESCRIBE_SCENE: self._describe_scene,
            commands.SNAPSHOT: self._snapshot,
        }

    # --- the two event methods ----------------------------------------------------------------

    def on_frame(self, frame: np.ndarray, tracks: list[Track]) -> None:
        """Each frame: decide the alarm state from who is on screen, then update the overlay."""
        self._last_frame = frame
        self._last_tracks = tracks
        now = monotonic()
        if self.registry.is_person_present(tracks):
            self._person_seen_at = now
        if self.registry.is_registered_user_present(tracks):
            self._user_seen_at = now

        if self.state.is_armed:
            is_person_recent = now - self._person_seen_at < PERSON_MEMORY_S
            is_user_recent = now - self._user_seen_at < USER_MEMORY_S
            self._set_state(AlarmState.ALARM if is_person_recent and not is_user_recent
                            else AlarmState.ARMED)
        else:
            self._set_state(AlarmState.DISARMED)
        self.overlay.set_tracks(tracks)
        self.overlay.set_alarm_state(self.alarm_state)
        self.overlay.set_locked_objects(self.state.locked_objects)

    def on_command(self, command: Command) -> str:
        """Run one command and return what to say. Raises CommandError with a speakable reason."""
        handler = self._handlers.get(command.verb)
        if handler is None:  # only reachable from a C2D payload naming a verb we do not have
            raise CommandError(f"I do not know how to {command.verb.replace('_', ' ')}.")
        with self._lock:
            return handler(command)

    # --- handlers -----------------------------------------------------------------------------

    def _register_user(self, command: Command) -> str:
        """Bind a name to the most prominent face on screen (the worker holds the live embeddings)."""
        name = self._resolve_new_name(command.argument)
        if name in self.registry.user_names:
            raise CommandError(f"{name} is already registered. Say unregister user {name} first.")
        if self.face_worker.register_user(name, self._last_tracks) is None:
            raise CommandError(f"I cannot see a face to register as {name}. Please look at the camera.")
        print(f"[app] registered {name!r} (now knows: {', '.join(self.registry.user_names)})")
        return f"Registered {name}."

    def _unregister_user(self, command: Command) -> str:
        name = self._resolve_known_name(command.argument)
        self.registry.unregister_user(name)
        self.face_worker.forget_user(name)  # clear the label off the live tracks immediately
        return f"Unregistered {name}."

    def _unregister_last(self, command: Command) -> str:
        """The undo for a registration that bound the wrong face or the wrong name."""
        name = self.registry.unregister_last()
        if name is None:
            raise CommandError("There are no registered users to remove.")
        self.face_worker.forget_user(name)
        return f"Unregistered {name}, the most recent user."

    def _arm(self, command: Command) -> str:
        if self.state.is_armed:
            return "The alarm is already armed."
        self.state.set_armed(True)
        return "Alarm armed."

    def _disarm(self, command: Command) -> str:
        if not self.state.is_armed:
            return "The alarm is already disarmed."
        self.state.set_armed(False)
        self._set_state(AlarmState.DISARMED)  # do not wait for the next frame to clear the banner
        return "Alarm disarmed."

    def _lock_object(self, command: Command) -> str:
        """TODO(pilot): a stub. It proves the object can be *named and found*; the 2s/5s theft
        timing that makes a locked object actually trigger the alarm is the next pilot's slice.
        """
        class_name = self._resolve_visible_object(command.argument)
        if not self.state.lock_object(class_name):
            raise CommandError(f"The {class_name} is already locked.")
        return f"The {class_name} is locked. I am watching it."

    def _unlock_object(self, command: Command) -> str:
        """Unlocking matches against what is *locked*, not what is visible - you must be able to
        release an object that has already been carried out of shot.
        """
        heard = command.argument
        locked = self.state.locked_objects
        if not locked:
            raise CommandError("Nothing is locked at the moment.")
        class_name, score = vocabulary.match_choice(heard, locked)
        if class_name is None:
            raise CommandError(f"I do not have a {heard or 'thing'} locked. "
                               f"Locked right now: {', '.join(locked)}.")
        self.state.unlock_object(class_name)
        return f"The {class_name} is unlocked."

    def _snapshot(self, command: Command) -> str:
        """Write what the screen is showing to a timestamped JPEG.

        This exists because the compositor will not give the picture back: weston 14 only lets a
        client it launched itself take a screenshot, the board has no uinput to fake the Super+S
        key binding with, and `/dev/fb0` is an unused emulation buffer full of zeros. Re-rendering
        the frame we already have is both simpler and higher quality than a screen grab would be.
        """
        if self._last_frame is None:
            raise CommandError("I do not have a camera frame yet.")
        path = self.snapshot_dir / f"snapshot-{strftime('%Y%m%d-%H%M%S')}.jpg"
        saved = overlay.save_snapshot(path, self._last_frame, self._last_tracks,
                                      self.overlay.get_hud_lines())
        print(f"[app] snapshot -> {saved}")
        return "Snapshot saved."

    def _describe_scene(self, command: Command) -> str:
        """Hand the current frame, boxes and all, to the VLM. Slow on purpose - see scene.py."""
        if self.scene is None:
            raise CommandError("The scene describer is not available.")
        if self._last_frame is None:
            raise CommandError("I do not have a camera frame yet.")
        return self.scene.describe(self._last_frame, self._last_tracks)

    # --- matching what was heard to what exists -----------------------------------------------

    def _resolve_new_name(self, heard: str) -> str:
        """A name for a NEW user: a vocabulary entry if it matches one, otherwise what was heard.

        Registering someone who is not in vocabulary.json has to work - that is the whole live
        demo - so an unmatched name is accepted as spoken. Adding them to vocabulary.json
        afterwards is what makes the *next* recognition of that name reliable.

        The strict threshold matters here: loose matching would quietly register Michael as Marija,
        because the two are not that far apart as sounds. A recorded `heard_as` still scores 1.0.
        """
        if not heard:
            raise CommandError("Who should I register? Say, register user, and then a name.")
        name, score = vocabulary.match_name(heard, self.vocabulary,
                                            vocabulary.STRICT_MATCH_THRESHOLD)
        if name is not None:
            print(f"[app] heard {heard!r} -> known name {name} ({score:.2f})")
            return name
        return heard.title()

    def _resolve_known_name(self, heard: str) -> str:
        """A name that must already be registered: matched against the face database itself."""
        known = self.registry.user_names
        if not known:
            raise CommandError("Nobody is registered yet.")
        if not heard:
            raise CommandError(f"Who should I remove? I know {', '.join(known)}.")
        name, score = vocabulary.match_choice(heard, known)
        if name is None:
            raise CommandError(f"I do not know anyone called {heard}. I know {', '.join(known)}.")
        print(f"[app] heard {heard!r} -> registered user {name} ({score:.2f})")
        return name

    def _resolve_visible_object(self, heard: str) -> str:
        """A YOLO class name that is on screen right now. People are not lockable objects."""
        visible = [name for name in self.registry.get_visible_classes(self._last_tracks)
                   if name != "person"]
        if not heard:
            raise CommandError("What should I lock? Say, lock object, and then what to guard.")
        if not visible:
            raise CommandError(f"I cannot see a {heard}. There is nothing on screen I could lock.")
        # vocabulary.json maps spoken words ("phone") to YOLO classes ("cell phone"); anything not
        # listed there falls back to matching straight against what the tracker is reporting.
        class_name, score = vocabulary.match_object(heard, self.vocabulary)
        if class_name is None or class_name not in visible:
            class_name, score = vocabulary.match_choice(heard, visible)
        if class_name is None or class_name not in visible:
            raise CommandError(f"I cannot see a {heard}. I can see {', '.join(visible)}.")
        print(f"[app] heard {heard!r} -> visible object {class_name} ({score:.2f})")
        return class_name

    def _set_state(self, new_state: AlarmState) -> None:
        """The single place alarm state changes - so every transition has one spot that announces it."""
        if new_state is self.alarm_state:
            return
        self.alarm_state = new_state
        print(f"[app] alarm state -> {new_state.label}")  # later: also /IOTCONNECT telemetry
