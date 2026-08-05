"""AntiTheftApp: the one place decisions get made. Everything else just feeds it or is called by it.

This is the composition-root pattern: the app takes its collaborators (`registry`, `overlay`,
`face_worker`, `state`, `scene`) as plain constructor arguments and calls out to them. It never
imports `main`, never constructs its own modules, and nothing calls back *into* it except two
methods:

    on_frame(frame, tracks)   - every camera frame: run the alarm state machine   [video thread]
    on_command(command)       - one parsed command, from voice, the file or C2D   [command thread]

Reading those two methods top to bottom is the whole behaviour of the demo.

**The LLM is a command source, not a decision maker.** `ask` hands a sentence to the 7B model on
the Ara-240 (`ask.py`), whose tools come back in through `on_command` - so the model can arm the
alarm, but it never decides *whether* the alarm should go off. That stays where it is below, in
about fifteen lines of `if`.

**The cloud is not a collaborator here.** `iotc.py` is nowhere in this file. What the dashboard
needs is *written* into `telemetry.TelemetryState` - a plain dict behind a lock - and the
/IOTCONNECT publisher reads it on its own thread. So the alarm logic never blocks on a network call,
and the whole video half still runs with the SDK not installed.

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
from time import monotonic
from typing import TYPE_CHECKING, Callable

import numpy as np

from applib import commands, overlay, vocabulary
from applib.commands import Command, CommandError
from applib.face_worker import FaceWorker
from applib.overlay import AlarmState, Overlay
from applib.registry import Registry
from applib.scene import SceneDescriber
from applib.state import SessionState
from applib.telemetry import TelemetryState
from applib.tracking import Track
from applib.vocabulary import Vocabulary

if TYPE_CHECKING:  # imported for the type only: ask.py pulls in strands-agents, which may not be there
    from app.ask import AskAgent

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
        vocab: Vocabulary, telemetry: TelemetryState, scene: SceneDescriber | None = None,
        capture_path: Path = Path("capture.jpg"), on_restart: Callable[[], None] | None = None,
        get_display_frame: Callable[[], np.ndarray | None] | None = None,
        ask_agent: "AskAgent | None" = None,
    ) -> None:
        self.registry = registry
        self.overlay = overlay
        self.face_worker = face_worker  # registration reads the latest face embedding from here
        self.state = state              # armed flag + locked objects, persisted across restarts
        self.vocabulary = vocab         # turns what was heard into a name or a YOLO class
        self.telemetry = telemetry      # written here, read by the /IOTCONNECT publisher thread
        self.scene = scene              # None when the VLM is unavailable
        self.capture_path = capture_path  # one file, overwritten: the snapshot the cloud uploads
        self.on_restart = on_restart    # None when nothing can restart us (see _restart)
        self.get_display_frame = get_display_frame or (lambda: None)  # the composed picture, if any
        self.ask_agent = ask_agent      # the LLM on the Ara-240; None when it is off or unreachable
        self.alarm_state = AlarmState.ARMED if state.is_armed else AlarmState.DISARMED
        # _set_state only speaks up on a *change*, so the state we booted into has to be published
        # here - otherwise a demo that is left disarmed never reports an alarm value at all.
        self.telemetry.set(alarm=self.alarm_state.label.lower(), objects="none")
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
            commands.RESTART: self._restart,
            commands.ASK: self._ask,
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
        # Written every frame rather than on change: it is a dict update under a lock, cheaper than
        # working out whether the answer moved. Nothing is sent until the publisher's next tick.
        self.telemetry.set(objects=describe_visible(tracks))

    def on_command(self, command: Command) -> str:
        """Run one command and return what to say. Raises CommandError with a speakable reason.

        Everything runs under one lock, so no two commands mutate the registry at once - except
        `ask`, which is held outside it. An ask sits inside the LLM for tens of seconds and its
        tools come back in through this same method, so taking the lock around it would deadlock
        the demo against itself. `ask.py` allows one question at a time for the same reason.
        """
        handler = self._handlers.get(command.verb)
        if handler is None:  # only reachable from a C2D payload naming a verb we do not have
            raise CommandError(f"I do not know how to {command.verb.replace('_', ' ')}.")
        if command.verb == commands.ASK:
            return handler(command)
        with self._lock:
            return handler(command)

    # --- handlers -----------------------------------------------------------------------------

    def _register_user(self, command: Command) -> str:
        """Bind a name to the most prominent face on screen (the worker holds the live embeddings)."""
        name = self._resolve_new_name(command.argument, command.is_exact)
        if name in self.registry.user_names:
            raise CommandError(f"{name} is already registered. Say unregister user {name} first.")
        if self.face_worker.register_user(name, self._last_tracks) is None:
            raise CommandError(f"I cannot see a face to register as {name}. Please look at the camera.")
        print(f"[app] registered {name!r} (now knows: {', '.join(self.registry.user_names)})")
        return f"Registered {name}."

    def _unregister_user(self, command: Command) -> str:
        name = self._resolve_known_name(command.argument, command.is_exact)
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
        class_name = self._resolve_visible_object(command.argument, command.is_exact)
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
        class_name = (match_exactly(heard, locked) if command.is_exact
                      else vocabulary.match_choice(heard, locked)[0])
        if class_name is None:
            raise CommandError(f"I do not have a {heard or 'thing'} locked. "
                               f"Locked right now: {', '.join(locked)}.")
        self.state.unlock_object(class_name)
        return f"The {class_name} is unlocked."

    def _snapshot(self, command: Command) -> str:
        """Write what the screen is showing to `capture.jpg`, overwriting the last one.

        hyena had to re-render the boxes in OpenCV here, because the compositor will not give the
        picture back: weston 14 only lets a client it launched itself take a screenshot, the board
        has no uinput to fake the Super+S key binding with, and `/dev/fb0` is an unused emulation
        buffer full of zeros. iguana taps the pipeline after the overlay instead, so the real
        composed frame is available and the JPEG is exactly what is on the HDMI screen - the same
        pixels the WebRTC viewer is being sent. The re-render stays as the fallback for when
        nothing is composing (no preview, no streaming) and for the VLM, which wants a different
        picture anyway.

        One file, not one per press: the board is where it is cheap to overwrite and S3 is where
        history is kept - /IOTCONNECT timestamps each upload of `capture.jpg` on the way in.
        """
        composed = self.get_display_frame()
        if composed is not None:
            saved = overlay.save_composed(self.capture_path, composed)
        elif self._last_frame is not None:
            saved = overlay.save_snapshot(self.capture_path, self._last_frame, self._last_tracks,
                                          self.overlay.get_hud_lines())
        else:
            raise CommandError("I do not have a camera frame yet.")
        print(f"[app] snapshot -> {saved}")
        return "Snapshot saved."

    def _describe_scene(self, command: Command) -> str:
        """Hand the current frame, boxes and all, to the VLM. Slow on purpose - see scene.py.

        A C2D `scene` command may carry its own question ("what is the person holding"); voice
        never does, and gets the default "describe what you see". Either way the answer goes to the
        cloud as a one-shot `scene` value as well as being spoken.
        """
        if self.scene is None:
            raise CommandError("The scene describer is not available.")
        if self._last_frame is None:
            raise CommandError("I do not have a camera frame yet.")
        description = self.scene.describe(self._last_frame, self._last_tracks, command.argument)
        self.telemetry.set_once(scene=description)
        return description

    def _ask(self, command: Command) -> str:
        """Hand a plain-English question to the LLM on the Ara-240. Slow on purpose - see ask.py.

        The whole sentence is the argument: unlike every other command here, nothing was parsed out
        of it. What comes back may be an answer ("the alarm is armed and I can see Nick") or the
        result of the model having *done* something - its tools run these same handlers, so
        "please disarm the alarm" arrives here and leaves through `_disarm`.

        The answer also goes out as a one-shot `answer` telemetry attribute, because a C2D ack is
        capped at 200 characters and the model is not reliably that brief.
        """
        if self.ask_agent is None:
            raise CommandError("The assistant is not available. Start the connector on the Ara-240.")
        question = (command.argument or command.text).strip()
        if not question:
            raise CommandError("What would you like to ask?")
        answer = self.ask_agent.ask(question)
        self.telemetry.set_once(answer=answer)
        return answer

    def _restart(self, command: Command) -> str:
        """Restart the process, keeping the same arguments. Not spoken - C2D only (commands.py).

        The demo has to come back by itself: eIQ stops the process an hour after the models load,
        and the booth cannot be the thing that notices. Everything a visitor set is on disk
        (`faces.json`, `state.json`), so a restart costs the ~15 s of loading and nothing else.
        The actual exec is deferred by `main.py` - this handler has to *return* first so its
        acknowledgement reaches /IOTCONNECT before the process goes away.
        """
        if self.on_restart is None:
            raise CommandError("Restart is not available in this configuration.")
        self.on_restart()
        return "Restarting."

    def set_ask_agent(self, ask_agent: "AskAgent | None") -> None:
        """Give the app its LLM after the fact - the one back-edge in the object graph.

        The agent's tools run these handlers, so it cannot be built before the app that owns them.
        `main.py` builds the app, then the agent, then hands it back here.
        """
        self.ask_agent = ask_agent

    def get_status(self) -> str:
        """One line of ground truth, for the model to read before it answers a question about us.

        Prose rather than JSON: it goes into a 4096-token prompt shared with the tool schemas and
        the answer, and the model reads a sentence at least as well as a dict.
        """
        return (f"The alarm is {self.alarm_state.label.lower()}. "
                f"Registered users: {', '.join(self.registry.user_names) or 'none'}. "
                f"Guarded objects: {', '.join(self.state.locked_objects) or 'none'}. "
                f"The camera can see: {describe_visible(self._last_tracks)}.")

    # --- matching an argument to what exists ----------------------------------------------------
    # Two rules, chosen by `command.is_exact` - the contract the *source* offered (commands.py).
    # Spoken: guess phonetically, because the recogniser mangles words and asking a visitor to
    # repeat themselves four times is the demo failing. Typed: match literally and say what exists,
    # because a name that came from the back end is already correct, and guessing at it deletes
    # the wrong user. The phonetic path also shells out to espeak per candidate (~28 ms each),
    # which a typed command has no reason to pay.

    def _resolve_new_name(self, heard: str, is_exact: bool = False) -> str:
        """A name for a NEW user: a vocabulary entry if it matches one, otherwise what was heard.

        Registering someone who is not in vocabulary.json has to work - that is the whole live
        demo - so an unmatched name is accepted as spoken. Adding them to vocabulary.json
        afterwards is what makes the *next* recognition of that name reliable.

        The strict threshold matters here: loose matching would quietly register Michael as Marija,
        because the two are not that far apart as sounds. A recorded `heard_as` still scores 1.0.
        A typed name is simply taken as given - there is nothing to improve on it.
        """
        if not heard:
            raise CommandError("Who should I register? I need a name.")
        if is_exact:
            return heard.strip()
        name, score = vocabulary.match_name(heard, self.vocabulary,
                                            vocabulary.STRICT_MATCH_THRESHOLD)
        if name is not None:
            print(f"[app] heard {heard!r} -> known name {name} ({score:.2f})")
            return name
        return heard.title()

    def _resolve_known_name(self, heard: str, is_exact: bool = False) -> str:
        """A name that must already be registered: matched against the face database itself."""
        known = self.registry.user_names
        if not known:
            raise CommandError("Nobody is registered yet.")
        if not heard:
            raise CommandError(f"Who should I remove? I know {', '.join(known)}.")
        name = match_exactly(heard, known) if is_exact else vocabulary.match_choice(heard, known)[0]
        if name is None:
            raise CommandError(f"I do not know anyone called {heard}. I know {', '.join(known)}.")
        print(f"[app] {heard!r} -> registered user {name}")
        return name

    def _resolve_visible_object(self, heard: str, is_exact: bool = False) -> str:
        """A YOLO class name that is on screen right now. People are not lockable objects."""
        visible = [name for name in self.registry.get_visible_classes(self._last_tracks)
                   if name != "person"]
        if not heard:
            raise CommandError("What should I lock? I need the name of an object.")
        if not visible:
            raise CommandError(f"I cannot see a {heard}. There is nothing on screen I could lock.")
        if is_exact:
            class_name = match_exactly(heard, visible)  # a typed 'laptop' is a YOLO class already
        else:
            # vocabulary.json maps spoken words ("phone") to YOLO classes ("cell phone"); anything
            # not listed there falls back to matching against what the tracker is reporting.
            class_name, _ = vocabulary.match_object(heard, self.vocabulary)
            if class_name is None or class_name not in visible:
                class_name, _ = vocabulary.match_choice(heard, visible)
        if class_name is None or class_name not in visible:
            raise CommandError(f"I cannot see a {heard}. I can see {', '.join(visible)}.")
        print(f"[app] {heard!r} -> visible object {class_name}")
        return class_name

    def _set_state(self, new_state: AlarmState) -> None:
        """The single place alarm state changes - so every transition has one spot that announces it.

        And exactly one spot that tells the cloud. `wake()` rather than waiting for the 4-second
        tick: an alarm that shows up in the dashboard three seconds late is a different demo.
        """
        if new_state is self.alarm_state:
            return
        self.alarm_state = new_state
        print(f"[app] alarm state -> {new_state.label}")
        self.telemetry.set(alarm=new_state.label.lower())
        self.telemetry.wake()


def match_exactly(given: str, candidates: list[str]) -> str | None:
    """The typed-argument rule: case-insensitive equality, nothing else. `None` when it is not there.

    Deliberately dull. The whole point is that a command from the dashboard acts on the thing it
    named or on nothing at all, and that the failure says which names do exist.
    """
    given = given.strip().lower()
    return next((candidate for candidate in candidates if candidate.lower() == given), None)


def describe_visible(tracks: list[Track]) -> str:
    """'Nick, person, laptop' - what is on screen, named where we recognise the face.

    Duplicates are kept: two unrecognised people read as "person, person", which is the honest
    answer and is what makes the dashboard row match the picture.
    """
    return ", ".join(track.identity or track.class_name for track in tracks) or "none"
