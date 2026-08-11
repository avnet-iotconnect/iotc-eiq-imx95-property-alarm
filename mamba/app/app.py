"""AntiTheftApp: the one place decisions get made. Everything else just feeds it or is called by it.

This is the composition-root pattern: the app takes its collaborators (`registry`, `overlay`,
`face_worker`, `state`, `scene`) as plain constructor arguments and calls out to them. It never
imports `main`, never constructs its own modules, and nothing calls back *into* it except two
methods:

    on_frame(frame, tracks)   - every camera frame: watch it, then show it        [video thread]
    on_command(command)       - one parsed command, from voice, the file or C2D   [command thread]

Reading those two methods top to bottom is the whole behaviour of the demo.

**The LLM is a command source, not a decision maker.** `agent` hands a sentence to the 7B model on
the Ara-240 (`agent.py`), whose tools come back in through `on_command` - so the model can arm the
alarm, but it never decides *whether* the alarm should go off. That decision is deterministic code
in `watchdog.py`, next door.

**Deciding when the alarm has a reason is `watchdog.py`'s job, not this file's.** `on_frame` hands
it the frame's tracks and then reads three flags back - armed, alert, recording - and paints them.
The split is by how the code reads: a command is a straight line from a sentence to an answer, and
the watchdog is a clock-driven state machine. Mixing them was what made koala's alarm hard to
follow.

**The cloud is not a collaborator here.** `iotc.py` is nowhere in this file. What the dashboard
needs is *written* into `telemetry.TelemetryState` - a plain dict behind a lock - and the
/IOTCONNECT publisher reads it on its own thread. So the alarm logic never blocks on a network call,
and the whole video half still runs with the SDK not installed. The one thing this file does ask the
cloud to *do* - upload a snapshot - arrives as a plain callable (`upload_capture`), so uploading
stays one line here and every line of S3 stays in `iotc.py`.

**Handlers return the sentence to say, or raise `CommandError`.** That is the entire contract with
`commands.py`, and it is what makes one implementation serve both voice (which speaks the text) and
a future C2D message (which acks with it). Handlers are the layer that knows what *exists* - which
users are registered, what is on screen - so this is where a heard word ("money", "computer") is
matched to a real thing (Marija, laptop); `commands.parse` deliberately does none of that.

**Threading.** `on_command` runs on the command pool, `on_frame` on the video loop. One lock
serialises commands against each other, so no two handlers mutate the registry at once. `on_frame`
takes no lock: what it reads (an armed flag, a list of locked names) is slow-changing enough that
one frame's stale value is harmless, and what the watchdog writes on that thread is a single
assignment at a time - the same reasoning `face_worker.py` uses.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import TYPE_CHECKING, Callable

import numpy as np

from app.watchdog import Watchdog
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

if TYPE_CHECKING:  # imported for the type only: agent.py pulls in strands-agents, which may not be there
    from app.agent import AgentService

# Registration: how long the person is given to look at the camera, and how many goes they get.
# The first countdown is short because it starts *after* the command was recognised, and the person
# has been waiting through that already. A rejected look adds a second, on the grounds that
# somebody who has just been told what to fix needs a moment to read it and do it.
REGISTER_COUNTDOWN_S = 2.0
REGISTER_EXTRA_S = 1.0
REGISTER_ATTEMPTS = 3


class AntiTheftApp:
    """The command handlers + the overlay, driven entirely by on_frame / on_command."""

    def __init__(
        self, registry: Registry, overlay: Overlay, face_worker: FaceWorker, state: SessionState,
        watchdog: Watchdog, vocab: Vocabulary, telemetry: TelemetryState,
        scene: SceneDescriber | None = None,
        capture_path: Path = Path("capture.jpg"), on_restart: Callable[[], None] | None = None,
        get_display_frame: Callable[[], np.ndarray | None] | None = None,
        agent: "AgentService | None" = None,
        upload_capture: Callable[[], str] | None = None,
    ) -> None:
        self.registry = registry
        self.overlay = overlay
        self.face_worker = face_worker  # registration takes (and judges) its look at a face here
        self.state = state              # armed flag + locked objects, persisted across restarts
        self.watchdog = watchdog        # decides when the alarm has a reason; this file only shows it
        self.vocabulary = vocab         # turns what was heard into a name or a YOLO class
        self.telemetry = telemetry      # written here, read by the /IOTCONNECT publisher thread
        self.scene = scene              # None when the VLM is unavailable
        self.capture_path = capture_path  # one file, overwritten: the snapshot the cloud uploads
        self.on_restart = on_restart    # None when nothing can restart us (see _restart)
        self.get_display_frame = get_display_frame or (lambda: None)  # the composed picture, if any
        self.agent = agent              # the LLM on the Ara-240; None when it is off or unreachable
        self.upload_capture = upload_capture  # the cloud's uploader; None when there is no cloud
        self.alarm_state = AlarmState.ARMED if state.is_armed else AlarmState.DISARMED
        self.reported_alarm = self.alarm_state.label.lower()  # the cloud's rollup; see _set_state
        # _set_state only speaks up on a *change*, so the state we booted into has to be published
        # here - otherwise a demo that is left disarmed never reports an alarm value at all.
        self.telemetry.set(alarm=self.reported_alarm, objects="none")
        self._last_tracks: list[Track] = []
        self._last_frame: np.ndarray | None = None
        self._lock = Lock()

        self._handlers = {
            commands.REGISTER_USER: self._register_user,
            commands.UNREGISTER_USER: self._unregister_user,
            commands.UNREGISTER_LAST: self._unregister_last,
            commands.ARM: self._arm,
            commands.DISARM: self._disarm,
            commands.CLEAR_ALERT: self._clear_alert,
            commands.DESCRIBE_ALERT: self._describe_alert,
            commands.LOCK_OBJECT: self._lock_object,
            commands.UNLOCK_OBJECT: self._unlock_object,
            commands.DESCRIBE_SCENE: self._describe_scene,
            commands.SNAPSHOT: self._snapshot,
            commands.RESTART: self._restart,
            commands.AGENT: self._agent,
        }

    # --- the two event methods ----------------------------------------------------------------

    def on_frame(self, frame: np.ndarray, tracks: list[Track]) -> None:
        """Each frame: let the watchdog look at it, then show what it made of it.

        The three flags stay three things on the screen. `alarm:` is the switch and nothing else -
        armed or disarmed, never "ALARM". What the demo has *caught* is the `alert:` line, and
        whether something is happening right now is REC in the corner. An alert can be up with the
        alarm disarmed, because a locked object is watched either way.
        """
        self._last_frame = frame
        self._last_tracks = tracks
        self.watchdog.on_frame(monotonic(), tracks)

        self._set_state()
        self.overlay.set_tracks(tracks)
        self.overlay.set_alarm_state(self.alarm_state)
        self.overlay.set_alert(self.watchdog.describe_alert())
        self.overlay.set_locked_objects(self.watchdog.describe_locks())
        self.overlay.set_recording(self.watchdog.is_recording)
        # Written every frame rather than on change: it is a dict update under a lock, cheaper than
        # working out whether the answer moved. Nothing is sent until the publisher's next tick.
        self.telemetry.set(objects=describe_visible(tracks))

    def on_command(self, command: Command) -> str:
        """Run one command and return what to say. Raises CommandError with a speakable reason.

        Everything runs under one lock, so no two commands mutate the registry at once - except
        `agent`, which is held outside it. A question sits inside the LLM for tens of seconds and its
        tools come back in through this same method, so taking the lock around it would deadlock
        the demo against itself. `agent.py` allows one question at a time for the same reason.

        `register_user` is the one handler that *holds* the lock for a while - it counts the person
        down in front of the camera first. That is the intended behaviour: while somebody is being
        registered, the next command waits its turn rather than arriving in the middle of it.
        """
        handler = self._handlers.get(command.verb)
        if handler is None:  # only reachable from a C2D payload naming a verb we do not have
            raise CommandError(f"I do not know how to {command.verb.replace('_', ' ')}.")
        if command.verb == commands.AGENT:
            return handler(command)
        with self._lock:
            return handler(command)

    # --- handlers -----------------------------------------------------------------------------

    def _register_user(self, command: Command) -> str:
        """Count the person down, take a good look at their face, and bind a name to it.

        Registration is the one command that takes *time on purpose*. Spoken, it arrives seconds
        after the person said it - the wake word, the recogniser, the transcript being read back -
        and it used to bind whatever the camera happened to be seeing at that instant, which was
        usually somebody looking at the screen to find out whether anything had happened. So the
        screen now says when to look and for how long, and the face is only stored if the look was
        good enough (`face.find_quality_problem`).

        A rejected look restarts the countdown a second longer, with the reason where the person can
        read it: "Please come closer to the camera - registering Nick  3". Three goes, then it gives
        up and says why, because a fourth is not going to be the one that works.

        Every attempt logs what it measured. The `nearest` figure in that line is the one to read
        when identity looks wrong: it says which *already registered* user this new face is most
        like, so a new person scoring high against an old one is a mix-up caught in the act. No
        threshold here can catch that - it is the embedder failing, not the pose.

        This runs on a command thread with the command lock held, so the demo takes no other command
        for the up-to-nine seconds it can last. The video loop is untouched - it takes no lock - so
        the picture, the alarm and the stream all carry on at 30 fps while the count runs.
        """
        name = self._resolve_new_name(command.argument, command.is_exact)
        if name in self.registry.user_names:
            raise CommandError(f"{name} is already registered. Say unregister user {name} first.")
        problem = None
        for attempt in range(REGISTER_ATTEMPTS):
            message = (f"Registering {name} - look at the camera" if problem is None
                       else f"{problem} - registering {name}")
            self._count_down(message, REGISTER_COUNTDOWN_S + attempt * REGISTER_EXTRA_S)
            result = self.face_worker.try_register_user(name, self._last_tracks)
            print(f"[app] register {name!r} {attempt + 1}/{REGISTER_ATTEMPTS}: {result.report}  "
                  f"nearest {result.nearest}  ->  {result.problem or 'registered'}")
            if result.is_registered:
                print(f"[app] registered {name!r} (now knows: {', '.join(self.registry.user_names)})")
                return f"Registered {name}."
            problem = result.problem
        raise CommandError(f"{problem}. I did not get a good enough look to register {name}.")

    def _count_down(self, message: str, seconds: float) -> None:
        """Put a message and a ticking number on the screen, and wait for it to run out.

        The waiting is here and the drawing is in the overlay, on the display thread - so this is a
        plain sleep, and the digit on the screen still ticks. The print is for the booth's other
        screen: with `--no-preview` and nobody streaming, the console is the only OSD there is.
        """
        print(f"[app] {message} ({seconds:.0f}s)")
        self.overlay.start_countdown(message, seconds)
        sleep(seconds)
        self.overlay.stop_countdown()

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
        self.watchdog.arm()  # the person who just said it gets the full grace, not what is left of it
        self._set_state()
        return "Alarm armed."

    def _disarm(self, command: Command) -> str:
        """Disarm, and clear whatever the watchdog is holding - the demo's big reset.

        Disarming an already-disarmed alarm is not a mistake to be corrected: a locked object is
        watched whether or not the alarm is armed, so an alert can be up with the alarm off. Say
        `clear the alert` for the narrow version, which leaves the locks anchored where they are.
        """
        was_armed = self.state.is_armed
        if self.watchdog.is_alert:
            self._publish_log(self.watchdog.get_log())  # disarming destroys it too - send it first
        self.state.set_armed(False)
        is_cleared = self.watchdog.disarm()
        self._set_state()  # do not wait for the next frame to update the HUD and the cloud
        if is_cleared:
            return "Alarm disarmed, and the alert is cleared."
        if not was_armed:
            return "The alarm is already disarmed."
        return "Alarm disarmed."

    def _describe_alert(self, command: Command) -> str:
        """Read the event log back: what happened, in order, with ages rather than timestamps.

        The whole log goes to the cloud as `answer` as well as being returned. It has to: several
        events in plain English is far more than a C2D acknowledgement can carry (the dashboard
        renders one as a tooltip), so the ack is the first line of it and this is the rest.
        """
        return self._publish_log(self.watchdog.get_log())

    def _clear_alert(self, command: Command) -> str:
        """Empty the event log, and with it the alert. The acknowledgement, without disarming.

        Its own command because the alarm and the alert are different things: the alarm is the
        switch the user throws, the alert is what the demo has caught. Before this existed, an alert
        raised by a locked object - which is watched whether or not the alarm is armed - could only
        be cleared by saying "disarm", which is a strange thing to say about an alarm that was never
        armed.

        **The log is published before it is destroyed**, whichever way the clearing was asked for -
        voice, dashboard, or the LLM deciding to call the tool. It only ever existed in RAM, so
        clearing without sending it is the one way to lose it for good.
        """
        if not self.watchdog.is_alert:
            return "There is nothing to clear."
        self._publish_log(self.watchdog.get_log())
        self.watchdog.clear_alert()
        return "Alert cleared."

    def _publish_log(self, description: str) -> str:
        """Send a copy of the event log to the cloud as `answer`, and hand it back to say aloud.

        `set_once` queues rather than overwrites (see telemetry.py), which is what makes this safe
        when the *LLM* is the one clearing the alert: the log goes out, and the model's own reply
        follows it in the next message instead of replacing it.
        """
        self.telemetry.set_once(answer=description)
        return description

    def _lock_object(self, command: Command) -> str:
        """Guard an object, anchored to the box it is standing in right now.

        Locking it again is not refused - it is how somebody says "it lives *here* now" after the
        thing has been moved, and re-locking is one of the three ways to clear a disturbed lock
        (the others are unlocking and disarming). See `watchdog.py` and `guard.py`.
        """
        class_name = self._resolve_visible_object(command.argument, command.is_exact)
        anchor = largest_box(self._last_tracks, class_name)
        if self.watchdog.lock(class_name, anchor):
            return f"The {class_name} is locked. I am watching it."
        return f"The {class_name} is locked again, where it is now."

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
        self.watchdog.unlock(class_name)
        return f"The {class_name} is unlocked."

    def _snapshot(self, command: Command) -> str:
        """Write what the screen is showing to `capture.jpg` and upload it, overwriting the last one.

        Saving and uploading are one command because to everyone who asks for one - the dashboard
        button, a spoken "take a screenshot", the LLM's `take_screenshot` tool - a picture that
        stayed on the board is not a screenshot at all. `upload_capture` is `iotc.py`'s, handed over
        by `main.py`; with no cloud it is None and the file is simply written.

        Re-rendering the boxes in OpenCV was once the only way to do this, because the compositor
        will not give the picture back: weston 14 only lets a client it launched itself take a
        screenshot, the board has no uinput to fake the Super+S key binding with, and `/dev/fb0` is
        an unused emulation buffer full of zeros. Tapping the pipeline after the overlay gets round
        all of that: the composed frame is available and the JPEG is exactly what is on screen - the same
        pixels the WebRTC viewer is being sent. The re-render stays as the fallback for when
        nothing is composing (no preview, no streaming) and for the VLM, which wants a different
        picture anyway.

        One file, not one per press: the board is where it is cheap to overwrite and S3 is where
        history is kept - /IOTCONNECT timestamps each upload of `capture.jpg` on the way in.

        The upload happens with the command lock held, so a slow one delays the *next* command by
        as long as it takes. That is a hundred kilobytes over the booth's network, and paying it
        here is what lets one command mean one finished thing.
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
        if self.upload_capture is None:
            return "Snapshot saved."
        return self.upload_capture()

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

    def _agent(self, command: Command) -> str:
        """Hand a plain-English question to the LLM on the Ara-240. Slow on purpose - see agent.py.

        The whole sentence is the argument: unlike every other command here, nothing was parsed out
        of it. What comes back may be an answer ("the alarm is armed and I can see Nick") or the
        result of the model having *done* something - its tools run these same handlers, so
        "please disarm the alarm" arrives here and leaves through `_disarm`.

        The whole answer is returned - voice would speak it, and it goes to the cloud as a one-shot
        `answer` telemetry attribute, once rather than on every tick. Only the C2D acknowledgement
        is shortened, in `iotc.py`, because the dashboard renders an ack as a tooltip.
        """
        if self.agent is None:
            raise CommandError("The assistant is not available. Start the connector on the Ara-240.")
        question = (command.argument or command.text).strip()
        if not question:
            raise CommandError("What would you like to ask?")
        answer = self.agent.ask(question)
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

    def set_agent(self, agent: "AgentService | None") -> None:
        """Give the app its LLM after the fact - one of two back-edges in the object graph.

        The agent's tools run these handlers, so it cannot be built before the app that owns them.
        `main.py` builds the app, then the agent, then hands it back here.
        """
        self.agent = agent

    def set_upload_capture(self, upload_capture: Callable[[], str] | None) -> None:
        """Give the app a way to upload `capture.jpg` - the other back-edge, for the same reason.

        `iotc.py` owns the upload, and the /IOTCONNECT client cannot be built before the command
        service, which cannot be built before this app. It returns what to say ("Snapshot
        uploaded.") or raises `CommandError` like any handler would.
        """
        self.upload_capture = upload_capture

    def get_status(self) -> str:
        """One line of ground truth, for the model to read before it answers a question about us.

        Prose rather than JSON: it goes into a 4096-token prompt shared with the tool schemas and
        the answer, and the model reads a sentence at least as well as a dict.
        """
        return (f"The alarm is {'armed' if self.state.is_armed else 'disarmed'}. "
                f"Registered users: {', '.join(self.registry.user_names) or 'none'}. "
                f"Guarded objects: {', '.join(self.watchdog.describe_locks()) or 'none'}. "
                f"The camera can see: {describe_visible(self._last_tracks)}. "
                f"{self.watchdog.get_status()}")

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

    def _set_state(self) -> None:
        """The single place alarm state changes - so every transition has one spot that announces it.

        Two audiences, one decision. **The screen** gets the switch and only the switch: ARMED or
        DISARMED, because what is happening *now* is REC in the corner and what has happened is the
        alert line. **The cloud** gets a rollup in the same `alarm` attribute it always had -
        `disarmed` / `armed` / `alarm` - where `alarm` means the same thing REC does: something is
        going on this minute. A dashboard is not sitting next to the screen and has room for one
        word, so that word is the interesting one.

        `wake()` rather than waiting for the 4-second tick: an alarm that reaches the dashboard three
        seconds late is a different demo.
        """
        state = AlarmState.ARMED if self.state.is_armed else AlarmState.DISARMED
        reported = "alarm" if self.watchdog.is_recording else state.label.lower()
        if state is self.alarm_state and reported == self.reported_alarm:
            return
        self.alarm_state, self.reported_alarm = state, reported
        print(f"[app] alarm state -> {reported}")
        self.telemetry.set(alarm=reported)
        self.telemetry.wake()


def match_exactly(given: str, candidates: list[str]) -> str | None:
    """The typed-argument rule: case-insensitive equality, nothing else. `None` when it is not there.

    Deliberately dull. The whole point is that a command from the dashboard acts on the thing it
    named or on nothing at all, and that the failure says which names do exist.
    """
    given = given.strip().lower()
    return next((candidate for candidate in candidates if candidate.lower() == given), None)


def largest_box(tracks: list[Track], class_name: str) -> list[int] | None:
    """The biggest box of that class on screen - the one being pointed at when somebody says "lock".

    None when there is nothing to anchor to, which the guard reads as "anchor it the first time you
    see it" rather than as an error.
    """
    boxes = [track.box for track in tracks if track.class_name == class_name]
    return max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])) if boxes else None


def describe_visible(tracks: list[Track]) -> str:
    """'Nick, person, laptop' - what is on screen, named where we recognise the face.

    Duplicates are kept: two unrecognised people read as "person, person", which is the honest
    answer and is what makes the dashboard row match the picture.
    """
    return ", ".join(track.identity or track.class_name for track in tracks) or "none"
