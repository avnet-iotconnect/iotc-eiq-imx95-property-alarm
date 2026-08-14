"""The watchdog: three flags, two timers, and everything that decides when the alarm has a reason.

`app.py` answers "what happens when somebody says something". This file answers the other half -
"what happens when nobody says anything and the camera keeps running" - and it is deliberately a
separate file, because the two read very differently. A command is a straight line from a sentence
to an answer. This is a state machine driven by a clock.

**Three flags, not one state.** This started out as a single ALARM state that was simply "armed and
a stranger is on screen this instant", so it flickered with YOLO and forgot everything the moment
the stranger stepped out of shot. The redesign splits that into three things that change
independently:

    arming     the user's own switch, and *only* that (`state.is_armed`, on disk so that it survives
               the restart, and disarmed until somebody arms it). "Alarm" here means the switch - it
               is an input, never a thing on fire. Disarmed, people are nobody's business and only
               locked objects are watched; armed, the room is guarded and an intruder is
               photographed for as long as they stay
    alert      the event log is not empty. There is no separate flag: an alert *is* a log with
               something in it, which is why clearing either one clears the other
    recording  a *timed* flag. Refreshed by anything still happening, and it lasts RECORDING_HOLD_S
               after the last refresh. While it is up, the screen says REC and a screenshot goes to
               the cloud every RECORDING_INTERVAL_S

**What the screen shows is the recording, not the alert.** REC in the corner means something is
happening *now*; the alert is one HUD line saying what happened and how long ago. There is no big
red banner - it could not tell those two apart, so a recognised user ended up looking at a permanent
ALARM with nothing on the screen to explain it.

**What raises an alert**, and each one is a paragraph in the code below rather than a table here:

- an unrecognised person in view while armed, for INTRUDER_GRACE_S - the grace is for the face
  recogniser, which needs a moment and a look at the face before it can say "that is Nick";
- a locked object moved by nobody the camera recognises;
- a locked object that has disappeared.

**A recognised user excuses all three**, armed or not: an unknown person is fine as long as a known
one is there. That is one rule with one window, `USER_GRACE_S`, and it is what makes a booth
workable - the visitors are strangers, and the person showing them round is not.

The last two come from `guard.py` and they are **not gated on arming**. That is the owner's call and
it is the right one for a booth: locking a laptop is an explicit act, and an explicit act should not
also require remembering to arm. Arming is about *people* being where they should not be.

**Two ways to clear it.** `clear alert` empties the log and nothing else - the acknowledgement, for
when somebody has read what happened and wants the screen back. **Disarming** is the bigger reset:
the log, the recording, and every locked object re-anchored where it now stands. Disarming works
even when the alarm is already disarmed, because a locked object raises an alert either way.

**Threading.** `on_frame` runs on the video loop and touches nothing else does; `lock`/`unlock`/
`arm`/`disarm` run on a command thread under `app.py`'s command lock. The shared state is small
dicts and floats, and every mutation is a single assignment - the same reasoning as `face_worker.py`
uses, and the reason there is no lock here. The one thing that must never happen on the video loop
is the *upload*, so `on_capture` submits a snapshot command and returns; `main.py` wires it.
"""

from __future__ import annotations

from time import monotonic
from typing import Callable

from applib import eventlog
from applib.eventlog import EventLog
from applib.guard import ObjectGuard
from applib.registry import Registry
from applib.state import SessionState
from applib.tracking import Track

# How long a sighting still counts for. The tracker reports only what it saw *this* frame and YOLO
# drops a person for the odd frame, so asking "is somebody on screen right now" makes an empty room
# flicker several times a second. Remembering the last sighting for a moment fixes it with no state
# machine to speak of.
PERSON_MEMORY_S = 1.0
# ... and how long a *recognised* user's presence excuses everything - a stranger standing beside
# them, and whatever happens to a locked object. One window for both, because it is one rule: an
# unknown person is fine as long as a known one is there, whether the alarm is armed or not. Five
# seconds is USER_GOAL.md's, and the slack matters - the owner may pick a laptop up and turn away
# from the camera while doing it, and the face worker cannot recognise the back of a head.
USER_GRACE_S = 5.0

INTRUDER_GRACE_S = 3.0    # an unrecognised face gets this long to be recognised before it is one

RECORDING_HOLD_S = 5.0    # a recording outlives its last refresh by this long
RECORDING_INTERVAL_S = 3.0  # ... and takes a screenshot this often while it runs (<= the hold)


class Watchdog:
    """Arming, alert and recording - the alarm's memory, driven one frame at a time."""

    def __init__(
        self, registry: Registry, state: SessionState, guard: ObjectGuard, log: EventLog,
        on_capture: Callable[[], None] | None = None,
    ) -> None:
        self.registry = registry      # who is on screen: the facts, per GUIDELINES.md
        self.state = state            # the armed flag and the locks, both persisted
        self.guard = guard            # has a locked object moved, or gone?
        self.log = log                # the two most recent events of each kind
        self.on_capture = on_capture  # take a screenshot: submitted as a command, never done here
        self.is_recording = False
        self._recording_until = 0.0
        self._captured_at = float("-inf")
        self._person_seen_at = float("-inf")
        self._user_seen_at = float("-inf")
        self._stranger_since: float | None = None
        self._is_stranger_reported = False

    @property
    def is_alert(self) -> bool:
        """Whether anything is in the event log. That is the whole definition of the word here."""
        return not self.log.is_empty

    # --- the frame loop -------------------------------------------------------------------------

    def on_frame(self, now: float, tracks: list[Track]) -> None:
        """One frame's worth of watching. Never blocks: the slow part is a command someone else runs."""
        if self.registry.is_person_present(tracks):
            self._person_seen_at = now
        if self.registry.is_registered_user_present(tracks):
            self._user_seen_at = now

        self._check_intruder(now)
        is_user_present = now - self._user_seen_at < USER_GRACE_S
        for kind, description in self.guard.check(now, tracks, is_user_present):
            self._trigger(now, kind, description)
        self._keep_recording(now)

    def _check_intruder(self, now: float) -> None:
        """Armed, somebody there, and the face recogniser has had its three seconds and found nobody.

        Reported once per visit, not once per frame - `_is_stranger_reported` is cleared only when
        the condition goes away, which for the person on screen means actually leaving. While they
        are still there the recording keeps being refreshed, so a visitor who stays for a minute is
        photographed for the whole minute.
        """
        is_stranger = (self.state.is_armed
                       and now - self._person_seen_at < PERSON_MEMORY_S
                       and now - self._user_seen_at >= USER_GRACE_S)
        if not is_stranger:
            self._stranger_since = None
            self._is_stranger_reported = False
            return
        if self._stranger_since is None:
            self._stranger_since = now
        if now - self._stranger_since < INTRUDER_GRACE_S:
            return
        if self._is_stranger_reported:
            self._refresh_recording(now)
        else:
            self._is_stranger_reported = True
            self._trigger(now, eventlog.INTRUDER, "an unrecognised person came into view")

    def _trigger(self, now: float, kind: str, description: str) -> None:
        """One event: log it - which *is* raising the alert - start recording, and take the picture."""
        self.log.add(kind, description, now)
        print(f"[watchdog] {kind}: {description}")
        self._refresh_recording(now)
        self._capture(now)

    # --- recording ------------------------------------------------------------------------------

    def _refresh_recording(self, now: float) -> None:
        if not self.is_recording:
            print(f"[watchdog] recording for {RECORDING_HOLD_S:.0f}s, a screenshot every "
                  f"{RECORDING_INTERVAL_S:.0f}s")
            self.is_recording = True
        self._recording_until = now + RECORDING_HOLD_S

    def _keep_recording(self, now: float) -> None:
        """Take the periodic screenshot, or let the recording lapse when nothing refreshed it."""
        if not self.is_recording:
            return
        if now >= self._recording_until:
            self.is_recording = False
            print("[watchdog] recording stopped")
            return
        if now - self._captured_at >= RECORDING_INTERVAL_S:
            self._capture(now)

    def _capture(self, now: float) -> None:
        """Ask for a screenshot. This is one `snapshot` command, run and uploaded by a command worker.

        Going through the command path rather than saving a file here is what makes a watchdog
        screenshot the same thing as a screenshot anybody else asked for - the same picture, the same
        upload, the same line in the log - and it keeps the ~100 KB upload off the video loop.
        """
        self._captured_at = now
        if self.on_capture is not None:
            self.on_capture()

    def set_capture(self, on_capture: Callable[[], None]) -> None:
        """Give the watchdog its screenshot button after the fact: it is a command, and commands
        cannot be submitted until the service exists, which cannot happen until the app does.
        """
        self.on_capture = on_capture

    # --- what the commands do to it ---------------------------------------------------------------

    def arm(self) -> None:
        """Start counting from now: whoever is standing in front of the camera gets the full grace."""
        self._forget_stranger()

    def _forget_stranger(self) -> None:
        self._stranger_since = None
        self._is_stranger_reported = False

    def clear_alert(self) -> bool:
        """Empty the event log - the acknowledgement. True if there was anything in it.

        Deliberately *only* the log. The recording is not stopped, because it is a statement about
        what is happening right now rather than about what has been read; it lapses on its own
        within RECORDING_HOLD_S if nothing refreshes it. Anchors are not touched either - that is
        what disarming is for. And the intruder is not re-reported the instant this is said: whoever
        is on screen has to leave and come back first.
        """
        was_raised = self.is_alert
        self.log.clear()
        return was_raised

    def disarm(self) -> bool:
        """Clear the log, the recording and every anchor. True if there was something to clear.

        The return value is the difference between "Alarm disarmed" and "there was nothing to
        disarm": disarming twice is a real thing a person does when they cannot remember whether
        they already said it.
        """
        was_raised = self.is_alert or self.is_recording
        self.is_recording = False
        self._recording_until = 0.0
        self.log.clear()
        self.guard.reset()
        self._forget_stranger()
        return was_raised

    def lock(self, class_name: str, anchor: list[int] | None) -> bool:
        """Guard an object where it is standing now. False if it was already locked and is re-anchored.

        Re-locking is not refused: it is how the user says "it lives *here* now" after the thing has
        been moved, and it is one of the three ways out of a disturbed lock (the others are
        unlocking and disarming).
        """
        is_new = self.state.lock_object(class_name, anchor)
        if not is_new:
            self.state.set_anchor(class_name, anchor)
        self.guard.forget(class_name)
        return is_new

    def unlock(self, class_name: str) -> bool:
        is_unlocked = self.state.unlock_object(class_name)
        self.guard.forget(class_name)
        return is_unlocked

    # --- what it looks like ------------------------------------------------------------------------

    def describe_locks(self) -> list[str]:
        """The locked objects for the HUD, with anything missing marked - 'cell phone (gone)'."""
        return self.guard.describe()

    def describe_alert(self) -> str:
        """The alert in one HUD line: the newest event, its age, and how many others there are.

        '' when the log is empty, which the overlay draws as 'alert: -'. The whole log is a spoken
        question away (`get_status`) - this line only has to answer "has anything happened?".
        """
        newest = self.log.get_newest()
        if newest is None:
            return ""
        others = len(self.log.get_events()) - 1
        age = eventlog.describe_age(monotonic() - newest.at)
        return f"{newest.description}, {age}" + (f"  (+{others} more)" if others else "")

    def get_log(self) -> str:
        """The whole event log as a sentence, for `describe alert` - spoken, and sent to the cloud."""
        if not self.is_alert:
            return "Nothing has been logged since the alert was last cleared."
        return f"Here is what happened: {self.log.get_log(monotonic())}."

    def get_status(self) -> str:
        """The watchdog half of `app.get_status`: has anything happened, and what.

        Short on purpose - it goes into the LLM's 4096-token prompt beside the tool schemas.
        """
        if not self.is_alert:
            return "Nothing has been logged."
        recording = " Recording now." if self.is_recording else ""
        return f"Events since the log was last cleared: {self.log.get_log(monotonic())}.{recording}"
