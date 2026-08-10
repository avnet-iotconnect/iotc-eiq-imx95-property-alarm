"""What the device wants the cloud to know, in one small object nobody has to wait on.

The problem this solves: telemetry is produced on three threads that must never block - the 30 fps
video loop (fps, what is on screen), the command threads (a VLM scene description), the alarm state
machine - and it is *sent* on exactly one, the /IOTCONNECT publisher in `iotc.py`. Having producers
call the MQTT client directly would put a network call on the frame loop and would make `app.py`
depend on the cloud.

So producers only ever write here, and the publisher reads here. Two kinds of value:

    set(...)       sticky: the latest reading, resent on every 4-second tick (fps, alarm, objects)
    set_once(...)  one-shot: **queued**, one per message, then forgotten (scene, answer)

and one nudge:

    wake()         "do not wait for the tick, send now" - used when the alarm state changes or a
                   command produced something worth seeing immediately

`wait_for_wake(timeout)` is the publisher's sleep: it returns early when someone calls `wake()`, so
the 4-second cadence costs nothing and an interesting event still arrives in the dashboard at once.

Deliberately stdlib-only and cloud-free: `app.py` imports this, and the video half of the pilot has
to keep running with the /IOTCONNECT SDK not installed at all.
"""

from __future__ import annotations

from threading import Event, Lock

TelemetryValue = str | float | int | bool | None

MAX_QUEUED_ONCE = 4  # per attribute, so a booth with no network cannot grow this without bound


class TelemetryState:
    """Latest telemetry values + a wake flag, safe to write from any thread."""

    def __init__(self) -> None:
        self._values: dict[str, TelemetryValue] = {}
        self._once: dict[str, list[TelemetryValue]] = {}
        self._lock = Lock()
        self._wake = Event()

    def set(self, **values: TelemetryValue) -> None:
        """Record the latest reading of one or more attributes; sent on the next tick."""
        with self._lock:
            self._values.update(values)

    def set_once(self, **values: TelemetryValue) -> None:
        """Queue values to send one per message and then forget, and ask for the first one now.

        `scene` and `answer` are the cases: a description or a reply is an answer to a question
        somebody asked, not a reading. Repeating it every four seconds for the rest of the day would
        be noise - and see `collect` for why the other ticks omit the attribute rather than nulling it.

        **A queue rather than a slot**, because two one-shots can be produced within one publishing
        interval and the second must not swallow the first. The case that forced it: asking the LLM
        to clear the alert makes the *handler* publish the event log as `answer` (the log is about to
        be destroyed, so it has to leave the device first) and then the model's own reply arrives as
        `answer` a moment later. Both matter, and they are sent in the order they happened.
        """
        with self._lock:
            for name, value in values.items():
                queued = self._once.setdefault(name, [])
                queued.append(value)
                del queued[:-MAX_QUEUED_ONCE]  # a backlog this long means nothing is being sent
        self.wake()

    def wake(self) -> None:
        """Ask the publisher to send now instead of at the end of its interval."""
        self._wake.set()

    def wait_for_wake(self, timeout_s: float) -> bool:
        """The publisher's sleep. Returns True if it was woken rather than timed out."""
        is_woken = self._wake.wait(timeout_s)
        self._wake.clear()
        return is_woken

    def get(self, name: str) -> TelemetryValue:
        """Read one sticky value without consuming anything - `collect` eats the one-shots."""
        with self._lock:
            return self._values.get(name)

    def collect(self) -> dict[str, TelemetryValue]:
        """Everything to publish now: the sticky values plus **one** queued value per one-shot attribute.

        An attribute whose value is `None` is **left out of the packet entirely** rather than sent
        as `null`. The back end does not treat those two the same: a missing field means "no reading
        this time" and the last one stands, while a null is a reading of nothing. `answer` is the
        case that matters - the LLM's reply belongs to the one message that follows the question,
        and every other message should look as though nobody asked.

        Anything still queued wakes the publisher again on the way out, so a second answer follows in
        its own message (a second later - `MIN_SEND_GAP_S`) rather than being dropped or merged.
        """
        with self._lock:
            values = dict(self._values)
            for name, queued in self._once.items():
                if queued:
                    values[name] = queued.pop(0)
            is_more_queued = any(self._once.values())
        if is_more_queued:
            self.wake()
        return {name: value for name, value in values.items() if value is not None}
