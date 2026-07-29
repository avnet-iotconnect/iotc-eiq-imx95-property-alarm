"""What the device wants the cloud to know, in one small object nobody has to wait on.

The problem this solves: telemetry is produced on three threads that must never block - the 30 fps
video loop (fps, what is on screen), the command threads (a VLM scene description), the alarm state
machine - and it is *sent* on exactly one, the /IOTCONNECT publisher in `iotc.py`. Having producers
call the MQTT client directly would put a network call on the frame loop and would make `app.py`
depend on the cloud.

So producers only ever write here, and the publisher reads here. Two kinds of value:

    set(...)       sticky: the latest reading, resent on every 4-second tick (fps, alarm, objects)
    set_once(...)  one-shot: sent with the next message and then forgotten (scene)

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


class TelemetryState:
    """Latest telemetry values + a wake flag, safe to write from any thread."""

    def __init__(self) -> None:
        self._values: dict[str, TelemetryValue] = {}
        self._once: dict[str, TelemetryValue] = {}
        self._lock = Lock()
        self._wake = Event()

    def set(self, **values: TelemetryValue) -> None:
        """Record the latest reading of one or more attributes; sent on the next tick."""
        with self._lock:
            self._values.update(values)

    def set_once(self, **values: TelemetryValue) -> None:
        """Record values to send with the next message and then forget, and ask for it now.

        `scene` is the case: a scene description is an answer to a question somebody asked, not a
        reading. Repeating it every four seconds for the rest of the day would be noise.
        """
        with self._lock:
            self._once.update(values)
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
        """Everything to publish now: the sticky values plus the one-shots, which are consumed."""
        with self._lock:
            values = dict(self._values)
            values.update(self._once)
            self._once.clear()
        return values
