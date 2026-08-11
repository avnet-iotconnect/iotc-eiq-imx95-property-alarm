"""What the watchdog has seen happen, in the few lines somebody would actually read out.

Two rules make this small enough to keep in RAM and to put in front of a language model:

- **The first and the last of each kind.** Not the two most recent - the *first* is the one that
  cannot be recovered later. "The laptop has been moved, and it started four minutes ago" is a
  different fact from "the laptop was moved a second ago", and only the first tells you when the
  trouble began. Everything between them is noise: a laptop nudged for a minute produces an event
  every few seconds and the fortieth says nothing the second did not.
- **Relative time.** "four minutes ago" is what a person at the booth wants and what the LLM can
  repeat without doing arithmetic. Times are `monotonic()`, so a clock correction cannot make an
  event look like it happened tomorrow.

**An alert is simply this log being non-empty** - there is no separate flag anywhere. That is the
whole meaning of the word in this demo: *alarm* is the switch the user arms, *alert* is "something
is in the log". Clearing one clears the other because they are the same thing (`watchdog.py`).

It is deliberately RAM-only: an event is a thing that just happened, and after a restart nothing
here is true any more. `state.json` keeps *settings*; this keeps history (see `state.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

# The kinds. Each gets its own slot in the log, and the guard/watchdog name one when they trigger.
INTRUDER = "intruder"
OBJECT_MOVED = "object_moved"
OBJECT_GONE = "object_gone"

@dataclass(frozen=True)
class Event:
    """One thing that happened: which kind, what to say about it, and when (monotonic seconds)."""

    kind: str
    description: str
    at: float


class EventLog:
    """The first and the most recent event of each kind, read back in time order."""

    def __init__(self) -> None:
        self._first: dict[str, Event] = {}
        self._last: dict[str, Event] = {}

    def add(self, kind: str, description: str, now: float) -> Event:
        """Record one event. The first of its kind is kept for good; the rest replace each other."""
        event = Event(kind, description, now)
        self._first.setdefault(kind, event)
        self._last[kind] = event
        return event

    def clear(self) -> None:
        """Forget everything - what `clear alert` and disarming both mean (see `watchdog.py`)."""
        self._first.clear()
        self._last.clear()

    @property
    def is_empty(self) -> bool:
        return not self._first

    def get_events(self) -> list[Event]:
        """Every event kept, oldest first. A kind that happened once appears once, not twice."""
        kept = {id(event): event for event in (*self._first.values(), *self._last.values())}
        return sorted(kept.values(), key=lambda event: event.at)

    def get_newest(self) -> Event | None:
        """The most recent event of any kind - what the HUD has room to say."""
        return max(self._last.values(), key=lambda event: event.at, default=None)

    def get_log(self, now: float) -> str:
        """The whole log as one line of prose: 'the laptop was moved, 4 seconds ago; ...'.

        Prose rather than a list of dicts because both readers want prose - the console, and the
        LLM whose 4096-token window this shares with the tool schemas and its own answer.
        """
        return "; ".join(f"{event.description}, {describe_age(now - event.at)}"
                         for event in self.get_events())


def describe_age(seconds: float) -> str:
    """'just now', '40 seconds ago', '7 minutes ago', '2 hours ago' - one useful figure, never two."""
    if seconds < 5:
        return "just now"
    if seconds < 90:
        return f"{round(seconds)} seconds ago"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} minutes ago"
    return f"{round(seconds / 3600)} hours ago"
