"""Turning a heard sentence into a command, and running it off the video loop.

Two halves, and the split is deliberate:

**`parse()` - text to structure.** Pure string work: no camera, no database, no knowledge of who is
registered. It answers "which verb, and what argument was spoken", nothing more. Matching that
argument to a real user or a real object is `app.py`'s job, because that is the layer that knows what
exists. Parsing anchors on single words rather than whole phrases, because the recogniser mangles
leading verbs freely - "Bridge the user plantier" for "register user Plantier", on this board - while
usually getting one anchor word right.

**`CommandService` - running it.** Every command source funnels through here: voice, the debug file
and /IOTCONNECT C2D. Handlers run on a small thread pool (4 workers) so a slow command - `describe
scene` takes tens of seconds inside the VLM, and `agent` tens of seconds inside the LLM on the
Ara-240 - never stalls the 30 fps video loop.

There is one command that comes back the *other* way. `agent` hands a sentence to the model on the
Ara, whose tools then run commands of their own (`app/agent.py`) - so a handler can be re-entered
from inside another handler. `AntiTheftApp.on_command` is where that is dealt with, and it is the
only exception to "one command at a time" in the demo.

There are two ways in, and which one a source uses says what it knows. Voice has a *sentence*, so it
calls `submit(text)` and gets it parsed. The cloud already has a **verb and its arguments** (the
/IOTCONNECT template names them), so it builds a `Command` itself and calls `submit_command(...)`;
running "user-register Nick" through a parser tuned for mangled speech would only lose information.
Both end in the same `execute()`, the same handler and the same `CommandResult`.

The result type is the reason this file exists at all. A handler either returns a sentence or raises
`CommandError`; both come back as a `CommandResult` carrying `is_ok` plus **text**. Voice speaks that
text either way; a C2D ack sends it back to the cloud as the command response. Making the failure
path a string rather than an exception is what lets one implementation serve both.

    service = CommandService(app.on_command)
    result = service.submit("register user Michael", source="voice").result()
    tts.speak(result.message)
"""

from __future__ import annotations

import logging
import string
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

# The verbs. Strings rather than an Enum so an /IOTCONNECT C2D payload can name one directly.
REGISTER_USER = "register_user"
UNREGISTER_USER = "unregister_user"
UNREGISTER_LAST = "unregister_last"
ARM = "arm"
DISARM = "disarm"
CLEAR_ALERT = "clear_alert"
DESCRIBE_ALERT = "describe_alert"
LOCK_OBJECT = "lock_object"
UNLOCK_OBJECT = "unlock_object"
DESCRIBE_SCENE = "describe_scene"
SNAPSHOT = "snapshot"
RESTART = "restart"  # no anchor words below on purpose: a mangled transcript must not restart us
AGENT = "agent"      # likewise: a whole sentence for the LLM, so there is nothing to anchor on

NOT_UNDERSTOOD = "Sorry, I did not understand that."


@dataclass(frozen=True)
class Command:
    """A command to run. `argument` is not yet matched to anything real - see `is_exact` for how.

    `is_exact` is the contract the *source* offers about its own argument. Voice cannot promise
    anything: the recogniser emits "money" for Marija, so its arguments have to be guessed at
    phonetically. A C2D message carries text somebody typed against a name the back end already
    knows, so guessing is not a service - it is a way to delete the wrong user. `app.py` reads this
    flag and picks the matching rule; nothing else changes between the two.
    """

    verb: str
    argument: str = ""
    text: str = ""  # the original sentence, kept for logging
    is_exact: bool = False  # True: match the argument literally, or fail and say what exists


@dataclass(frozen=True)
class CommandResult:
    """The answer to one command: spoken aloud by voice, sent as the ack by /IOTCONNECT."""

    is_ok: bool
    message: str
    source: str = "voice"
    command: Command | None = None


class CommandError(Exception):
    """Raised by a handler when a command cannot be carried out; the message is spoken to the user.

    Phrase it as something a person at the booth should hear: "Michael is already registered", not
    "duplicate key". It is the demo's entire error channel.
    """


# --- parsing ---------------------------------------------------------------------------------
# Anchor words, checked as whole words after normalisation. Each set holds the real word plus the
# manglings STT has been seen to produce for it. Add to these when the booth catches a new one.

_UNREGISTER = {"unregister", "unregistered", "unregistering", "deregister", "deregistered",
               "onregister", "anregister", "remove", "removed", "forget", "delete", "deleted",
               "erase", "drop"}
_REGISTER = {"register", "registered", "registering", "registry", "bridge", "rechester"}
_UNLOCK = {"unlock", "unlocked", "unlocking", "onlock", "anlock", "delock"}
_LOCK = {"lock", "locked", "locking", "guard", "block"}
_DISARM = {"disarm", "disarmed", "disarming", "desarm", "dearm", "unarm", "disturb"}
_ARM = {"arm", "armed", "arms", "harm", "alarm"}
# "clear the alert" has to be tested before _ARM, because "clear the alarm" - which people say, and
# mean - contains a word in _ARM. The anchor is the verb, so the noun may come out as anything.
_CLEAR = {"clear", "cleared", "clearing", "acknowledge", "acknowledged", "dismiss", "dismissed",
          "reset", "resets"}
# ... and the *noun*, which is what tells "describe the alert" from "describe the scene". Tested
# before _DESCRIBE for exactly that reason. "happened" is here because "what happened?" is what
# people actually say, and it anchors nothing else.
_ALERT = {"alert", "alerts", "alerted", "alart", "log", "logs", "event", "events", "happened",
          "happening"}
_DESCRIBE = {"describe", "description", "describes", "scene", "seen", "see", "look"}
_SNAPSHOT = {"snapshot", "screenshot", "photo", "picture", "snap", "shot"}
_USER = {"user", "used", "usar", "uses", "eraser", "person", "object"}

# The recogniser splits the prefixed verbs: "un register", "dis arm". Rejoin them verbatim - the
# result ("unregister", "dearm", "onregister") is looked up in the sets above, which is why those
# hold the odd spellings too. Only these stems are ever joined, so "lock an apple" stays two words.
_PREFIXES = {"un", "on", "an", "de", "dis", "des"}
_JOINABLE = {"register", "registered", "registering", "lock", "locked", "locking",
             "arm", "armed", "arming"}

# Never part of an argument; leaving them in drags the phonetic score down ("of Maria" matched
# Marija at 0.47 where "Maria" alone scores far higher).
_FILLER = {"of", "the", "a", "an", "is", "as", "to", "for", "and", "in", "it", "my", "this",
           "please", "now", "okay", "ok", "thanks", "up", "called", "named", "name"}


def parse(text: str) -> Command | None:
    """Recognise one command in a spoken sentence. `None` when it is not one of ours.

    Ordered most-specific first, because the verbs overlap as words: "unlock" contains "lock",
    "disarm" contains "arm", and "unregister last user" contains "unregister ... user".
    """
    words = _normalise(text)
    spoken = set(words)
    if not words:
        return None

    def command(verb: str, anchors: set[str] = frozenset()) -> Command:
        return Command(verb, _argument_after(words, anchors), text.strip())

    if _SNAPSHOT & spoken:  # before describe: "take a picture of the scene" is a snapshot
        return command(SNAPSHOT)
    if _ALERT & spoken and not _CLEAR & spoken:  # before describe: the noun picks which description
        return command(DESCRIBE_ALERT)
    if _DESCRIBE & spoken:
        return command(DESCRIBE_SCENE)
    if _UNREGISTER & spoken:
        if "last" in spoken:
            return command(UNREGISTER_LAST)
        return command(UNREGISTER_USER, _UNREGISTER | _USER)
    if _REGISTER & spoken:
        return command(REGISTER_USER, _REGISTER | _USER)
    if _UNLOCK & spoken:
        return command(UNLOCK_OBJECT, _UNLOCK | _USER)
    if _LOCK & spoken:
        return command(LOCK_OBJECT, _LOCK | _USER)
    if _CLEAR & spoken:  # before both: "clear the alarm" carries an _ARM word and is not an arming
        return command(CLEAR_ALERT)
    if _DISARM & spoken:
        return command(DISARM)
    if _ARM & spoken:
        return command(ARM)
    return None


def _normalise(text: str) -> list[str]:
    """Lower-case words in order, punctuation stripped, split verb prefixes rejoined ("un register")."""
    stripped = text.lower().translate(str.maketrans("", "", string.punctuation))
    words: list[str] = []
    for word in stripped.split():
        if words and words[-1] in _PREFIXES and word in _JOINABLE:
            words[-1] += word
        else:
            words.append(word)
    return words


def _argument_after(words: list[str], anchors: set[str]) -> str:
    """Everything spoken after the LAST anchor word, filler dropped. '' when there is no anchor.

    The last anchor wins so "register user Michael" cuts after "user", not after "register", and
    "lock the laptop" still works when only the verb is there to anchor on.
    """
    if not anchors:
        return ""
    cut = max((index for index, word in enumerate(words) if word in anchors), default=-1)
    return " ".join(word for word in words[cut + 1:] if word not in _FILLER)


# --- running ---------------------------------------------------------------------------------


class CommandService:
    """Runs commands off the video loop, on at most `max_workers` threads.

    Four workers, not one: `describe scene` occupies a thread for tens of seconds inside the VLM,
    and an arm/disarm arriving meanwhile should not wait behind it. Four is also the ceiling that
    keeps the A55s free enough for the camera and YOLO to hold 30 fps (`run.sh` pins OMP the same
    way). The handler serialises itself internally - see `AntiTheftApp.on_command`.
    """

    def __init__(self, handler: Callable[[Command], str], max_workers: int = 4) -> None:
        self.handler = handler
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="command")

    def submit(self, text: str, source: str = "voice") -> Future[CommandResult]:
        """Queue a command sentence. Voice waits on the future; C2D does not."""
        return self._executor.submit(self.run, text, source)

    def submit_command(self, command: Command, source: str = "c2d") -> Future[CommandResult]:
        """Queue an already-structured command, for a source that did not have to guess the verb."""
        return self._executor.submit(self.execute, command, source)

    def run(self, text: str, source: str = "voice") -> CommandResult:
        """Parse and execute one sentence, turning every failure into speakable text."""
        command = parse(text)
        if command is None:
            print(f"[cmd] {source}: {text.strip()!r} -> not a command")
            return CommandResult(False, NOT_UNDERSTOOD, source)
        return self.execute(command, source)

    def execute(self, command: Command, source: str = "voice") -> CommandResult:
        """Run one parsed command. The only place a handler is called, whatever the source."""
        print(f"[cmd] {source}: {command.text.strip()!r} -> {command.verb}"
              f"{' ' + repr(command.argument) if command.argument else ''}")
        try:
            message = self.handler(command)
            return CommandResult(True, message, source, command)
        except CommandError as error:
            print(f"[cmd] refused: {error}")
            return CommandResult(False, str(error), source, command)
        except Exception as error:  # a bug in a handler must not kill the demo
            logger.exception("command %s raised", command.verb)
            return CommandResult(False, f"Something went wrong: {type(error).__name__}", source, command)

    def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
