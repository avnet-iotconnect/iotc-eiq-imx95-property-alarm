"""Turning a heard sentence into a command, and running it off the video loop.

Two halves, and the split is deliberate:

**`parse()` - text to structure.** Pure string work: no camera, no database, no knowledge of who is
registered. It answers "which verb, and what argument was spoken", nothing more. Matching that
argument to a real user or a real object is `app.py`'s job, because that is the layer that knows what
exists. Parsing anchors on single words rather than whole phrases, because the recogniser mangles
leading verbs freely - falcon caught "Bridge the user plantier" for "register user Plantier" - while
usually getting one anchor word right.

**`CommandService` - running it.** Every command source funnels through here: voice today, an
/IOTCONNECT C2D message next. Handlers run on a small thread pool (4 workers) so a slow command -
`describe scene` takes tens of seconds inside the VLM - never stalls the 30 fps video loop.

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
LOCK_OBJECT = "lock_object"
UNLOCK_OBJECT = "unlock_object"
DESCRIBE_SCENE = "describe_scene"
SNAPSHOT = "snapshot"

NOT_UNDERSTOOD = "Sorry, I did not understand that."


@dataclass(frozen=True)
class Command:
    """A parsed command. `argument` is what was *heard*, not yet matched to anything real."""

    verb: str
    argument: str = ""
    text: str = ""  # the original sentence, kept for logging


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
        """Queue a command sentence. Voice waits on the future; C2D will not have to."""
        return self._executor.submit(self.run, text, source)

    def run(self, text: str, source: str = "voice") -> CommandResult:
        """Parse and execute one sentence, turning every failure into speakable text."""
        command = parse(text)
        if command is None:
            print(f"[cmd] {source}: {text.strip()!r} -> not a command")
            return CommandResult(False, NOT_UNDERSTOOD, source)

        print(f"[cmd] {source}: {text.strip()!r} -> {command.verb}"
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
