"""'agent': one plain-English question, answered by the 7B model on the Ara-240 - with tools.

This is the only piece of the demo that is *not* deterministic, and the boundary is drawn tightly on
purpose. The model does not decide when the alarm goes off (`app.py` does that, in code). What it
does is turn a sentence a visitor typed into the dashboard - "please disarm the alarm", "register
another user Michael", "who can you see?" - into the same commands the dashboard has buttons for.

**Where the model actually is.** Not in this process. `connector/` installs NXP's eIQ AAF Connector
on the board, in its own venv, and it keeps Qwen2.5-7B-Instruct resident on the Ara-240 behind an
OpenAI-shaped REST endpoint on port 3000. That boundary is load-bearing: the connector's optimum-ara
hard-pins its own torch and cannot share a process with the BSP's `tflite_runtime`/`gi`/`cv2` stack
that the video half needs. On this side of HTTP the whole dependency list is `strands-agents` and
`openai` - no ML packages at all.

**The tools are the command verbs.** Every tool below is a one-line call into `app.on_command`, the
same entry point voice and /IOTCONNECT C2D use, with the same handlers, the same refusals and the
same lock. So the model cannot do anything a dashboard command could not do, and a refusal
("Michael is already registered") comes back to it as text, which it relays in its own words.

**Three limits worth knowing before changing anything here:**

- *4096 tokens, prompt plus generation*, compiled into the model - and every tool schema is part of
  every prompt. Eleven tools cost roughly 700 of them, which is why the descriptions below are one
  line each and why what a tool *returns* is kept short too. That budget is also why each question
  gets a **fresh `Agent`**: strands keeps conversation history, and a booth that runs all day would
  overflow the window by mid-morning. Each question is a clean slate.
- *~5 tokens/second.* A tool call plus an answer is 20-40 seconds. It runs on a command thread, off
  the video loop, and the /IOTCONNECT ack is sent when it finishes - so nothing waits on it but the
  person who asked.
- *Temperature must not be 0.0.* The Ara does its own sampling on-device and cannot sample from
  degenerate parameters; the connector's shipped default of 0.0 makes every request fail with an
  HTTP 500 that looks like a version mismatch. See `connector/README.md`.

One question at a time (`_lock`): the Ara generates serially anyway, so a second concurrent question
would only queue inside the connector - and this keeps command workers free for the tools' own
commands.
"""

from __future__ import annotations

import logging
from datetime import datetime
from threading import Lock
from time import perf_counter
from typing import Callable

from applib import commands
from applib.commands import Command, CommandError

logger = logging.getLogger(__name__)

try:
    from strands import Agent, tool
    from strands.models.openai import OpenAIModel
    IS_STRANDS_AVAILABLE = True
except ImportError:  # the rest of the demo runs fine without it; the HUD says "agent: unavailable"
    IS_STRANDS_AVAILABLE = False

ARA_URL = "http://127.0.0.1:3000/v1"   # the connector, on the board, beside us
ARA_MODEL = "Qwen2.5-7B-Instruct"      # the 1.5B also loads, but loops forever instead of calling tools
MAX_TOKENS = 256                       # ~50 s of generation at 5 tok/s; long enough for any answer here
TEMPERATURE = 0.7                      # NOT 0.0 - see the module docstring

SYSTEM_PROMPT = (
    "You are the assistant built into an anti-theft camera demo on an NXP i.MX95 board. "
    "Use a tool whenever the answer depends on what the demo knows, or when the user asks you to "
    "change something. Never guess at the state of the demo. "
    "Answer in one or two short sentences, in plain language."
)


class AgentService:
    """A question in, a sentence out. Tools call straight back into the command handlers."""

    def __init__(
        self, run_command: Callable[[Command], str], get_status: Callable[[], str],
        base_url: str = ARA_URL, model_id: str = ARA_MODEL, max_tokens: int = MAX_TOKENS,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.run_command = run_command  # app.on_command: runs one command, returns what to say
        self.get_status = get_status    # app.get_status: one line of ground truth for the model
        self.model_id = model_id
        self.base_url = base_url
        self.on_status = on_status or (lambda status: None)
        self._model = OpenAIModel(
            client_args={"api_key": "unused", "base_url": base_url},
            model_id=model_id,
            params={"temperature": TEMPERATURE, "max_tokens": max_tokens},
        )
        self._tools = self._build_tools()
        self._lock = Lock()
        self.on_status("agent: ready")

    def ask(self, question: str) -> str:
        """Answer one question, running whatever tools the model decides it needs. Blocks for ~30 s."""
        with self._lock:
            self.on_status("agent: thinking")
            started = perf_counter()
            try:
                # A new Agent per question: no history, so the 4096-token window is never eaten by
                # what somebody asked an hour ago.
                # callback_handler=None: strands otherwise prints the answer to stdout as it
                # streams, which on a board with four other threads logging is unreadable. We print
                # one line below instead.
                agent = Agent(model=self._model, tools=self._tools, system_prompt=SYSTEM_PROMPT,
                              callback_handler=None)
                answer = str(agent(question)).strip()
            except Exception as error:
                # One line, not a traceback. A connector that is not running is an ordinary answer
                # here - "start it" - and strands' stack for it is forty frames of asyncio, httpx
                # and openai internals, none of which say anything the message below does not.
                # `--iotc-verbose`-style detail is a `logging.DEBUG` away when it is really wanted.
                logger.debug("the agent failed", exc_info=True)
                print(f"[agent] {type(error).__name__}: {error}")
                raise CommandError(
                    f"I could not reach the language model on the Ara ({type(error).__name__}). "
                    f"Check that the connector is running: cd connector && ./run.sh") from error
            finally:
                self.on_status("agent: ready")
            print(f"[agent] {perf_counter() - started:.1f}s | {question!r} -> {answer!r}")
        return answer or "I do not have an answer for that."

    def _build_tools(self) -> list:
        """The tools, as closures over the command handlers. Docstrings are what the model reads.

        Keep them short and keep the list short: both the description and every argument name go
        into the prompt of *every* request, against a 4096-token budget shared with the answer.
        """
        run, describe_status = self._run, self.get_status

        @tool
        def get_time() -> str:
            """Return the current date, time and time zone on the device."""
            # astimezone() is what puts the zone on it: a naive datetime has none, and %Z would be
            # blank. It reads the board's own TZ, so a demo travelling to a booth reports local time.
            now = datetime.now().astimezone().strftime("%A %Y-%m-%d %H:%M %Z (UTC%z)")
            print(f"    [agent] tool get_time -> {now}")
            return now

        @tool
        def get_status() -> str:
            """Return the alarm state, the registered users and what the camera can see right now."""
            status = describe_status()
            print(f"    [agent] tool get_status -> {status}")
            return status

        @tool
        def take_screenshot() -> str:
            """Take a picture of what is on screen now and upload it to the cloud."""
            return run(commands.SNAPSHOT)

        @tool
        def arm_alarm() -> str:
            """Arm the theft alarm."""
            return run(commands.ARM)

        @tool
        def disarm_alarm() -> str:
            """Disarm the theft alarm."""
            return run(commands.DISARM)

        @tool
        def describe_alert() -> str:
            """Return the event log: what the demo has caught, in order, and how long ago."""
            return run(commands.DESCRIBE_ALERT)

        @tool
        def clear_alert() -> str:
            """Clear the event log, acknowledging whatever the demo has caught."""
            return run(commands.CLEAR_ALERT)

        @tool
        def register_user(name: str) -> str:
            """Register a new user, binding the given name to the face the camera can see.

            Args:
                name: the person's name, spelled as the user gave it.
            """
            return run(commands.REGISTER_USER, name)

        @tool
        def unregister_user(name: str) -> str:
            """Remove a registered user.

            Args:
                name: the name of a user who is already registered.
            """
            return run(commands.UNREGISTER_USER, name)

        @tool
        def lock_object(name: str) -> str:
            """Guard an object that is on screen, so that taking it triggers the alarm.

            Args:
                name: what to guard, as the camera labels it, for example "laptop".
            """
            return run(commands.LOCK_OBJECT, name)

        @tool
        def unlock_object(name: str) -> str:
            """Stop guarding an object.

            Args:
                name: the object to release, for example "laptop".
            """
            return run(commands.UNLOCK_OBJECT, name)

        return [get_time, get_status, take_screenshot, arm_alarm, disarm_alarm,
                describe_alert, clear_alert,
                register_user, unregister_user, lock_object, unlock_object]

    def _run(self, verb: str, argument: str = "") -> str:
        """Run one command for the model, and hand a refusal back as text rather than as an error.

        `is_exact`, as for a C2D command: the model produces the name the user typed, so matching it
        phonetically ("Michael" -> Marija) would act on the wrong person. When the name is not there,
        `app.py`'s refusal says which names are, and the model relays that.
        """
        print(f"    [agent] tool {verb}{' ' + repr(argument) if argument else ''}")
        try:
            return self.run_command(Command(verb, argument, f"agent: {verb}", is_exact=True))
        except CommandError as error:
            return str(error)
