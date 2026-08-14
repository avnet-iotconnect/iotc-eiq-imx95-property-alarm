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

**Every tool call is printed** - the name, the argument the model chose and what came back, two
indented lines per call (`log_call`). The model's own closing sentence is not evidence of what it
did - it describes what it meant to do - and the argument is the half most worth seeing, because
that is where a mangled name becomes the wrong user.

**A request can need more than one tool, and two things had to change for that to work.**
"Register user Joe and take a screenshot" is two commands, and it used to do neither.

1. *The connector loses tool calls when it streams* - `IS_STREAMING = False`, and this is the half
   that mattered. Measured against the board: streamed, that sentence comes back as an empty
   message with `finish_reason: stop` and no tool call at all, while the **same** request
   non-streamed returns a correct `register_user('Joe')`. Single-tool questions do stream
   correctly, which is how this hid for so long. Nothing here needs tokens as they are produced -
   the answer is spoken and sent to the cloud when it is finished - so not streaming costs the
   demo nothing. See `connector/README.md`.
2. *The prompt has to say so*: a model this size will otherwise do the first thing, answer, and
   consider the sentence dealt with. `SYSTEM_PROMPT` asks for one tool at a time, each after the
   last one's result, and for an answer only when nothing is left to do - the sequential phrasing
   matters, because parallel tool calls would run two of `app.py`'s handlers against the same lock.

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
from threading import Lock, Thread
from time import perf_counter, sleep
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
IS_STREAMING = False                   # the connector loses tool calls when it streams - see above

# The HUD says "ready" only once the model has actually answered something. Building this object
# proves nothing - the connector is a separate process that takes ~4 minutes to load its model and
# may not be running at all - so one throwaway question is asked in the background, and asked again
# every PROBE_INTERVAL_S until it lands. A question that arrives before then is refused in a
# sentence instead of waiting ~40 s to fail, which is what a C2D command used to do.
PROBE_QUESTION = "Say ready."
PROBE_SYSTEM_PROMPT = "Answer with a single word."
PROBE_INTERVAL_S = 30.0

SYSTEM_PROMPT = (
    "You are the assistant built into an anti-theft camera demo on an NXP i.MX95 board. "
    "Use a tool whenever the answer depends on what the demo knows, or when the user asks you to "
    "change something. Never guess at the state of the demo. "
    "A request may ask for more than one thing: do every part of it, one tool at a time, waiting "
    "for each result before calling the next tool, and only answer once nothing is left to do. "
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
            # stream=False is not a preference - see IS_STREAMING.
            params={"temperature": TEMPERATURE, "max_tokens": max_tokens, "stream": IS_STREAMING},
        )
        self._tools = self._build_tools()
        self._lock = Lock()
        self.is_ready = False
        self.on_status("agent: starting")
        Thread(target=self._watch_the_connector, name="agent-probe", daemon=True).start()

    def _watch_the_connector(self) -> None:
        """Ask the model one throwaway question in the background until it answers - and again if a
        real question later fails, so the line recovers by itself when the connector comes back.

        A thread because the connector may still be loading its model, and a demo that waited for
        that - or refused to start without it - would be a worse demo. Nothing here can fail loudly:
        the only thing at stake is one word on the HUD.
        """
        is_first_failure = True
        while True:
            if self.is_ready:
                sleep(PROBE_INTERVAL_S)
                continue
            try:
                with self._lock:
                    Agent(model=self._model, system_prompt=PROBE_SYSTEM_PROMPT,
                          callback_handler=None)(PROBE_QUESTION)
            except Exception as error:
                if is_first_failure:  # once, not every thirty seconds for the rest of the day
                    print(f"[agent] the Ara is not answering yet ({type(error).__name__}) - "
                          f"retrying every {PROBE_INTERVAL_S:.0f}s")
                    is_first_failure = False
                self._set_ready(False)
                sleep(PROBE_INTERVAL_S)
                continue
            self._set_ready(True)
            print("[agent] the Ara answered - ready for questions")

    def _set_ready(self, is_ready: bool) -> None:
        self.is_ready = is_ready
        self.on_status("agent: ready" if is_ready else "agent: offline")

    def ask(self, question: str) -> str:
        """Answer one question, running whatever tools the model decides it needs. Blocks for ~30 s."""
        if not self.is_ready:
            # Every source funnels through here - voice, C2D, `--text-commands` - so this is the one
            # place the wait has to be cut short. Without it a dashboard command sits for ~40 s and
            # comes back with a connection error.
            raise CommandError("The assistant is still starting up on the Ara-240. Try again shortly.")
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
                # A question that failed is evidence about the connector, the same as the probe's:
                # the line goes back to "offline" rather than claiming to be ready.
                self._set_ready(False)
                raise CommandError(
                    f"I could not reach the language model on the Ara ({type(error).__name__}). "
                    f"Check that the connector is running: cd connector && ./run.sh") from error
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
            log_call("get_time")
            return log_result("get_time", datetime.now().astimezone()
                              .strftime("%A %Y-%m-%d %H:%M %Z (UTC%z)"))

        @tool
        def get_status() -> str:
            """Return the alarm state, the registered users and what the camera can see right now."""
            log_call("get_status")
            return log_result("get_status", describe_status())

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

        A refusal is logged as the result, not as a failure, because that is what it is from here:
        the model reads it and says it back in its own words.
        """
        log_call(verb, argument)
        try:
            return log_result(verb, self.run_command(
                Command(verb, argument, f"agent: {verb}", is_exact=True)))
        except CommandError as error:
            return log_result(verb, str(error))


def log_call(name: str, argument: str = "") -> None:
    """Print what the model decided to do, *before* it happens.

    Two lines per tool, indented under the question, because a run of the demo used to show only the
    model's final sentence - and a sentence is exactly the part that cannot be trusted to say what
    was really done, or with which argument. The call line comes first because tools take time (a
    registration counts a person down for up to nine seconds), and a console that speaks only
    afterwards looks hung.

    `flush` because these lines are interleaved with four other threads' logging and stdout is a
    pipe whenever the demo is run from anything but a terminal.
    """
    print(f"    [agent] tool {name}({argument!r})" if argument else f"    [agent] tool {name}()",
          flush=True)


def log_result(name: str, result: str) -> str:
    """... and print what came back, which is all the model has to answer from. Returns it."""
    print(f"    [agent] tool {name} -> {result!r}", flush=True)
    return result
