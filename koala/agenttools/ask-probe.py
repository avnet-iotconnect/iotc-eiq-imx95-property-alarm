#!/usr/bin/env python3
"""Exercise the `ask` path with a fake demo behind it - no camera, no models, no board needed.

`AskAgent` only ever talks to two callables and one HTTP endpoint, so everything below the LLM can
be replaced by thirty lines of dictionary. That makes this the fastest way to answer the three
questions that actually go wrong:

    is the connector up and the model resident?     (it says so, immediately)
    does the model pick the right tool?             (every tool call prints)
    how long does one question take?                (it times each one)

Run it from the host, pointed at the board:

    .venv/bin/python koala/agenttools/ask-probe.py --url http://192.168.38.203:3000/v1
    .venv/bin/python koala/agenttools/ask-probe.py "register another user Michael"

Needs only `strands-agents` and `openai` - no BSP packages, nothing from the vision half. If this
works and the demo does not, the difference is the demo, not the Ara.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ask  # noqa: E402
from app.ask import AskAgent  # noqa: E402
from applib import commands  # noqa: E402
from applib.commands import Command, CommandError  # noqa: E402

QUESTIONS = [
    "What time is it?",
    "Please disarm the alarm.",
    "Register another user called Michael.",
    "Is the alarm on, and who can you see?",
]


class FakeApp:
    """The whole contract `AskAgent` has with the demo: run a command, and describe the state.

    Same method names and same failure channel as `app.py`, so a tool that works here works there.
    """

    def __init__(self) -> None:
        self.is_armed = True
        self.users = ["Nick"]
        self.locked: list[str] = []
        self.visible = "Nick, person, laptop"

    def on_command(self, command: Command) -> str:
        if command.verb == commands.ARM:
            self.is_armed = True
            return "Alarm armed."
        if command.verb == commands.DISARM:
            self.is_armed = False
            return "Alarm disarmed."
        if command.verb == commands.REGISTER_USER:
            if command.argument in self.users:
                raise CommandError(f"{command.argument} is already registered.")
            self.users.append(command.argument)
            return f"Registered {command.argument}."
        if command.verb == commands.UNREGISTER_USER:
            if command.argument not in self.users:
                raise CommandError(f"I do not know anyone called {command.argument}. "
                                   f"I know {', '.join(self.users)}.")
            self.users.remove(command.argument)
            return f"Unregistered {command.argument}."
        if command.verb == commands.LOCK_OBJECT:
            self.locked.append(command.argument)
            return f"The {command.argument} is locked. I am watching it."
        if command.verb == commands.UNLOCK_OBJECT:
            if command.argument not in self.locked:
                raise CommandError(f"I do not have a {command.argument} locked.")
            self.locked.remove(command.argument)
            return f"The {command.argument} is unlocked."
        raise CommandError(f"I do not know how to {command.verb}.")

    def get_status(self) -> str:
        return (f"The alarm is {'armed' if self.is_armed else 'disarmed'}. "
                f"Registered users: {', '.join(self.users) or 'none'}. "
                f"Guarded objects: {', '.join(self.locked) or 'none'}. "
                f"The camera can see: {self.visible}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("questions", nargs="*", help="what to ask (default: four sample questions)")
    parser.add_argument("--url", default="http://192.168.38.203:3000/v1",
                        help="the connector's OpenAI endpoint")
    parser.add_argument("--model", default=ask.ARA_MODEL, help="model name the connector serves")
    args = parser.parse_args()

    if not ask.IS_STRANDS_AVAILABLE:
        print("strands-agents is not installed: pip install strands-agents openai")
        return 1

    fake = FakeApp()
    agent = AskAgent(fake.on_command, fake.get_status, base_url=args.url, model_id=args.model)
    print(f"{args.model} at {args.url}\n{fake.get_status()}\n" + "-" * 70)

    for question in args.questions or QUESTIONS:
        started = perf_counter()
        try:
            answer = agent.ask(question)
        except CommandError as error:
            print(f"\nQ: {question}\n!  {error}")
            return 1
        print(f"\nQ: {question}\nA: {answer}\n   ({perf_counter() - started:.1f}s)  {fake.get_status()}")
    print("-" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
