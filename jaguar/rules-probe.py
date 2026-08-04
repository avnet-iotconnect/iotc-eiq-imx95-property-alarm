"""How often does the model get a rule right? Twelve deliberately sloppy requests.

Guided generation guarantees the rule is *structurally* valid -- an invalid object
or action cannot be emitted. It guarantees nothing about meaning, so this measures
the part that can still be wrong: the wrong condition, the wrong duration, a
dropped clause.

    .venv/bin/python jaguar/rules-probe.py

Roughly four minutes. Judge the output by reading the "I HEARD" line against the
request: that read-back is the same one a user would confirm, so if the error is
not visible there, it would not be visible to them either.
"""

import time

from rules_config import generate, say

REQUESTS = [
    "guard my laptop",
    "if someone i don't know shows up turn the light orange",
    "watch my phone, alarm after 3 seconds if nobody i know is there",
    "after 6pm at night arm everything",
    "if my bike gets taken sound the alarm",
    "tell me what happened when the backpack goes missing",
    "keep an eye on the cup and shout if it disappears",
    "disarm when i'm here",
    "if the laptop comes back go back to armed",
    "protect everything on the desk",
    "handbag gone for a minute with a stranger around -> red light and a photo",
    "dont let anyone steal my keyboard ok",
]

for request in REQUESTS:
    start = time.time()
    rule, usage = generate(request)
    elapsed = time.time() - start

    print(f'\nYOU SAID : "{request}"')
    if rule is None:
        print("I HEARD  : (no rule produced)")
        continue
    print(f"I HEARD  : {say(rule)}")
    print(f"           [{usage['total_tokens']} tokens, {elapsed:.0f} s]")
