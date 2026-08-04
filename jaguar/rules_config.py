"""Configure the anti-theft demo in plain English, on the Ara-240.

One LLM call turns a sentence into a rule object, which would be written to disk
and then evaluated every frame by ordinary Python. The model is never in the frame
loop -- it runs once, at startup or when the user asks for a change.

    .venv/bin/python jaguar/rules-config.py
    .venv/bin/python jaguar/rules-config.py "watch my backpack and shout if it goes"

Two ideas carry the whole design, both explained in work/STATUS-rules.md:

1. Generate *configuration*, not code. A rule is ~120 tokens; a plugin would be
   ~2000, which at 5.3 tok/s is six minutes and leaves no room to fix a mistake.

2. Read the rule back with a template, never by asking the model to describe its
   own output. A paraphrase tends to restate the request and hide the error; a
   template can only describe what will actually run.

Needs the connector up with Qwen2.5-7B-Instruct loaded (jaguar/connector/run.sh).
"""

import json
import sys
import time
import urllib.request

URL = "http://192.168.38.203:3000/v1/chat/completions"

OBJECTS = ["laptop", "cell phone", "backpack", "handbag", "bottle", "book",
           "keyboard", "mouse", "cup", "remote"]
STATES = ["disarmed", "armed", "unregistered_user", "theft"]
KINDS = ["registered", "unregistered", "any"]
COLORS = ["green", "yellow", "orange", "red"]

# The vocabulary the model may draw on. This is a *static* description of what can
# be said -- never live state. Tracks, positions and timestamps belong to the rule
# engine, which grows without bound and has no business in a prompt.
SYSTEM = f"""You configure an anti-theft demo. Emit exactly one rule using the create_rule tool.

WORLD
Guardable objects: {", ".join(OBJECTS)}.
People: a person is either "registered" (face enrolled by name) or "unregistered".
Alarm states: {", ".join(STATES)}.

CONDITIONS
object_missing(object, seconds)      object gone from the stand for N seconds
object_present(object)               object is on the stand
person_present(kind)                 kind is registered | unregistered | any
person_absent_for(kind, seconds)     no such person seen for N seconds
alarm_state_is(state)
time_between(start, end)             24h clock, e.g. "18:00", "07:00"

ACTIONS
set_alarm(state), announce(text), capture_and_upload(), describe_scene(), set_light(color)

Durations are in seconds: "a minute" is 60, "half a minute" is 30.
"Someone I don't know" is unregistered. "Tell me what happened" is describe_scene."""

# Every closed vocabulary is an enum, so the connector's guided generation makes an
# invalid value impossible rather than unlikely. Only genuinely free values
# (seconds, spoken text, clock times) are plain strings.
CONDITION = {
    "type": "object",
    "properties": {
        "check": {"type": "string", "enum": [
            "object_missing", "object_present", "person_present",
            "person_absent_for", "alarm_state_is", "time_between"]},
        "object": {"type": "string", "enum": OBJECTS},
        "kind": {"type": "string", "enum": KINDS},
        "state": {"type": "string", "enum": STATES},
        "seconds": {"type": "string"},
        "start": {"type": "string"},
        "end": {"type": "string"},
    },
    "required": ["check"],
}

ACTION = {
    "type": "object",
    "properties": {
        "do": {"type": "string", "enum": [
            "set_alarm", "announce", "capture_and_upload", "describe_scene", "set_light"]},
        "state": {"type": "string", "enum": STATES},
        "color": {"type": "string", "enum": COLORS},
        "text": {"type": "string"},
    },
    "required": ["do"],
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "create_rule",
        "description": "Create one anti-theft rule from the user's description.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "conditions": {"type": "array", "items": CONDITION},
                "actions": {"type": "array", "items": ACTION},
            },
            "required": ["name", "conditions", "actions"],
        },
    },
}]


def generate(text):
    """One call: sentence in, rule object out."""
    body = {
        "model": "Qwen2.5-7B-Instruct",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        "tools": TOOLS,
        "llm_params": {"generate_max_tokens": 300},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        answer = json.loads(r.read().decode())

    calls = answer["choices"][0]["message"].get("tool_calls")
    if not calls:
        return None, answer["usage"]
    return json.loads(calls[0]["function"]["arguments"]), answer["usage"]


# --- reading the rule back -------------------------------------------------
# Templates, so the sentence describes what will run rather than what was asked
# for. This is what makes a wrong duration or a dropped condition audible.

PERSON = {"registered": "someone you know", "unregistered": "a stranger", "any": "anyone"}


def duration(seconds):
    """Speak durations the way a person would, whatever is stored."""
    s = int(seconds)
    if s >= 120 and s % 60 == 0:
        return f"{s // 60} minutes"
    if s == 60:
        return "1 minute"
    return f"{s} second" + ("" if s == 1 else "s")


def say_condition(c):
    check = c.get("check")
    if check == "object_missing":
        # An absent duration is said out loud rather than quietly defaulted.
        gone = f" for {duration(c['seconds'])}" if c.get("seconds") else " (no time given)"
        return f"the {c.get('object', '?')} has been gone{gone}"
    if check == "object_present":
        return f"the {c.get('object', '?')} is back on the stand"
    if check == "person_present":
        return f"{PERSON.get(c.get('kind'), '?')} is in view"
    if check == "person_absent_for":
        return f"{PERSON.get(c.get('kind'), '?')} has not been seen for {duration(c.get('seconds', 0))}"
    if check == "alarm_state_is":
        return f"the alarm is {c.get('state', '?')}"
    if check == "time_between":
        return f"the time is between {c.get('start', '?')} and {c.get('end', '?')}"
    return f"[unknown condition {check}]"


def say_action(a):
    do = a.get("do")
    if do == "set_alarm":
        return f"set the alarm to {a.get('state', '?')}"
    if do == "announce":
        return f'say "{a.get("text", "")}"'
    if do == "capture_and_upload":
        return "send a photo to the cloud"
    if do == "describe_scene":
        return "describe what happened out loud"
    if do == "set_light":
        return f"turn the light {a.get('color', '?')}"
    return f"[unknown action {do}]"


def say(rule):
    conditions = [say_condition(c) for c in rule.get("conditions", [])]
    actions = [say_action(a) for a in rule.get("actions", [])]
    # Anything past three conditions is almost certainly a misunderstanding, so
    # count them aloud instead of burying it in a long "and" chain.
    if len(conditions) > 3:
        joined = f"all {len(conditions)} of: " + "; ".join(conditions)
    else:
        joined = " and ".join(conditions)
    return f"If {joined}, then I will {', '.join(actions)}."


# Guarded so rules-probe.py can import the schema, generate() and say() from here
# rather than keeping a second copy of them.
if __name__ == "__main__":
    request = sys.argv[1] if len(sys.argv) > 1 else (
        "if my laptop is off the stand for more than 5 seconds and no registered person "
        "has been around for 10 seconds, sound the theft alarm, turn the light red, "
        "take a photo to the cloud and describe what happened"
    )

    start = time.time()
    rule, usage = generate(request)
    elapsed = time.time() - start

    print(f'YOU SAID : "{request}"')
    if rule is None:
        print("no rule produced")
        sys.exit(1)

    print(f"I HEARD  : {say(rule)}")
    print()
    print(json.dumps(rule, indent=2))
    print(f"\n{usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion "
          f"= {usage['total_tokens']} of 4096 tokens, {elapsed:.0f} s")
