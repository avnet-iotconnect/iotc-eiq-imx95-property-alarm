# benchmarks/ara — how fast the LLM on the Ara-240 actually is

`bench-ara.py` streams one completion and times the first token separately from the rest, because
those two numbers behave completely differently: time-to-first-token is roughly fixed, while decode
scales with how much the model says. It is the number that decides whether an idea is demo-able.

    .venv/bin/python koala/benchmarks/ara/bench-ara.py

## Measured, Qwen2.5-7B-Instruct on rt-sdk-ara2 2.0.4

| | |
|---|---|
| model load onto the Ara (once, at connector start) | ~225 s — ~28 s per GB of `.dvm` |
| TTFT through the connector | ~2.3 s |
| decode, connector v2.0.0 + optimum-ara 2.0.0.2 | **5.28 tok/s** |
| decode, driving optimum-ara directly (no REST) | 6.3 tok/s |

The REST boundary costs about 16–30 % of decode throughput and nothing on TTFT. It buys a resident
model, an OpenAI endpoint, and a dependency wall between the Ara's torch and the BSP's
`tflite_runtime`/`gi`/`cv2` — worth it, but not free.

## What that means for `agent` (measured 2026-08-04, host → board)

| question | tool called | wall clock |
|---|---|---|
| "Please disarm the alarm" | `disarm_alarm` | 11.3 s |
| "Register another user Michael" | `register_user("Michael")` | 12.0 s |
| "What time is it?" | `get_time` | 36.6 s |

Two turns, not one: the model calls a tool, reads the result, then writes prose. The spread is the
length of the *answer*, nothing else — the third one wrote a long sentence about the date.

Context is **4096 tokens total**, prompt plus generation, compiled into the model. Nine tool
schemas cost roughly 600 of them, which is why `app/agent.py` builds a fresh agent per question
rather than keeping the conversation.
