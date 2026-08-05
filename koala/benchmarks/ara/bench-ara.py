"""Measure TTFT and decode rate through the eIQ AAF Connector.

Streams a completion and times the first token separately from the rest, so
prefill cost does not get averaged into the decode rate. Compare against the
direct-optimum numbers in work/STATUS-jaguar.md.

Run it from the host, with the connector serving on the board:

    .venv/bin/python koala/benchmarks/ara/bench-ara.py

Counting caveat: chunks are not quite tokens. The connector's tool-call state machine buffers while
a `<tool_call>` prefix is still possible, so a 151-token answer can arrive as 119 chunks. For an
exact figure trust the server's own `usage.completion_tokens` on a non-streamed request.
"""

import time

from openai import OpenAI

client = OpenAI(api_key="unused", base_url="http://192.168.38.203:3000/v1")
MODEL = "Qwen2.5-7B-Instruct"
PROMPT = "Explain in about 150 words what an NPU is and why edge AI benefits from one."

start = time.time()
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    extra_body={"llm_params": {"generate_max_tokens": 200}},
    stream=True,
)

ttft = None
chunks = 0
text = []
for chunk in stream:
    for ch in chunk.choices:
        if ch.delta and ch.delta.content:
            if ttft is None:
                ttft = time.time() - start
                decode_start = time.time()
            chunks += 1
            text.append(ch.delta.content)

total = time.time() - start
decode = total - ttft

print("".join(text))
print("\n--- %s ---" % MODEL)
print(f"chunks (tokens) : {chunks}")
print(f"TTFT            : {ttft:.2f} s")
print(f"decode          : {decode:.2f} s")
print(f"decode rate     : {chunks / decode:.2f} tok/s")
print(f"wall clock      : {total:.2f} s")
