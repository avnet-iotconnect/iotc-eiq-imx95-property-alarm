#!/usr/bin/env python3
"""Test whether an Ara240-compiled Qwen2.5 model can drive tools — the prerequisite for MCP.

Runs one full round trip: the model is offered two anti-theft tools, is expected to emit a
`<tool_call>` block, we execute it locally, feed the result back as a `tool` message, and let the
model answer in words. If this works, strands-agents / MCP are viable; if the model ignores the
tools or emits malformed JSON, they are not.

The tool protocol is Qwen's own, and it lives entirely in the chat template that ships in the
model's `tokenizer/` directory — nothing about it is Ara-specific. The model emits text; parsing
that text back into a call is the harness's job.

Usage:
    .venv/bin/python tool-call.py --model /usr/share/llm/Qwen2.5-7B-Instruct
"""

import argparse
import json
import re
import sys
import time

_process_start = time.monotonic()

from transformers import AutoTokenizer

from optimum.ara import AraGenerationConfig, AraModelForCausalLM

_imports_done = time.monotonic()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_alarm_state",
            "description": "Return the current state of the anti-theft alarm.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_alarm_state",
            "description": "Arm or disarm the anti-theft alarm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["armed", "disarmed"],
                        "description": "The state to put the alarm into.",
                    }
                },
                "required": ["state"],
            },
        },
    },
]

alarm_state = "disarmed"


def get_alarm_state() -> str:
    return alarm_state


def set_alarm_state(state: str) -> str:
    global alarm_state
    alarm_state = state
    return f"alarm is now {alarm_state}"


TOOL_IMPLEMENTATIONS = {"get_alarm_state": get_alarm_state, "set_alarm_state": set_alarm_state}

WRAPPED_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
BARE_CALL_PATTERN = re.compile(r"\{[^{}]*\"name\"\s*:.*?\"arguments\"\s*:\s*\{[^{}]*\}[^{}]*\}", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls, tolerating models that omit Qwen's `<tool_call>` wrapper.

    The 7B Instruct model wraps its calls as trained; the base Coder model emits the same JSON
    bare. Accepting both keeps the harness from reporting a protocol failure as a model failure.
    """
    raw_calls = WRAPPED_CALL_PATTERN.findall(text)
    if not raw_calls:
        raw_calls = BARE_CALL_PATTERN.findall(text)

    calls = []
    for raw in raw_calls:
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            print(f"  !! malformed tool call JSON: {exc}\n     raw: {raw!r}")
    return calls


def generate_reply(model, tokenizer, messages: list[dict], generation_config, max_new_tokens: int) -> str:
    """Render the conversation through the chat template, generate, return only the new text."""
    text = tokenizer.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    prompt_length = inputs["input_ids"].shape[-1]

    output = model.generate(
        **inputs, generation_config=generation_config, max_new_tokens=max_new_tokens
    )

    # Ara models have returned both "prompt + completion" and "completion only" across versions;
    # trim the prompt only when it is actually there.
    ids = output.flatten()
    if ids.shape[-1] > prompt_length:
        ids = ids[prompt_length:]

    # Decode with the special tokens left in: `<tool_call>` is what we are here to look for, and
    # seeing `<|im_end|>` verbatim tells us the model stopped on its own rather than hit the cap.
    reply = tokenizer.decode(ids, skip_special_tokens=False)
    return reply.replace("<|endoftext|>", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/usr/share/llm/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", default="Please arm the alarm, then tell me what state it is in.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"[stage] imports: {_imports_done - _process_start:.1f}s", flush=True)
    mark = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model}/tokenizer")
    model = AraModelForCausalLM.from_pretrained(args.model)
    generation_config = AraGenerationConfig.from_pretrained(args.model)
    print(f"[stage] load model.dvm onto Ara240: {time.monotonic() - mark:.1f}s", flush=True)

    messages: list[dict] = [
        {"role": "system", "content": "You are the controller for an anti-theft alarm. Use the supplied tools."},
        {"role": "user", "content": args.prompt},
    ]

    print(f"\n=== USER ===\n{args.prompt}")

    for turn in range(1, 4):
        reply = generate_reply(model, tokenizer, messages, generation_config, args.max_new_tokens)
        print(f"\n=== MODEL (turn {turn}) ===\n{reply}")

        calls = parse_tool_calls(reply)
        if not calls:
            print("\n=== VERDICT ===\nNo tool call emitted; the model answered in prose.")
            break

        # The chat template re-adds the turn markers, so they must not go back in as content.
        messages.append({"role": "assistant", "content": reply.replace("<|im_end|>", "").strip()})
        for call in calls:
            name, arguments = call.get("name"), call.get("arguments", {})
            print(f"\n=== TOOL CALL ===\n{name}({arguments})")
            if name not in TOOL_IMPLEMENTATIONS:
                result = f"error: no such tool {name!r}"
            else:
                result = TOOL_IMPLEMENTATIONS[name](**arguments)
            print(f"=== TOOL RESULT ===\n{result}")
            messages.append({"role": "tool", "name": name, "content": result})
    else:
        print("\n=== VERDICT ===\nStopped after 3 turns without a final prose answer.")

    print(f"\nFinal alarm state: {alarm_state}")
    model.display_perf_statistics()
    del model


if __name__ == "__main__":
    main()
