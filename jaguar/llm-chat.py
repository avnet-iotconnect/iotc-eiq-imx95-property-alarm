#!/usr/bin/env python3
"""Run one of NXP's pre-compiled Ara240 LLMs and stream the answer.

The model directory holds a `model.dvm` (weights already quantised and compiled for the DNPU),
a `config.json` that names the model_type and how to reach the proxy, and a `tokenizer/` copy of
the stock Hugging Face tokenizer. Only the tokenizer is a stock Transformers object; the model is
loaded by `AraModelForCausalLM`, which opens a session on the Ara240 proxy and streams tokens back.

Usage:
    .venv/bin/python llm-chat.py "Write a haiku about NPUs"
    .venv/bin/python llm-chat.py --model /usr/share/llm/Qwen2.5-7B-Instruct "Who are you?"
"""

import argparse
import sys
import time

_process_start = time.monotonic()

from transformers import AutoTokenizer, TextStreamer

from optimum.ara import AraGenerationConfig, AraModelForCausalLM

_imports_done = time.monotonic()


def log_stage(label: str, since: float) -> float:
    """Print how long a startup stage took and return the clock for the next one."""
    now = time.monotonic()
    print(f"[stage] {label}: {now - since:.1f}s", flush=True)
    return now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Write a Python function that reverses a string.")
    parser.add_argument("--model", default="/usr/share/llm/Qwen2.5-Coder-1.5B")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"[stage] import torch/transformers/optimum.ara: {_imports_done - _process_start:.1f}s", flush=True)
    mark = _imports_done
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model}/tokenizer")
    mark = log_stage("load tokenizer", mark)
    model = AraModelForCausalLM.from_pretrained(args.model)
    mark = log_stage("load model.dvm onto Ara240", mark)
    generation_config = AraGenerationConfig.from_pretrained(args.model)

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    print(f"\n=== PROMPT ===\n{args.prompt}\n=== ANSWER ===")
    streamer = TextStreamer(tokenizer, stream=sys.stdout, skip_prompt=True)
    model.generate(
        **inputs,
        streamer=streamer,
        generation_config=generation_config,
        max_new_tokens=args.max_new_tokens,
    )

    print()
    model.display_perf_statistics()
    del model


if __name__ == "__main__":
    main()
