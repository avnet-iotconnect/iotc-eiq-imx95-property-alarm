#!/usr/bin/env python3
"""Narrate a short video clip with Qwen2.5-VL-7B on the Ara240 — the VLM offload path.

Unlike the text models, this one loads two `.dvm` files: an 11.2 GB language model and a separate
1.06 GB vision encoder, named by `vision_model_path` in the model's `config.json`. The vision
encoder is fed 336x336 frames in pairs, and its embeddings are spliced into the token stream where
the `<|video_pad|>` tokens sit.

NXP's build is video-oriented (their own benchmark uses an 8 second clip), so a still keyframe has
to be presented as a two-frame clip rather than as an image.

Record a clip first:
    gst-launch-1.0 -e v4l2src device=/dev/video2 num-buffers=180 ! \\
        video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! v4l2h264enc ! \\
        video/x-h264,level=(string)4 ! h264parse ! mp4mux ! filesink location=clip.mp4

Usage:
    .venv/bin/python vlm-describe.py --video clip.mp4
"""

import argparse
import time

_process_start = time.monotonic()

from optimum.ara import AraGenerationConfig, AraQwen2_5_VLForConditionalGeneration, QwenVLProcessor

_imports_done = time.monotonic()

MODEL_PATH = "/usr/share/llm/Qwen2.5-VL-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="clip.mp4")
    parser.add_argument("--prompt", default="Describe what happens in this video.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"[stage] imports: {_imports_done - _process_start:.1f}s", flush=True)

    # Point the processor at the local tokenizer copy; the default is a Hugging Face repo id,
    # which would need the board to reach the internet on every run.
    mark = time.monotonic()
    processor = QwenVLProcessor(model_id=f"{MODEL_PATH}/tokenizer")
    print(f"[stage] load processor: {time.monotonic() - mark:.1f}s", flush=True)

    mark = time.monotonic()
    model = AraQwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH)
    generation_config = AraGenerationConfig.from_pretrained(MODEL_PATH)
    print(f"[stage] load model.dvm + vision encoder onto Ara240: {time.monotonic() - mark:.1f}s", flush=True)

    mark = time.monotonic()
    processed_inputs = processor(args.prompt, args.video)
    print(f"[stage] decode and preprocess video (CPU): {time.monotonic() - mark:.1f}s", flush=True)

    print(f"\n=== PROMPT ===\n{args.prompt}\n=== ANSWER ===", flush=True)
    result = model.generate(
        **processed_inputs, generation_config=generation_config, max_new_tokens=args.max_new_tokens
    )
    print(processor.decode(result.flatten(), skip_special_tokens=True))

    model.display_perf_statistics()
    del model


if __name__ == "__main__":
    main()
