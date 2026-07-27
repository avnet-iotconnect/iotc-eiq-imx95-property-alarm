"""'Describe scene': the current camera frame, with our boxes burned in, handed to SmolVLM.

The VLM is the one piece here that is genuinely slow - seconds, not milliseconds - so it lives well
off the frame loop: it only ever runs on a command thread, when someone asks for it. That is the
same shape the alarm narration will use later ("describe the thief"), which is why this is a small
object with one public method rather than something wired into the video path.

Three things are worth knowing about how it is driven:

**The boxes go into the pixels.** `overlay.annotate_frame` paints the tracker's boxes and names into
a copy of the frame before it is saved. The Cairo overlay on the HDMI preview cannot be read back,
and a model asked about rectangles it cannot see will still answer confidently.

**The question carries the count.** We tell the model how many boxes there are and what we think
they are, then ask it to describe them. If it comes back describing two people when we drew one box,
the mismatch is the signal - that is the check this stage exists to make possible.

**The image is rebound, not reloaded.** NXP's `VLM.__init__` takes a `fixed_image` and precomputes
`image_inputs` from it, which would mean a 12-second model load per picture. Swapping `image_inputs`
and clearing `image_features` makes `process_message` recompute the vision pass for the new image on
its next call (`modeling_vlm.py:214-217`), which is the same work their own code does, without the
reload. The keyframe is written to disk on the way through - `/IOTCONNECT` S3 upload will want
exactly that file later.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from threading import Lock
from time import perf_counter
from types import SimpleNamespace

import cv2

from overlay import annotate_frame
from tracking import Track

logger = logging.getLogger(__name__)


class SceneDescriber:
    """SmolVLM behind one method: `describe(frame, tracks)` -> a sentence to speak."""

    def __init__(self, payload: Path, weights_dir: Path, keyframe_path: Path,
                 max_new_tokens: int = 64, num_threads: int = 4) -> None:
        self.payload = payload
        self.weights_dir = weights_dir
        self.keyframe_path = keyframe_path
        self.max_new_tokens = max_new_tokens
        self.num_threads = num_threads
        self._model = None
        self._lock = Lock()  # one describe at a time; the pool has four threads, the VLM has one

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Build the model (~12 s, ~1 GB). Called on first use, or up front via `--preload-vlm`."""
        import vlm.modeling_vlm as modeling_vlm
        from vlm.modeling_vlm import make_VLM

        # models_config.py points models_dir inside the payload, which we treat as read-only and
        # replace wholesale on every refresh. Redirect the download to our own tree instead.
        self.weights_dir.mkdir(exist_ok=True)
        modeling_vlm.models_dir = str(self.weights_dir) + "/"

        params = SimpleNamespace(
            n_threads=self.num_threads,  # not -1: the camera and YOLO need cores of their own
            use_neutron=False,           # eIQ's VLM is CPU/onnxruntime only on this build
            max_new_tokens=self.max_new_tokens,
            assistant_prompt="How can I help you?",
        )
        # A VLM must be constructed with *an* image; the first real keyframe replaces it. NXP's
        # reference shot is always present in the payload, so use that as the bootstrap.
        bootstrap = self.payload / "testdata" / "delivery.jpg"
        started = perf_counter()
        self._model = make_VLM("smolvlm-256M", "q8", user_params=params, fixed_image=str(bootstrap))
        print(f"[vlm] smolvlm-256M q8 loaded in {perf_counter() - started:.1f}s "
              f"(weights in {self.weights_dir})")

    def describe(self, frame_rgb, tracks: list[Track]) -> str:
        """Annotate the frame, ask SmolVLM about it, and return what to say out loud."""
        with self._lock:
            if not self.is_loaded:
                self.load()
            summary = summarise(tracks)
            cv2.imwrite(str(self.keyframe_path), annotate_frame(frame_rgb, tracks))

            question = (
                f"This image has {len(tracks)} objects highlighted with colored rectangles: {summary}. "
                "Describe what you see, including what the people look like and what they are wearing."
            )
            started = perf_counter()
            self._set_image(self.keyframe_path)
            description = "".join(self._model.process_message(question)).strip()
            print(f"[vlm] {perf_counter() - started:.1f}s | {len(tracks)} boxes | {description!r}")

        # The count is spoken by us, the description by the model: hearing both makes it obvious
        # when the model is describing something we never detected.
        return f"I can see {summary or 'nothing'}. {description}"

    def _set_image(self, path: Path) -> None:
        """Point the loaded model at a new picture without rebuilding it (see the module docstring)."""
        from transformers.image_utils import load_image

        self._model.image = load_image(str(path))
        self._model.image_inputs = self._model.processor.image_processor(
            [[self._model.image]], **{"return_row_col_info": True, "return_tensors": "np"})
        self._model.image_features = None  # forces the vision pass to rerun for this image


def summarise(tracks: list[Track]) -> str:
    """'2 person (Marija, unknown), 1 laptop' - what we believe is on screen, in speakable form."""
    if not tracks:
        return ""
    counts = Counter(track.class_name for track in tracks)
    named = [track.identity for track in tracks if track.identity]
    parts = [f"{count} {class_name}" for class_name, count in counts.most_common()]
    summary = ", ".join(parts)
    return f"{summary} ({', '.join(named)} recognised)" if named else summary
