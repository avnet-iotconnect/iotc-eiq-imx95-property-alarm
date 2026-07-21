"""Draw detection boxes and an FPS heads-up display onto the live preview, using Cairo.

The `cairooverlay` element in the preview pipeline fires a `draw` callback once per displayed
frame, handing us a Cairo context to paint on. We read the most recent detections and timing from
a shared `OverlayState` (written by the detection loop) and draw them. Because the preview and the
inference run on separate frames, the boxes lag the live image by a frame or two - expected.
"""

from __future__ import annotations

import cairo


class OverlayState:
    """The latest thing to draw. Written each frame by the detection loop, read by the draw callback."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.detections: list[tuple[str, float, list[int]]] = []
        self.inference_ms = 0.0
        self.end_to_end_ms = 0.0

    @property
    def inference_fps(self) -> float:
        return 1000.0 / self.inference_ms if self.inference_ms else 0.0

    @property
    def end_to_end_fps(self) -> float:
        return 1000.0 / self.end_to_end_ms if self.end_to_end_ms else 0.0


def draw_overlay(context: cairo.Context, state: OverlayState) -> None:
    _draw_boxes(context, state.detections)
    _draw_fps(context, state)


def _draw_boxes(context: cairo.Context, detections: list[tuple[str, float, list[int]]]) -> None:
    context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    context.set_font_size(16)
    context.set_line_width(2.0)
    for label, score, (x1, y1, x2, y2) in detections:
        context.set_source_rgb(0.0, 1.0, 0.4)  # green
        context.rectangle(x1, y1, x2 - x1, y2 - y1)
        context.stroke()
        _draw_text(context, f"{label} {score:.2f}", x1 + 3, max(y1 - 5, 14), (0.0, 1.0, 0.4))


def _draw_fps(context: cairo.Context, state: OverlayState) -> None:
    context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    context.set_font_size(18)
    lines = [
        state.backend,
        f"inference : {state.inference_fps:5.0f} fps  ({state.inference_ms:4.1f} ms)",
        f"end-to-end: {state.end_to_end_fps:5.0f} fps  ({state.end_to_end_ms:4.1f} ms)",
    ]
    for row, line in enumerate(lines):
        _draw_text(context, line, 10, 24 + row * 22, (1.0, 1.0, 0.0))  # yellow


def _draw_text(context: cairo.Context, text: str, x: float, y: float, rgb: tuple[float, float, float]) -> None:
    """Text with a 1px black shadow so it stays legible over any background."""
    context.set_source_rgb(0.0, 0.0, 0.0)
    context.move_to(x + 1, y + 1)
    context.show_text(text)
    context.set_source_rgb(*rgb)
    context.move_to(x, y)
    context.show_text(text)
