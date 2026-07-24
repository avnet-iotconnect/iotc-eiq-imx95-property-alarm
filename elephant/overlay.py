"""Draw tracked faces + the alarm-state HUD onto the live preview, using Cairo (the "Video" lineage).

The `cairooverlay` element fires `draw` once per displayed preview frame with a Cairo context. The
`Overlay` object owns the latest thing to paint (tracks, alarm state, timing) and renders it. `app.py`
pushes updates in via `set_tracks` / `set_alarm_state`; `main.py` pushes timing. The overlay never
reads back into the app - one-way, matching the composition-root pattern.

Alarm presentation lives here on purpose: color-per-state and the big centered "ALARM" banner are
display concerns. `app.py` decides *which* state we're in; this file decides how it looks.
"""

from __future__ import annotations

from enum import Enum

import cairo

from tracking import Track, color_for


class AlarmState(Enum):
    """The demo's alarm states, each carrying how it should look (label + HUD color, RGB 0..1)."""

    DISARMED = ("DISARMED", (0.20, 0.85, 0.30))  # green
    ARMED = ("ARMED", (1.00, 0.85, 0.10))  # yellow
    ALARM = ("ALARM", (1.00, 0.20, 0.20))  # red

    def __init__(self, label: str, color: tuple[float, float, float]) -> None:
        self.label = label
        self.color = color


class Overlay:
    """Holds the latest tracks/state/timing and paints them onto each preview frame."""

    def __init__(self, frame_width: int, frame_height: int, backend: str) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.backend = backend
        self.tracks: list[Track] = []
        self.alarm_state = AlarmState.ARMED
        self.inference_ms = 0.0
        self.end_to_end_ms = 0.0

    def set_tracks(self, tracks: list[Track]) -> None:
        self.tracks = tracks  # atomic reference swap; the draw callback reads the latest

    def set_alarm_state(self, state: AlarmState) -> None:
        self.alarm_state = state

    def set_timing(self, inference_ms: float, end_to_end_ms: float) -> None:
        self.inference_ms = inference_ms
        self.end_to_end_ms = end_to_end_ms

    def draw(self, context: cairo.Context) -> None:
        """The cairooverlay `draw` callback: paint boxes, the HUD, and the alarm banner if armed-tripped."""
        self._draw_boxes(context)
        self._draw_hud(context)
        if self.alarm_state is AlarmState.ALARM:
            self._draw_alarm_banner(context)

    def _draw_boxes(self, context: cairo.Context) -> None:
        context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(16)
        context.set_line_width(2.0)
        for track in self.tracks:
            x1, y1, x2, y2 = track.box
            color = color_for(track.short_id)
            context.set_source_rgb(*color)
            context.rectangle(x1, y1, x2 - x1, y2 - y1)
            context.stroke()
            _draw_text(context, self._label_for(track), x1 + 3, max(y1 - 5, 14), color)

    def _label_for(self, track: Track) -> str:
        """A recognized person shows their name; anyone/anything else shows the bare id + class."""
        tag = f"#{track.short_id}" if track.short_id else "#-"
        if track.identity is not None:
            return f"{tag} {track.identity}"
        return f"{tag} {track.class_name} {track.score:.2f}"

    def _draw_hud(self, context: cairo.Context) -> None:
        context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(18)
        _draw_text(context, self.backend, 10, 24, (1.0, 1.0, 0.0))
        _draw_text(context, f"end-to-end: {_fps(self.end_to_end_ms):5.0f} fps  "
                   f"({self.end_to_end_ms:4.1f} ms)", 10, 46, (1.0, 1.0, 0.0))
        _draw_text(context, f"alarm: {self.alarm_state.label}", 10, 68, self.alarm_state.color)

    def _draw_alarm_banner(self, context: cairo.Context) -> None:
        """Big centered red 'ALARM' - fires when armed and a person is present but no user recognized."""
        context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(72)
        text = "ALARM"
        extents = context.text_extents(text)
        x = (self.frame_width - extents.width) / 2 - extents.x_bearing
        y = (self.frame_height - extents.height) / 2 - extents.y_bearing
        _draw_text(context, text, x, y, (1.0, 0.15, 0.15))


def _fps(ms: float) -> float:
    return 1000.0 / ms if ms else 0.0


def _draw_text(context: cairo.Context, text: str, x: float, y: float, rgb: tuple[float, float, float]) -> None:
    """Text with a 1px black shadow so it stays legible over any background."""
    context.set_source_rgb(0.0, 0.0, 0.0)
    context.move_to(x + 1, y + 1)
    context.show_text(text)
    context.set_source_rgb(*rgb)
    context.move_to(x, y)
    context.show_text(text)
