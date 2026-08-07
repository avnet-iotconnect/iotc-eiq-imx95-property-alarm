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

from applib.tracking import Track, color_for


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
        self.alarm_state = AlarmState.DISARMED
        self.locked_objects: list[str] = []
        self.voice_status = "voice: loading"
        self.cloud_status = "cloud: off"
        self.stream_status = "stream: off"
        self.agent_status = "agent: off"
        self.inference_ms = 0.0
        self.end_to_end_ms = 0.0

    def set_tracks(self, tracks: list[Track]) -> None:
        self.tracks = tracks  # atomic reference swap; the draw callback reads the latest

    def set_alarm_state(self, state: AlarmState) -> None:
        self.alarm_state = state

    def set_locked_objects(self, class_names: list[str]) -> None:
        self.locked_objects = class_names

    def set_voice_status(self, status: str) -> None:
        """One short line about the voice stack - it takes ~15 s to load, and silence looks broken."""
        self.voice_status = status

    def set_cloud_status(self, status: str) -> None:
        """Same idea for /IOTCONNECT: whether the dashboard is seeing this board is worth a line."""
        self.cloud_status = status

    def set_stream_status(self, status: str) -> None:
        """How many people are watching over WebRTC - and this line is *in* what they are watching,
        because everything drawn here is downstream of the encoder tap (see camera.py)."""
        self.stream_status = status

    def set_agent_status(self, status: str) -> None:
        """And for the LLM on the Ara-240 - 'thinking' is worth showing, since it takes ~30 s."""
        self.agent_status = status

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
        for track in self.tracks:
            x1, y1, x2, y2 = track.box
            color = color_for(track.short_id)
            context.set_line_width(2.0)
            context.set_source_rgb(*color)
            context.rectangle(x1, y1, x2 - x1, y2 - y1)
            context.stroke()
            _draw_text(context, self._label_for(track), x1 + 3, max(y1 - 5, 14), color)
            self._draw_face_box(context, track)

    def _draw_face_box(self, context: cairo.Context, track: Track) -> None:
        """Debug: show the detected face inside the person - cyan box + best match cosine (or 'no match')."""
        if track.face_box is None:
            return
        fx1, fy1, fx2, fy2 = track.face_box
        context.set_line_width(1.5)
        context.set_source_rgb(0.15, 0.90, 0.90)  # cyan
        context.rectangle(fx1, fy1, fx2 - fx1, fy2 - fy1)
        context.stroke()

    def _label_for(self, track: Track) -> str:
        """Debug-friendly: recognized name, else the best cosine, else face-seen, else the bare class."""
        tag = f"#{track.short_id}" if track.short_id else "#-"
        if track.identity is not None:
            score = f" ({track.match_score:.2f})" if track.match_score is not None else ""
            return f"{tag} {track.identity}{score}"  # score is None on the just-registered frame
        if track.match_score is not None:
            return f"{tag} ? {track.match_score:.2f}"          # face seen, closest user below threshold
        if track.face_box is not None:
            return f"{tag} face (no users)"                     # face detected but nobody registered yet
        return f"{tag} {track.class_name} {track.score:.2f}"

    def get_hud_lines(self) -> list[tuple[str, tuple[float, float, float]]]:
        """The HUD as (text, color) pairs, so Cairo and the JPEG snapshot draw the same thing."""
        yellow, cyan = (1.0, 1.0, 0.0), (0.15, 0.90, 0.90)
        return [
            (self.backend, yellow),
            (f"end-to-end: {_fps(self.end_to_end_ms):5.0f} fps  ({self.end_to_end_ms:4.1f} ms)", yellow),
            (f"alarm: {self.alarm_state.label}", self.alarm_state.color),
            (f"locked: {', '.join(self.locked_objects) or '-'}", yellow),
            (self.voice_status, cyan),
            (self.cloud_status, cyan),
            (self.stream_status, cyan),
            (self.agent_status, cyan),
        ]

    def _draw_hud(self, context: cairo.Context) -> None:
        context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(18)
        for index, (text, color) in enumerate(self.get_hud_lines()):
            _draw_text(context, text, 10, 24 + index * 22, color)

    def _draw_alarm_banner(self, context: cairo.Context) -> None:
        """Big centered red 'ALARM' - fires when armed and a person is present but no user recognized."""
        context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(72)
        text = "ALARM"
        extents = context.text_extents(text)
        x = (self.frame_width - extents.width) / 2 - extents.x_bearing
        y = (self.frame_height - extents.height) / 2 - extents.y_bearing
        _draw_text(context, text, x, y, (1.0, 0.15, 0.15))


def annotate_frame(frame_rgb, tracks: list[Track], hud_lines=()):
    """Burn the boxes, labels and (optionally) the HUD into a *copy* of the frame, as BGR.

    This re-renders in OpenCV what Cairo already drew on the preview, and it is needed less than it
    once was. Tapping the pipeline *after* `cairooverlay` (see camera.py) means the composed
    picture can now be read back, so `snapshot` prefers the real thing and only falls back here.

    What still needs it is `scene.py`: `USER_GOAL.md` asks for "describe the thief highlighted in
    the orange rectangle", and asking a VLM about boxes it cannot see is how you get a confident
    answer about the wrong thing. The VLM wants the *inference* frame with boxes on it, which is a
    different picture from the display feed - so this stays.

    Same colors as the preview, so the file matches what the visitor sees on the HDMI screen.
    """
    import cv2

    annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for track in tracks:
        x1, y1, x2, y2 = track.box
        red, green, blue = color_for(track.short_id)
        bgr = (int(blue * 255), int(green * 255), int(red * 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
        label = f"{track.identity or track.class_name}"
        cv2.putText(annotated, label, (x1 + 2, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
    for index, (text, (red, green, blue)) in enumerate(hud_lines):
        origin = (10, 20 + index * 18)
        cv2.putText(annotated, text, (origin[0] + 1, origin[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)  # shadow, as in Cairo
        cv2.putText(annotated, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (int(blue * 255), int(green * 255), int(red * 255)), 1, cv2.LINE_AA)
    return annotated


def save_snapshot(path, frame_rgb, tracks: list[Track], hud_lines=()) -> str:
    """Re-render the frame with boxes and HUD, and write it as a JPEG. Returns the path."""
    import cv2

    cv2.imwrite(str(path), annotate_frame(frame_rgb, tracks, hud_lines))
    return str(path)


def save_composed(path, frame_rgb) -> str:
    """Write an already-composed frame - the one off the display feed - straight to a JPEG.

    Nothing is drawn here, because the OSD is already in these pixels. This is the snapshot that
    genuinely matches the HDMI screen, down to the font.
    """
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    return str(path)


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
