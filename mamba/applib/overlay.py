"""Draw tracked faces + the alarm-state HUD onto the live preview, using Cairo (the "Video" lineage).

The `cairooverlay` element fires `draw` once per displayed preview frame with a Cairo context. The
`Overlay` object owns the latest thing to paint (tracks, alarm state, timing) and renders it. `app.py`
pushes updates in via `set_tracks` / `set_alarm_state`; `main.py` pushes timing. The overlay never
reads back into the app - one-way, matching the composition-root pattern.

Alarm presentation lives here on purpose: which colour a state is drawn in, and where the REC mark
sits, are display concerns. `app.py` decides *what is true*; this file decides how it looks.
"""

from __future__ import annotations

from enum import Enum
from math import ceil, pi
from time import monotonic

import cairo

from applib.tracking import Track, color_for

# What a finished command flashes on the screen: how long it stays, and how much of it fits on one
# line across the frame at 24 pt. An `agent` answer is a couple of sentences and has to be cut.
ACTION_MESSAGE_S = 2.0
ACTION_MAX_CHARS = 46

# Where the HUD starts. The top offset is deliberately twice the left margin: a person detected at
# the top of the frame has their label drawn just above their box, which lands in the same corner,
# and one of the two has to give way. The HUD is the one that can afford to.
HUD_MARGIN_X = 10
HUD_TOP_Y = 48
HUD_LINE_HEIGHT = 22
HUD_SEGMENT_GAP = 14  # between two differently-coloured pieces of one line (ARMED and the fps)

# A line of HUD text may be several pieces in different colours - `ARMED` in the alarm's own colour
# beside the frame rate in yellow. One type for both, so a one-colour line is a list of one.
HudSegment = tuple[str, tuple[float, float, float]]


class AlarmState(Enum):
    """The alarm is a *switch*, and these are its two positions (label + HUD color, RGB 0..1).

    There is deliberately no third "ALARM" state. What the alarm has *caught* is the alert (a
    non-empty event log) and whether something is happening right now is the recording - two other
    things entirely, shown as their own HUD line and as REC in the corner. Collapsing all three into
    one red word is what made koala's screen impossible to read: a recognised user could be standing
    in front of a camera that still said ALARM, with nothing on the screen saying why.
    """

    DISARMED = ("DISARMED", (0.20, 0.85, 0.30))  # green
    ARMED = ("ARMED", (1.00, 0.85, 0.10))  # yellow

    def __init__(self, label: str, color: tuple[float, float, float]) -> None:
        self.label = label
        self.color = color


class Overlay:
    """Holds the latest tracks/state/timing and paints them onto each preview frame."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tracks: list[Track] = []
        self.alarm_state = AlarmState.DISARMED
        self.locked_objects: list[str] = []
        self.alert_text = ""
        self.is_recording = False
        self.voice_status = "voice: loading"
        self.cloud_status = "cloud: off"
        self.stream_status = "stream: off"
        self.agent_status = "agent: off"
        self.inference_ms = 0.0
        self.end_to_end_ms = 0.0
        self.countdown_message = ""
        self.countdown_until = 0.0
        self.action_message = ""
        self.action_until = 0.0
        self.is_action_ok = True

    def set_tracks(self, tracks: list[Track]) -> None:
        self.tracks = tracks  # atomic reference swap; the draw callback reads the latest

    def set_alarm_state(self, state: AlarmState) -> None:
        self.alarm_state = state

    def set_locked_objects(self, class_names: list[str]) -> None:
        """What is guarded, as the watchdog describes it - a missing one arrives as 'laptop (gone)'."""
        self.locked_objects = class_names

    def set_alert(self, text: str) -> None:
        """The newest thing in the event log, or '' when the log is empty - the alert, in one line."""
        self.alert_text = text

    def set_recording(self, is_recording: bool) -> None:
        """Whether the REC mark is up. The watchdog decides; this only draws it (see watchdog.py)."""
        self.is_recording = is_recording

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
        """Both already averaged over the last sixty frames - `main.py` owns the window, not this.

        `inference_ms` is what the frame loop ran *in order*, which today is YOLO. The face pass is
        on its own thread and a spare core, so it is concurrent with this loop rather than behind
        it, and counting it here would report more work per frame than a frame has room for.
        """
        self.inference_ms = inference_ms
        self.end_to_end_ms = end_to_end_ms

    def start_countdown(self, message: str, seconds: float) -> None:
        """Ask the person for something, and show how long they have to do it. Registration uses it.

        A deadline rather than a number, so the caller sets this once and the digit still ticks down
        on every drawn frame. That matters because the caller is a command handler on another
        thread, sleeping through exactly this interval - the one place in the demo where the person
        is being asked to hold a pose, and where the screen is the only thing that can tell them for
        how long.
        """
        self.countdown_message = message
        self.countdown_until = monotonic() + seconds

    def stop_countdown(self) -> None:
        self.countdown_until = 0.0

    def show_action(self, message: str, is_ok: bool = True) -> None:
        """Flash what a command just did, for two seconds. Every action's answer comes through here.

        Feedback rather than a log: a newer action replaces whatever is up and restarts the two
        seconds, because the only thing worth reading at a booth is the last thing that happened.
        Refusals are shown too, in red - "Michael is already registered" is exactly the sentence
        somebody standing in front of the camera needs, and until now it was only spoken.

        Which commands get here at all is `main.py`'s decision, not this file's - the watchdog's
        own three-second screenshots are filtered out there.

        A deadline rather than a timer, as with the countdown, so the display thread needs nothing
        from the command thread that set it. The three fields are written without a lock; at worst
        one frame shows a new message in the previous one's colour.
        """
        self.action_message = _shorten(message, ACTION_MAX_CHARS)
        self.is_action_ok = is_ok
        self.action_until = monotonic() + ACTION_MESSAGE_S

    def draw(self, context: cairo.Context) -> None:
        """The cairooverlay `draw` callback: boxes, the HUD, REC, and whatever is flashing at the bottom."""
        self._draw_boxes(context)
        self._draw_hud(context)
        if self.is_recording:
            self._draw_recording(context)
        self._draw_countdown(context)
        self._draw_action(context)

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

    def get_hud_lines(self) -> list[list[HudSegment]]:
        """The HUD as lines of coloured segments, so Cairo and the JPEG snapshot draw the same thing.

        **The alert is first, and it is there only when there is one** - in red, with no label in
        front of it. A red sentence at the top of the screen is already unmistakably an alert, and
        an "alert: -" that is present nine hundred and ninety-nine frames out of a thousand teaches
        whoever is watching to stop reading the top line. The rest of the HUD moves up a line when
        nothing has happened, which is the price and it is worth paying.

        The alarm is the bare word, in its own colour: this HUD used to say "alarm: ARMED", and the
        label carries the noun better than the prefix did. The frame rate rides on the same line, in
        yellow, because it is a number you check rather than read.
        """
        yellow, cyan, red = (1.0, 1.0, 0.0), (0.15, 0.90, 0.90), (1.0, 0.25, 0.25)
        alert = [[(self.alert_text, red)]] if self.alert_text else []
        return alert + [
            [(self.alarm_state.label, self.alarm_state.color), (self._timing_text(), yellow)],
            [(f"locked: {', '.join(self.locked_objects) or '-'}", yellow)],
            [(self.voice_status, cyan)],
            [(self.cloud_status, cyan)],
            [(self.stream_status, cyan)],
            [(self.agent_status, cyan)],
        ]

    def _timing_text(self) -> str:
        """`30(50) 33ms` - what the demo is doing, what the frame loop's models could do, per frame.

        The braced number is `inference_ms` read back as a frame rate, and it is a **sequential**
        budget: 20 ms of YOLO is (50), so it can never read worse than the rate beside it. The gap
        between the two is the headroom left for another model on this thread, which is the question
        it is here to answer. Whole milliseconds - the tenths moved with every frame and meant
        nothing, since both figures are already averaged over sixty of them.
        """
        return (f"{_fps(self.end_to_end_ms):.0f}({_fps(self.inference_ms):.0f})"
                f" {self.end_to_end_ms:.0f}ms")

    def _draw_hud(self, context: cairo.Context) -> None:
        context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(18)
        for index, segments in enumerate(self.get_hud_lines()):
            x = HUD_MARGIN_X
            for text, color in segments:
                _draw_text(context, text, x, HUD_TOP_Y + index * HUD_LINE_HEIGHT, color)
                x += context.text_extents(text).x_advance + HUD_SEGMENT_GAP

    def _draw_countdown(self, context: cairo.Context) -> None:
        """'Registering Nick - look at the camera  2', centered along the bottom while it runs.

        Bottom, not middle: the person is looking at the camera above the screen, so this sits where
        it does not cover the face they are checking, nor the ALARM banner. What each attempt
        *measured* is not drawn - that is debug, and it goes to the console.
        """
        remaining = self.countdown_until - monotonic()
        if remaining <= 0:
            return
        context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(28)
        self._draw_centered(context, f"{self.countdown_message}  {ceil(remaining)}",
                            self.frame_height - 24, (1.0, 1.0, 1.0))

    def _draw_action(self, context: cairo.Context) -> None:
        """'Registered Nick.' along the bottom for two seconds, white, or red when it was refused.

        A line above the countdown rather than sharing it: the two can be up at once, because an
        `agent` question runs outside the command lock and its answer can land while somebody else
        is being counted down. Nothing here is ever the *only* copy of the message - voice speaks
        it and the dashboard gets it as an ack - so cutting it to one line loses nothing.
        """
        if monotonic() >= self.action_until:
            return
        context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(24)
        color = (1.0, 1.0, 1.0) if self.is_action_ok else (1.0, 0.25, 0.25)
        self._draw_centered(context, self.action_message, self.frame_height - 62, color)

    def _draw_centered(self, context: cairo.Context, text: str, y: float,
                       rgb: tuple[float, float, float]) -> None:
        """Draw one line centred across the frame, at the font size the caller has already set."""
        extents = context.text_extents(text)
        _draw_text(context, text, (self.frame_width - extents.width) / 2 - extents.x_bearing, y, rgb)

    def _draw_recording(self, context: cairo.Context) -> None:
        """A red dot and REC in the top-right corner, while the watchdog is uploading screenshots.

        This is the demo's "something is happening" mark - it replaced the big red ALARM banner,
        which said only that something had happened *at some point* and would not go away.

        Steady rather than blinking: half the frames of a blinking mark have nothing in them, and
        one of those frames is eventually the screenshot that gets uploaded, which then looks like
        the mark is broken. The corner is the free one - the HUD is down the left and the countdown
        is along the bottom.
        """
        context.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(20)
        red = (1.0, 0.15, 0.15)
        x = self.frame_width - 66
        context.set_source_rgb(*red)
        context.arc(x, 18, 7, 0, 2 * pi)
        context.fill()
        _draw_text(context, "REC", x + 12, 25, red)


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
    for index, segments in enumerate(hud_lines):
        x = 10
        for text, (red, green, blue) in segments:
            origin = (x, 40 + index * 18)
            cv2.putText(annotated, text, (origin[0] + 1, origin[1] + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)  # shadow, as in Cairo
            cv2.putText(annotated, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (int(blue * 255), int(green * 255), int(red * 255)), 1, cv2.LINE_AA)
            (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            x += width + 10
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


def _shorten(text: str, limit: int) -> str:
    """One line, short enough to fit across the frame. Handler answers can be a paragraph."""
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[:limit - 3].rstrip() + "..."


def _draw_text(context: cairo.Context, text: str, x: float, y: float, rgb: tuple[float, float, float]) -> None:
    """Text with a 1px black shadow so it stays legible over any background."""
    context.set_source_rgb(0.0, 0.0, 0.0)
    context.move_to(x + 1, y + 1)
    context.show_text(text)
    context.set_source_rgb(*rgb)
    context.move_to(x, y)
    context.show_text(text)
