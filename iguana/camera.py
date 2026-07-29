"""Camera capture via GStreamer: one pipeline, one place the picture is composed.

    v4l2src -> tee -+-> queue -> videoconvert -> appsink            frames for inference (RGB)
                    |
                    +-> queue -> videoconvert -> cairooverlay -> tee -+-> videoconvert -> waylandsink
                                                (the OSD)             |      HDMI preview
                                                                      +-> v4l2h264enc -> appsink
                                                                      |      WebRTC (encoded)
                                                                      +-> appsink
                                                                             snapshots (BGRx)

A USB camera can only be opened once, so the first `tee` splits capture from inference. The second
`tee` is what iguana adds, and it is the whole trick behind streaming the display:

**Everything downstream of `cairooverlay` has the OSD burned in.** hyena could not read its preview
back - the compositor will not return those pixels - so `overlay.annotate_frame` re-drew the boxes
in OpenCV whenever something outside the screen needed the picture. Tapping *after* the overlay
instead means the WebRTC viewer and the S3 snapshot both get literally what is on the HDMI screen,
at the application's own resolution, with no second renderer to keep in sync.

The encoded branch matters as much. `v4l2h264enc` is the i.MX95's hardware encoder, so the WebRTC
feed is compressed by the VPU and no frame is ever copied into Python for it. See `webrtc.py` for
the other half of that.
"""

from __future__ import annotations

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402  (gi needs require_version first)

Gst.init(None)

# Two seconds between IDR frames. aiortc cannot ask a hardware encoder for a keyframe (see
# webrtc.py), so this is the worst case a viewer waits before the first picture appears; we also
# force one the moment somebody connects, which is what actually makes it feel instant.
KEYFRAME_INTERVAL_S = 2
CAPTURE_FRAMERATE = 30
STREAM_BITRATE_BPS = 2_000_000  # generous for 640x480; the VPU does not care and neither does a LAN


class Camera:
    """Owns the GStreamer pipeline: RGB frames for inference, and the composed display feed."""

    def __init__(self, device: str, width: int, height: int, show_preview: bool = True,
                 is_streaming: bool = False) -> None:
        # The camera speaks YUY2 (YUYV) natively; videoconvert turns it into RGB for numpy and into
        # whatever the sinks want. leaky=downstream + small queues keep us on the newest frame
        # instead of building latency - which is the whole game for a sub-500ms WebRTC feed.
        self.pipeline = Gst.parse_launch(build_pipeline_description(
            device, width, height, show_preview, is_streaming))
        self.sink = self.pipeline.get_by_name("sink")
        self.display_sink = self.pipeline.get_by_name("display_sink")
        self.encoder = self.pipeline.get_by_name("encoder")
        self.width = width
        self.height = height

    def connect_overlay(self, draw_callback) -> None:
        """Call `draw_callback(cairo_context)` once per composed frame, to paint the OSD on it."""
        overlay = self.pipeline.get_by_name("overlay")
        if overlay is not None:
            overlay.connect("draw", lambda _element, context, _timestamp, _duration: draw_callback(context))

    def connect_encoded(self, on_packet) -> None:
        """Call `on_packet(bytes)` for each encoded H.264 access unit, on GStreamer's own thread.

        A callback rather than a `read()` because the WebRTC side is asyncio and must not block a
        pipeline thread waiting for it. `webrtc.py` does nothing here but hand the bytes to whoever
        is watching, so there is no work of ours on this thread at all.
        """
        encoded_sink = self.pipeline.get_by_name("encoded_sink")
        if encoded_sink is None:
            return
        encoded_sink.set_property("emit-signals", True)
        encoded_sink.connect("new-sample", lambda sink: _pull_encoded(sink, on_packet))

    def request_keyframe(self) -> None:
        """Ask the encoder for an IDR now, so a viewer that just connected sees a picture at once."""
        if self.encoder is None:
            return
        self.encoder.send_event(GstVideo.video_event_new_upstream_force_key_unit(
            Gst.CLOCK_TIME_NONE, True, 0))

    def start(self) -> None:
        self.pipeline.set_state(Gst.State.PLAYING)

    def read(self, timeout_seconds: float = 5.0) -> np.ndarray | None:
        """Pull the next frame as an (H, W, 3) uint8 RGB array, or None on timeout/EOS."""
        sample = self.sink.emit("try-pull-sample", int(timeout_seconds * Gst.SECOND))
        if sample is None:
            return None
        return _sample_to_array(sample, self.height, self.width, channels=3)

    def read_display(self) -> np.ndarray | None:
        """The composed picture - OSD and all - as RGB, or None if nothing has been composed yet.

        Only pulled when something asks (a snapshot), so the branch costs a dropped buffer per frame
        and nothing else. BGRx is what `cairooverlay` hands on; dropping the padding byte and
        reversing the channels is a numpy view, not a conversion.
        """
        if self.display_sink is None:
            return None
        sample = self.display_sink.emit("try-pull-sample", 0)
        if sample is None:
            return None
        composed = _sample_to_array(sample, self.height, self.width, channels=4)
        return composed[:, :, 2::-1].copy() if composed is not None else None  # BGRx -> RGB

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)


def build_pipeline_description(device: str, width: int, height: int, show_preview: bool,
                               is_streaming: bool) -> str:
    """The pipeline as a string, kept separate so it can be read (and tested) without a camera."""
    # Nothing wants the composed picture, so do not compose one. Overlaying costs a YUY2->BGRx
    # conversion per frame whether or not anyone ever looks at the result, and this is the
    # configuration a benchmark run uses.
    if not show_preview and not is_streaming:
        composed = "t. ! queue ! fakesink sync=false"
    else:
        composed = compose_branch(show_preview, is_streaming)
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw,format=YUY2,width={width},height={height},"
        f"framerate={CAPTURE_FRAMERATE}/1 ! "
        f"tee name=t "
        f"{composed} "
        "t. ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! "
        "video/x-raw,format=RGB ! appsink name=sink emit-signals=false max-buffers=1 drop=true"
    )


def compose_branch(show_preview: bool, is_streaming: bool) -> str:
    """Overlay once, then fan the composed picture out to the screen, the encoder and snapshots.

    Every `videoconvert` here would be a full-frame software colour conversion on the CPU that YOLO
    is trying to use, so there is exactly one: YUY2 -> BGRx, in front of the overlay. **The format
    is pinned rather than negotiated.** `cairooverlay` accepts BGRx, ARGB or RGB16 and will happily
    settle on any of them depending on what the sinks want, which would leave `read_display`
    unpacking whichever one it got as if it were BGRx. Pinning it makes that assumption true, and
    means the encoder and the appsink are handed pixels they take natively - `v4l2h264enc` lists
    BGRx among its input formats, so the VPU is fed with no conversion at all.
    """
    branches = ["queue max-size-buffers=1 leaky=downstream ! "
                "appsink name=display_sink max-buffers=1 drop=true sync=false"]
    if show_preview:
        branches.append("queue max-size-buffers=2 leaky=downstream ! videoconvert ! "
                        "waylandsink sync=false")
    if is_streaming:
        # h264_profile=4 is Constrained Baseline, the profile every WebRTC viewer accepts.
        # config-interval=-1 repeats SPS/PPS before every IDR, so a viewer that joins late can
        # decode without having heard the start of the stream. alignment=au gives us one whole
        # access unit per buffer, which is exactly one video frame for aiortc to packetise.
        branches.append(
            f"queue max-size-buffers=2 leaky=downstream ! "
            f"v4l2h264enc name=encoder "
            f"extra-controls=\"controls,h264_profile=4,video_bitrate={STREAM_BITRATE_BPS},"
            f"h264_i_frame_period={KEYFRAME_INTERVAL_S * CAPTURE_FRAMERATE}\" ! "
            f"h264parse config-interval=-1 ! "
            f"video/x-h264,stream-format=byte-stream,alignment=au ! "
            f"appsink name=encoded_sink max-buffers=2 drop=true sync=false")

    return ("t. ! queue max-size-buffers=2 leaky=downstream ! videoconvert ! "
            "video/x-raw,format=BGRx ! cairooverlay name=overlay ! tee name=d "
            + " ".join(f"d. ! {branch}" for branch in branches))


def _buffer_bytes(buffer) -> bytes:
    """Copy a GStreamer buffer out as plain bytes.

    `extract_dup` rather than the usual `buffer.map(...)` / `buffer.unmap(mapinfo)` pair, and the
    reason is worth recording. NXP's eIQ payload imports `gi` and GStreamer for its own audio
    backend, so with voice enabled there are two live `Gst` typelib wrappers in the process, and
    handing a `MapInfo` produced by one to the other fails with

        TypeError: argument info: Expected Gst.MapInfo, but got gi.repository.Gst.MapInfo

    - which is only reachable in the full configuration, and so went unnoticed in gazelle and
    hyena, where only `--no-voice` was ever run. `extract_dup` never materialises a MapInfo, and
    since every caller here copies the data anyway, it costs nothing to be immune.
    """
    return buffer.extract_dup(0, buffer.get_size())


def _sample_to_array(sample, height: int, width: int, channels: int) -> np.ndarray | None:
    frame = np.frombuffer(_buffer_bytes(sample.get_buffer()), dtype=np.uint8)
    if frame.size != height * width * channels:
        return None
    return frame.reshape((height, width, channels))


def _pull_encoded(sink, on_packet) -> Gst.FlowReturn:
    """appsink `new-sample` handler: hand the access unit on and get off this thread."""
    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR
    on_packet(_buffer_bytes(sample.get_buffer()))
    return Gst.FlowReturn.OK
