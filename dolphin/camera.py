"""Camera capture via GStreamer: one pipeline, two outputs.

    v4l2src -> tee -+-> queue -> videoconvert -> waylandsink   (live HDMI preview)
                    +-> queue -> videoconvert -> appsink        (frames for inference)

A USB camera can only be opened once, so `tee` splits the stream: waylandsink
renders the live preview on its own thread while we pull frames off the appsink
for inference. GStreamer (with GLib underneath) does the plumbing; we only ask
the appsink for the next frame and hand back an RGB numpy array.
"""

from __future__ import annotations

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)


class Camera:
    """Owns the GStreamer pipeline and yields decoded RGB frames one at a time."""

    def __init__(self, device: str, width: int, height: int, show_preview: bool = True) -> None:
        # The camera speaks YUY2 (YUYV) natively; videoconvert turns it into RGB for numpy
        # and (on the preview branch) into whatever waylandsink wants. leaky=downstream +
        # small queues keep us on the newest frame instead of building latency.
        preview_branch = (
            "t. ! queue max-size-buffers=2 leaky=downstream ! videoconvert ! waylandsink sync=false"
            if show_preview
            else "t. ! queue ! fakesink sync=false"
        )
        pipeline_description = (
            f"v4l2src device={device} ! "
            f"video/x-raw,format=YUY2,width={width},height={height},framerate=30/1 ! "
            f"tee name=t "
            f"{preview_branch} "
            "t. ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=sink emit-signals=false max-buffers=1 drop=true"
        )
        self.pipeline = Gst.parse_launch(pipeline_description)
        self.sink = self.pipeline.get_by_name("sink")
        self.width = width
        self.height = height

    def start(self) -> None:
        self.pipeline.set_state(Gst.State.PLAYING)

    def read(self, timeout_seconds: float = 5.0) -> np.ndarray | None:
        """Pull the next frame as an (H, W, 3) uint8 RGB array, or None on timeout/EOS."""
        sample = self.sink.emit("try-pull-sample", int(timeout_seconds * Gst.SECOND))
        if sample is None:
            return None
        buffer = sample.get_buffer()
        is_mapped, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not is_mapped:
            return None
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
            frame = frame.reshape((self.height, self.width, 3)).copy()
        finally:
            buffer.unmap(mapinfo)
        return frame

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)
