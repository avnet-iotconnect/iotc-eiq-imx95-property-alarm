"""Which `/dev/videoN` is the camera, asked at startup rather than written down.

The number moves. This board's own IP blocks claim fifty-odd video nodes before the USB camera gets
one -- `neoisp` alone takes /dev/video2 through /dev/video51 -- and how many they take depends on
the BSP, so a camera that was /dev/video4 on one image is /dev/video52 on the next. Hard-coding it
means the demo comes up with "camera returned no frame" on a board that is working perfectly.

`v4l2-ctl --list-devices` groups the nodes by the device that owns them, and names the bus each one
is on:

    neoisp (platform:4ae00000.isp):
            /dev/video2
            ...
    HD Pro Webcam C920 (usb-ci_hdrc.0-1):
            /dev/video52
            /dev/video53

**The bus is the discriminator, not the name.** Excluding the names we know (`neoisp`, `mxc-jpeg`,
`wave6-*`) would work on this image and quietly break on the next one that adds an IP block. Every
built-in block is on `platform:`, and a USB webcam -- whichever one the booth has -- is on `usb-`.
So: the first device on a USB bus, whatever it is called.

That still leaves two nodes. A UVC camera exposes a second one for metadata, and opening it gets
you a pipeline that negotiates and then never produces a frame; it is distinguishable because it
enumerates no pixel formats at all. So each candidate node is asked what it can do, and the first
one that can actually capture wins.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

# "HD Pro Webcam C920 (usb-ci_hdrc.0-1):" -> name, bus. Names contain spaces and dashes; the bus is
# the last parenthesised group on the line, and always present.
DEVICE_HEADER = re.compile(r"^(?P<name>.+) \((?P<bus>[^()]+)\):$")


@dataclass
class VideoDevice:
    """One device from `v4l2-ctl --list-devices`, with the `/dev/video*` nodes it owns."""

    name: str
    bus: str
    nodes: list[str] = field(default_factory=list)

    @property
    def is_usb(self) -> bool:
        return self.bus.startswith("usb-")

    def __str__(self) -> str:
        return f"{self.name} ({self.bus})"


def find_camera_device() -> tuple[str, VideoDevice] | None:
    """The first USB capture node and the device it belongs to, or None if no camera is attached."""
    for device in list_video_devices():
        if not device.is_usb:
            continue
        for node in device.nodes:
            if is_capture_node(node):
                return node, device
    return None


def list_video_devices() -> list[VideoDevice]:
    """Every device `v4l2-ctl --list-devices` reports, in its order. `/dev/media*` is dropped."""
    listing = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True).stdout
    devices: list[VideoDevice] = []
    for line in listing.splitlines():
        header = DEVICE_HEADER.match(line)
        if header:
            devices.append(VideoDevice(header["name"].strip(), header["bus"]))
        elif devices and line.strip().startswith("/dev/video"):
            devices[-1].nodes.append(line.strip())
    return devices


def is_capture_node(node: str) -> bool:
    """True if `node` enumerates at least one capture format, i.e. it is a node frames come out of.

    The C920's /dev/video52 answers with `[0]: 'YUYV' (YUYV 4:2:2)` and its /dev/video53, the
    metadata node, answers with nothing after the header line.
    """
    formats = subprocess.run(["v4l2-ctl", "-d", node, "--list-formats"],
                             capture_output=True, text=True).stdout
    return "[0]:" in formats


def describe_devices() -> str:
    """What was found, one device per line -- the useful half of "no camera found".

    Node lists are cut short because `neoisp` alone owns fifty of them, and a wall of numbers is
    not what somebody looking for their camera needs to read.
    """
    lines = []
    for device in list_video_devices():
        shown = ", ".join(device.nodes[:4])
        if len(device.nodes) > 4:
            shown += f", ... (+{len(device.nodes) - 4} more)"
        lines.append(f"  {device}: {shown or 'no video nodes'}")
    return "\n".join(lines) or "  (v4l2-ctl reported nothing)"
