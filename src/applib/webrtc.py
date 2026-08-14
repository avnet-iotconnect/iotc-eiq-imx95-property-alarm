"""KVS WebRTC: the display feed, live in a browser, with about a quarter of a second of lag.

The device is the **master** on an Amazon Kinesis Video Streams signalling channel. /IOTCONNECT
hands us the channel ARN and temporary AWS credentials (see `iotc.py`); we keep one signed WebSocket
open to that channel for the whole run, and every viewer that presses play on the dashboard arrives
as an `SDP_OFFER` on it. Nothing here starts or stops with the back end's "streaming" flag - the
channel is either configured or it is not.

    /IOTCONNECT ---> channel ARN + AWS credentials ---> this file
    GStreamer   ---> encoded H.264 access units     ---> this file ---> RTP ---> viewer

## Why there is no encoder in this file

`camera.py` hands us H.264 that the i.MX95's VPU already compressed, and aiortc takes it as-is:
`RTCRtpSender` checks what `recv()` returned and calls `Encoder.pack()` - packetise only - for
anything that is not a raw `Frame`. So a WebRTC frame is never copied into Python, never converted,
and never encoded on the CPU that YOLO is using. Two consequences fall out of that, and both are
handled here rather than by aiortc:

- **aiortc cannot ask for a keyframe.** `force_keyframe` is only consulted on the encode path, so a
  viewer joining mid-stream would wait for the encoder's own IDR interval. We ask GStreamer for one
  the moment a peer connects, and again whenever a viewer falls behind.
- **the RTCP bitrate hint is ignored.** The bitrate is whatever `camera.py` configured. Fine for a
  booth; it is the thing to revisit on a bad network.

## Why the timestamps are wall-clock

The obvious way to stamp frames - `pts += 1` at `time_base = 1/30` - is what the predecessor project
did, and it is why that one buffered up to ten seconds. If the pipeline actually delivers 24 fps,
every wall-clock second produces 0.8 s of media time, and the viewer falls *cumulatively* further
behind for as long as it runs. Stamping each access unit with the real time it arrived, at 90 kHz,
means media time and wall-clock time cannot drift apart no matter what the frame rate does.
"""

from __future__ import annotations

import asyncio
import json
import logging
from base64 import b64decode, b64encode
from fractions import Fraction
from threading import Thread
from time import monotonic
from typing import Callable

logger = logging.getLogger(__name__)

try:
    import av
    import boto3
    import websockets
    from aiortc import (RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCRtpSender,
                        RTCSessionDescription)
    from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
    from aiortc.sdp import candidate_from_sdp
    from botocore.auth import SigV4QueryAuth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials
    IS_WEBRTC_AVAILABLE = True
except ImportError:  # the demo runs without it, exactly as it runs without the /IOTCONNECT SDK
    IS_WEBRTC_AVAILABLE = False
    MediaStreamTrack = object

VIDEO_CLOCK_RATE = 90000     # the RTP clock for video, fixed by the standard
SIGNED_URL_TTL_S = 299       # KVS rejects a signature older than 5 minutes
RECONNECT_WAIT_S = 5.0
QUEUE_DEPTH = 4              # access units a viewer may fall behind before we resynchronise it

# A viewer in any of these is gone as far as we are concerned. `disconnected` is included on
# purpose: a browser closed with the X does not say goodbye, and waiting for ICE consent to expire
# instead leaves it counted as "watching" for another half minute. WebRTC does allow `disconnected`
# to recover on its own, but a viewer that comes back simply sends a fresh offer - which costs one
# handshake and is a great deal easier to reason about than a peer we are holding open on spec.
DEAD_STATES = ("disconnected", "failed", "closed")


class WebRtcStreamer:
    """One signalling channel, held open, and one peer connection per viewer watching."""

    def __init__(
        self, on_keyframe_wanted: Callable[[], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.on_keyframe_wanted = on_keyframe_wanted or (lambda: None)
        self.on_status = on_status or (lambda status: None)
        self.channel_arn = ""
        self.region = ""
        self.get_credentials: Callable[[], object] = lambda: None

        self._viewers: dict[str, tuple[RTCPeerConnection, "H264Track"]] = {}
        self._tracks: set[H264Track] = set()
        self.endpoints: dict[str, str] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._thread = Thread(target=self._run, name="webrtc", daemon=True)
        self._is_stopping = False

    # --- lifecycle ------------------------------------------------------------------------------

    def configure(self, channel_arn: str, get_credentials: Callable[[], object]) -> None:
        """Point this streamer at a channel. Separate from `start` so `iotc-check.py` can use the
        signing and endpoint lookup without running a stream."""
        self.channel_arn = channel_arn
        self.region = region_of(channel_arn)
        self.get_credentials = get_credentials  # pulled, never pushed - see the note in iotc.py

    def start(self, channel_arn: str, get_credentials: Callable[[], object]) -> None:
        """Begin streaming on this channel. Called by `iotc.py` once /IOTCONNECT hands over the ARN.

        `main.py` builds and wires this object, but only the cloud knows which channel to use and
        it does not know it until it has connected - so construction and starting are separate.
        Calling it again after a reconnect is a no-op: the channel does not change under us.
        """
        if self._thread.is_alive():
            return
        self.configure(channel_arn, get_credentials)
        self.on_status("stream: connecting")
        self._thread.start()

    def stop(self) -> None:
        """Cancel the signalling task and let the loop wind itself up.

        Stopping the event loop outright - the obvious thing - leaves websockets' keepalive task
        and every peer connection half-torn-down, and asyncio complains loudly about both while the
        rest of the demo is trying to shut down cleanly. Cancelling instead gives `_runner` a
        chance to close the viewers first.
        """
        self._is_stopping = True
        if self._loop is not None and self._task is not None:
            self._loop.call_soon_threadsafe(self._task.cancel)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def on_encoded(self, access_unit: bytes) -> None:
        """GStreamer's thread, once per encoded frame. Hand it to the event loop and return.

        Timestamped here, on arrival, rather than on send: this is the closest thing we have to
        when the picture existed, and it is what the wall-clock note at the top is about.

        The hop onto the loop is what keeps every track queue purely asyncio. Feeding a
        `queue.Queue` across threads instead means `recv()` has to block a worker thread to read
        it, and a worker blocked on a track nobody will ever write to again cannot be joined at
        interpreter shutdown - which showed up as a demo that needed two Ctrl-Cs, but only ever
        after somebody had watched the stream.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        timestamp = monotonic()
        try:
            loop.call_soon_threadsafe(self._dispatch, access_unit, timestamp)
        except RuntimeError:  # the loop closed between the check and the call
            pass

    def _dispatch(self, access_unit: bytes, timestamp: float) -> None:
        """Event-loop thread: give the frame to every viewer currently watching."""
        for track in tuple(self._tracks):
            track.offer(access_unit, timestamp)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._runner())
        except Exception:  # a dead stream must never take the demo with it
            logger.exception("webrtc thread failed")
        finally:
            self._loop.close()

    async def _runner(self) -> None:
        """The one task the loop runs, so that `stop` has something to cancel."""
        self._task = asyncio.current_task()
        try:
            await self._signalling_loop()
        except asyncio.CancelledError:
            pass
        for viewer_id in list(self._viewers):
            await self._drop_viewer(viewer_id)

    # --- the signalling channel -----------------------------------------------------------------

    async def _signalling_loop(self) -> None:
        """Stay connected to the channel for the whole run, reconnecting for as long as we live."""
        while not self._is_stopping:
            try:
                url = await asyncio.to_thread(self.build_signed_url)
                async with websockets.connect(url) as socket:
                    print(f"[webrtc] master on {self.channel_arn.rsplit('/', 2)[-2]}, waiting for viewers")
                    self.on_status("stream: ready")
                    async for message in socket:
                        await self._on_signal(message, socket)
            except Exception as error:
                if self._is_stopping:
                    return
                logger.exception("signalling channel dropped")
                print(f"[webrtc] signalling: {type(error).__name__}: {error} -- "
                      f"retrying in {RECONNECT_WAIT_S:.0f}s")
                self.on_status("stream: offline")
                await asyncio.sleep(RECONNECT_WAIT_S)

    def build_signed_url(self) -> str:
        """A SigV4-signed wss:// URL for the MASTER role. Blocking (botocore), so called off-loop.

        No `X-Amz-ClientId`: that parameter identifies a *viewer*. A master connects with the
        channel ARN alone, and KVS rejects the signed request if it carries one.
        """
        credentials = self.get_credentials()
        if not self.endpoints:
            self.endpoints = self._describe_endpoints(credentials)
        request = AWSRequest(method="GET", url=self.endpoints["WSS"],
                             params={"X-Amz-ChannelARN": self.channel_arn})
        SigV4QueryAuth(as_botocore_credentials(credentials), "kinesisvideo",
                       self.region, SIGNED_URL_TTL_S).add_auth(request)
        return request.prepare().url

    def _describe_endpoints(self, credentials) -> dict[str, str]:
        """Ask KVS where this channel's master endpoints live. Cached: they do not move."""
        client = self._aws_client("kinesisvideo", credentials)
        response = client.get_signaling_channel_endpoint(
            ChannelARN=self.channel_arn,
            SingleMasterChannelEndpointConfiguration={"Protocols": ["HTTPS", "WSS"],
                                                      "Role": "MASTER"})
        return {entry["Protocol"]: entry["ResourceEndpoint"]
                for entry in response["ResourceEndpointList"]}

    def _aws_client(self, service: str, credentials, endpoint_url: str | None = None):
        return boto3.client(service, region_name=self.region, endpoint_url=endpoint_url,
                            aws_access_key_id=credentials.access_key_id,
                            aws_secret_access_key=credentials.secret_access_key,
                            aws_session_token=credentials.session_token)

    def fetch_ice_servers(self) -> list[RTCIceServer]:
        """KVS's STUN server plus the TURN relays it hands out. Blocking; called off-loop.

        A viewer on the same LAN never needs the relays and a lot of internet pairs do not either,
        but the credentials cost one call and are the difference between "works from a phone on
        mobile data" and "works at my desk".
        """
        credentials = self.get_credentials()
        client = self._aws_client("kinesis-video-signaling", credentials,
                                  endpoint_url=self.endpoints["HTTPS"])
        config = client.get_ice_server_config(ChannelARN=self.channel_arn, ClientId="MASTER")
        servers = [RTCIceServer(urls=f"stun:stun.kinesisvideo.{self.region}.amazonaws.com:443")]
        servers.extend(RTCIceServer(urls=server["Uris"], username=server["Username"],
                                    credential=server["Password"])
                       for server in config["IceServerList"])
        return servers

    async def _on_signal(self, message: str, socket) -> None:
        message_type, payload, viewer_id = decode_signal(message)
        if message_type == "SDP_OFFER":
            await self._on_offer(payload, viewer_id, socket)
        elif message_type == "ICE_CANDIDATE":
            await self._on_ice_candidate(payload, viewer_id)

    # --- one viewer -----------------------------------------------------------------------------

    async def _on_offer(self, payload: dict, viewer_id: str, socket) -> None:
        """A viewer pressed play: build them a peer connection carrying the encoded display feed."""
        print(f"[webrtc] viewer {viewer_id} connecting")
        await self._drop_viewer(viewer_id)  # a reload sends a second offer for the same id

        ice_servers = await asyncio.to_thread(self.fetch_ice_servers)
        peer = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        track = H264Track(self.on_keyframe_wanted)
        self._viewers[viewer_id] = (peer, track)
        self._tracks.add(track)

        @peer.on("connectionstatechange")
        async def on_state_change() -> None:
            print(f"[webrtc] viewer {viewer_id}: {peer.connectionState}")
            if peer.connectionState in DEAD_STATES:
                await self._drop_viewer(viewer_id)
            self._report_viewers()

        peer.addTrack(track)
        # Must be pinned *before* the offer is applied: setRemoteDescription is where aiortc works
        # out the common codec list, and after that the preference is never looked at again. We
        # have exactly one thing to send and it is H.264, so offering VP8 could only mislead.
        pin_h264(peer)
        await peer.setRemoteDescription(RTCSessionDescription(sdp=payload["sdp"],
                                                              type=payload["type"]))
        await peer.setLocalDescription(await peer.createAnswer())
        await socket.send(encode_signal("SDP_ANSWER", peer.localDescription, viewer_id))
        self.on_keyframe_wanted()  # so the first picture arrives now, not at the next IDR
        self._report_viewers()

    async def _on_ice_candidate(self, payload: dict, viewer_id: str) -> None:
        viewer = self._viewers.get(viewer_id)
        if viewer is None or not payload.get("candidate"):
            return
        candidate = candidate_from_sdp(payload["candidate"])
        candidate.sdpMid = payload.get("sdpMid")
        candidate.sdpMLineIndex = payload.get("sdpMLineIndex")
        await viewer[0].addIceCandidate(candidate)

    async def _drop_viewer(self, viewer_id: str) -> None:
        """Forget a viewer, and stop encoding for them *first*.

        Order matters: the track leaves `_tracks` before the peer is closed, so no frame is handed
        to a connection that is being torn down. Anything still in flight ends up sent to a socket
        whose far end has gone, which asyncio reports as `socket.send() raised exception`.
        """
        viewer = self._viewers.pop(viewer_id, None)
        if viewer is None:
            return
        peer, track = viewer
        self._tracks.discard(track)
        await peer.close()

    def _report_viewers(self) -> None:
        watching = sum(1 for peer, _ in self._viewers.values()
                       if peer.connectionState == "connected")
        self.on_status(f"stream: {watching} watching" if watching else "stream: ready")


class H264Track(MediaStreamTrack):
    """One viewer's view of the encoded feed: a small queue of access units, flushed on overflow.

    `recv()` returns an `av.Packet`, which is the signal to `RTCRtpSender` that this is already
    encoded and only needs packetising. Everything here happens on the event loop - see
    `WebRtcStreamer.on_encoded` for why that matters more than it looks.
    """

    kind = "video"

    def __init__(self, on_keyframe_wanted: Callable[[], None]) -> None:
        super().__init__()
        self.on_keyframe_wanted = on_keyframe_wanted
        self._queue: asyncio.Queue = asyncio.Queue()
        self._started_at: float | None = None

    def offer(self, access_unit: bytes, timestamp: float) -> None:
        """Event-loop thread. Never blocks, and never grows without bound.

        When a viewer falls behind we throw away everything queued rather than trickling out stale
        frames: dropping the middle of an H.264 stream corrupts every frame until the next IDR, so
        the honest recovery is to skip to now and ask for a fresh keyframe.
        """
        if self._queue.qsize() >= QUEUE_DEPTH:
            drain(self._queue)
            self.on_keyframe_wanted()
        self._queue.put_nowait((access_unit, timestamp))

    def stop(self) -> None:
        """Wake `recv()` so the sender task can finish, instead of waiting on a dead queue forever.

        aiortc calls this when the peer connection closes. Without the sentinel the sender is left
        awaiting a queue nothing will ever put to again, and that task keeps the connection's
        teardown from completing.
        """
        super().stop()
        self._queue.put_nowait(None)

    async def recv(self) -> av.Packet:
        item = await self._queue.get()
        if item is None:
            raise MediaStreamError  # the track was stopped; aiortc unwinds the sender on this
        access_unit, timestamp = item
        if self._started_at is None:
            self._started_at = timestamp
        packet = av.Packet(access_unit)
        packet.pts = int((timestamp - self._started_at) * VIDEO_CLOCK_RATE)
        packet.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        return packet


def pin_h264(peer: RTCPeerConnection) -> None:
    """Offer H.264 and nothing else on the video transceiver."""
    codecs = [codec for codec in RTCRtpSender.getCapabilities("video").codecs
              if codec.mimeType == "video/H264"]
    for transceiver in peer.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(codecs)


def decode_signal(message: str) -> tuple[str, dict, str]:
    """KVS wraps each signalling message's payload in base64'd JSON."""
    try:
        envelope = json.loads(message)
        payload = json.loads(b64decode(envelope["messagePayload"]).decode("utf-8"))
        return envelope["messageType"], payload, envelope.get("senderClientId", "")
    except (ValueError, KeyError):
        return "", {}, ""


def encode_signal(action: str, description, viewer_id: str) -> str:
    payload = {"type": description.type, "sdp": description.sdp}
    return json.dumps({
        "action": action,
        "messagePayload": b64encode(json.dumps(payload).encode("utf-8")).decode("ascii"),
        "recipientClientId": viewer_id,
    })


def as_botocore_credentials(credentials):
    """The SDK's credential object, in the shape botocore's signer wants."""
    return Credentials(access_key=credentials.access_key_id,
                       secret_key=credentials.secret_access_key,
                       token=credentials.session_token)


def region_of(channel_arn: str) -> str:
    """'arn:aws:kinesisvideo:us-east-1:2600...:channel/nik-imx95-01/178...' -> 'us-east-1'."""
    return channel_arn.split(":")[3]


def drain(queue: asyncio.Queue) -> None:
    try:
        while True:
            queue.get_nowait()
    except asyncio.QueueEmpty:
        pass
