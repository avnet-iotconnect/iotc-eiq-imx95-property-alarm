"""/IOTCONNECT: telemetry out, commands in, snapshots to S3. (the "iotconnect" lineage)

Everything cloud-shaped lives in this one file, on **one thread**, and the rest of the pilot does not
import it. That is the whole design, and it comes from two facts:

- The video loop must never wait on a network call. So producers write into
  `telemetry.TelemetryState` and this thread reads it - see the docstring there.
- The SDK's C2D callback runs on paho's MQTT thread. Anything slow done there stalls *all* MQTT
  traffic, including the acknowledgement we are about to send. So `_on_c2d` does three cheap things
  - look the verb up, build a `Command`, hand it to `CommandService` - and returns. The command
  runs on a command worker; the ack is sent when its future completes.

What flows each way:

    D2C   every 4 s (`TELEMETRY_INTERVAL_S`), or at once when something asks: sdk_version,
          version, fps, alarm, objects, `scene` when someone asked the VLM a question, and
          `answer` when someone asked the LLM one - or when the event log was read out or cleared,
          because clearing destroys it and it only ever existed in RAM
    C2D   the twelve commands in the alrmtheft device template, mapped to our verbs by `C2D_VERBS`;
          every one is acknowledged with the same sentence the demo would have spoken
    S3    `upload_capture()` puts capture.jpg in the bucket; /IOTCONNECT timestamps each version.
          The snapshot *handler* calls it, through a callable `main.py` handed the app - so saving
          and uploading are one command, and `app.py` still imports nothing from here
    KVS   the signalling channel ARN and the AWS credentials `webrtc.py` signs its socket with

Two acks are **shortened** here, and only here, because the dashboard shows an ack as a *tooltip*:
a sentence is readable there, a paragraph is not. `scene` says "Scene described." and the LLM's
answer is cut to `MAX_AGENT_ACK_CHARS` - enough to recognise which answer came back, no more.
Either way the text somebody actually wanted arrives as the `scene` or `answer` attribute -
**once**, by `set_once`, so a one-off answer is never repeated on the next tick.

`agent` is also the one command whose argument is a sentence rather than a name, and the SDK hands
arguments over already split on whitespace - so joining them back is what reconstructs the question.
Two commands here can take tens of seconds (`agent`, `scene`); nothing waits on them but the person
who sent them, because the ack goes out when the future completes.

The KVS half is a handover and nothing more. This file learns the channel ARN when it connects and
passes it, with a *callable* that returns current credentials, to the streamer `main.py` built. It
does not know what WebRTC is, and `webrtc.py` does not know what /IOTCONNECT is.

**The cloud is required.** A demo that quietly runs without a dashboard is how a booth discovers at
the worst moment that the certificate was wrong, so `connect()` raises and `main.py` does not catch
it: the process stops, printing whatever the SDK said. `--no-iotc` is the deliberate way to run the
vision half on its own. Once connected, a network that comes and goes is retried rather than fatal.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import TYPE_CHECKING, Callable

from applib import commands
from applib.commands import Command, CommandError, CommandService
from applib.telemetry import TelemetryState

if TYPE_CHECKING:  # imported for the type only: webrtc.py pulls in aiortc, which may not be there
    from webrtc import WebRtcStreamer

logger = logging.getLogger(__name__)

try:
    from avnet.iotconnect.sdk.lite import Callbacks, Client, ClientSettings, DeviceConfig
    from avnet.iotconnect.sdk.lite import __version__ as SDK_VERSION
    from avnet.iotconnect.sdk.sdklib.mqtt import C2dAck
    IS_SDK_AVAILABLE = True
except ImportError:  # the video half of the pilot must run on a board with no SDK installed
    SDK_VERSION = "not installed"
    IS_SDK_AVAILABLE = False

# The device template (files/alrmtheft.json) names the commands; this is the only place those names
# appear. Everything to the right is a verb `app.py` already implements - the cloud got a command
# path that was built for voice, and needed no new handlers except `restart`.
C2D_VERBS = {
    "user-register": commands.REGISTER_USER,
    "user-unregister": commands.UNREGISTER_USER,
    "object-lock": commands.LOCK_OBJECT,
    "object-unlock": commands.UNLOCK_OBJECT,
    "alarm-arm": commands.ARM,
    "alarm-disarm": commands.DISARM,
    "alert-clear": commands.CLEAR_ALERT,
    "alert-describe": commands.DESCRIBE_ALERT,
    "scene": commands.DESCRIBE_SCENE,
    "snapshot": commands.SNAPSHOT,
    "restart": commands.RESTART,
    "agent": commands.AGENT,
}

TELEMETRY_INTERVAL_S = 4.0
MIN_SEND_GAP_S = 1.0        # a flapping alarm state must not turn into an MQTT flood
RECONNECT_WAIT_S = 30.0     # after a failure that took the client down entirely
CREDENTIALS_MARGIN_S = 120  # refresh AWS credentials this long before they expire
MAX_ACK_CHARS = 200         # acks are a status line, not a transcript
MAX_AGENT_ACK_CHARS = 60    # ... and the LLM's answer is a paragraph; the rest is in `answer`


class IotcClient:
    """The /IOTCONNECT connection: one thread that publishes, and callbacks that hand work away."""

    def __init__(
        self, config_path: Path, service: CommandService, telemetry: TelemetryState,
        capture_path: Path, app_version: str, cert_path: Path, key_path: Path,
        streaming: "WebRtcStreamer | None" = None,
        on_status: Callable[[str], None] | None = None,
        on_stream_status: Callable[[str], None] | None = None,
        interval_s: float = TELEMETRY_INTERVAL_S, is_verbose: bool = False,
    ) -> None:
        self.config_path = config_path
        self.service = service
        self.telemetry = telemetry
        self.capture_path = capture_path
        self.app_version = app_version
        self.streaming = streaming  # started here, because only the cloud knows the channel ARN
        self.cert_path = cert_path
        self.key_path = key_path
        self.on_status = on_status or (lambda status: None)
        self.on_stream_status = on_stream_status or (lambda status: None)
        self.interval_s = interval_s
        self.is_verbose = is_verbose

        self._client: Client | None = None
        self._s3 = None
        self._kvs = None
        self._last_sent = 0.0
        self._stop = Event()
        self._thread = Thread(target=self._run, name="iotc", daemon=True)

    # --- lifecycle ----------------------------------------------------------------------------

    def connect(self) -> None:
        """Connect now, on the caller's thread, and **raise** if it cannot. Nothing is caught here.

        The cloud is not optional: `main.py` calls this before the camera or the models are opened,
        so a wrong certificate, an unreachable back end or a device that is disabled stops the demo
        immediately with the SDK's own message, rather than an hour later with a HUD line nobody
        read. `--no-iotc` is the way to run without it.
        """
        self.on_status("cloud: connecting")
        self._connect()

    def start(self) -> None:
        """Publish telemetry on our own thread, from here on. Call `connect()` first."""
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.telemetry.wake()  # so the publisher is not still sitting in its 4-second wait
        self._thread.join(timeout=10.0)
        if self._client is not None:
            self._client.disconnect()

    def _run(self) -> None:
        """The publisher. A connection that *drops* is retried - only the first one is fatal."""
        while not self._stop.is_set():
            try:
                if self._client is None:
                    self._connect()
                self._publish_loop()
            except Exception as error:  # a network that comes and goes must not end the demo
                logger.exception("iotconnect client failed")
                print(f"[iotc] {type(error).__name__}: {error} -- retrying in "
                      f"{RECONNECT_WAIT_S:.0f}s")
                self.on_status("cloud: offline")
                self._client = None
                self._stop.wait(RECONNECT_WAIT_S)

    # --- connecting ---------------------------------------------------------------------------

    def _connect(self) -> None:
        config = DeviceConfig.from_iotc_device_config_json_file(
            device_config_json_path=str(self.config_path),
            device_cert_path=str(self.cert_path),
            device_pkey_path=str(self.key_path),
        )
        self._client = Client(
            config=config,
            # No vs_cb: we do not take streaming start/stop instructions from the back end.
            # That flag suits PutMedia, where the device is the thing that decides to push video.
            # With WebRTC the viewer's browser drives the whole exchange, so the useful behaviour
            # is simply to be reachable - see `_prepare_aws`.
            callbacks=Callbacks(command_cb=self._on_c2d, ota_cb=self._on_ota,
                                disconnected_cb=self._on_disconnect),
            settings=ClientSettings(verbose=self.is_verbose),
        )
        self._client.connect()
        if not self._client.is_connected():
            raise ConnectionError("could not connect to /IOTCONNECT")
        print(f"[iotc] connected as {self._client.get_duid()} "
              f"(thing {self._client.get_client_id()}), SDK {SDK_VERSION}")
        self.on_status("cloud: online")
        self._prepare_aws()

    def _prepare_aws(self) -> None:
        """Fetch S3 and KVS credentials at connect, and hand the streamer its channel.

        The streaming half is deliberately *not* gated on the back end's streaming flag. A
        signalling channel costs nothing while nobody is watching, and the moment a viewer presses
        play the WebRTC handshake has to be answered - so the channel stays open for the whole run
        and `is_auto_start()` / `is_streaming()` are reported for information only.
        """
        self._s3 = self._client.get_s3_client()
        self._kvs = self._client.get_kvs_client()

        if self._s3 is None:
            print("[iotc] S3 disabled for this device (enable File Support in the template)")
        else:
            self._s3.obtain_credentials()
            bucket = self._s3.get_default_bucket()
            print(f"[iotc] S3 bucket {bucket.bucket_name if bucket else '(none)'}, credentials "
                  f"good for {self._s3.get_secs_to_expiry() / 60:.0f} min")

        if self._kvs is None:
            print("[iotc] KVS disabled for this device (enable Streaming in the template)")
            self.on_stream_status("stream: not enabled")
            return
        self._kvs.obtain_credentials()
        channel_arn = self._kvs.get_signaling_channel_arn()
        print(f"[iotc] KVS channel {channel_arn} (auto-start {self._kvs.is_auto_start()}), "
              f"credentials good for {self._kvs.get_secs_to_expiry() / 60:.0f} min")
        if self.streaming is None:
            self.on_stream_status("stream: off")
        elif not channel_arn:
            print("[iotc] no signalling channel -- enable Video Streaming -> WebRTC in the template")
            self.on_stream_status("stream: no channel")
        else:
            # A *callable*, not the credentials themselves: `_refresh_credentials` renews them on
            # this thread every few minutes, and the streamer signs each new WebSocket with
            # whatever is current. That is the whole credential handover - there is no push.
            self.streaming.start(
                channel_arn,
                lambda: self._kvs.get_credentials(refresh_if_secs_to_expiry=CREDENTIALS_MARGIN_S))

    def _refresh_credentials(self) -> None:
        """Keep both credential sets alive. They last an hour; we renew two minutes early."""
        for name, provider in (("S3", self._s3), ("KVS", self._kvs)):
            if provider is not None and provider.get_secs_to_expiry() < CREDENTIALS_MARGIN_S:
                print(f"[iotc] refreshing {name} credentials")
                provider.obtain_credentials()

    # --- publishing ---------------------------------------------------------------------------

    def _publish_loop(self) -> None:
        """Send every `interval_s`, or as soon as somebody calls `telemetry.wake()`."""
        while not self._stop.is_set():
            is_woken = self.telemetry.wait_for_wake(self.interval_s)
            if self._stop.is_set():
                return
            if is_woken:
                self._stop.wait(max(0.0, MIN_SEND_GAP_S - (monotonic() - self._last_sent)))
            if not self._client.is_connected():
                self.on_status("cloud: reconnecting")
                self._client.connect()
                self.on_status("cloud: online" if self._client.is_connected() else "cloud: offline")
            self._refresh_credentials()
            self._send_telemetry()

    def _send_telemetry(self) -> None:
        values = self.telemetry.collect()
        values["sdk_version"] = SDK_VERSION
        values["version"] = self.app_version
        self._client.send_telemetry(values)  # the SDK prints the packet itself when verbose
        self._last_sent = monotonic()

    # --- commands in --------------------------------------------------------------------------

    def _on_c2d(self, message) -> None:
        """MQTT thread. Look up the verb, hand the work to a command worker, return immediately.

        Nothing may escape from here: an exception on paho's callback thread stops MQTT processing
        for good, which would cost us the connection over one malformed command.
        """
        try:
            verb = C2D_VERBS.get(message.command_name)
            print(f"[iotc] c2d {message.command_raw!r} -> {verb or 'unknown command'}")
            if verb is None:
                self._send_ack(message, False, f"{message.command_name} is not implemented")
                return
            # is_exact: the dashboard's arguments are typed against names the back end already
            # holds, so they are matched literally. Guessing is voice's problem, not this channel's.
            command = Command(verb, " ".join(message.command_args), message.command_raw,
                              is_exact=True)
            received_at = monotonic()
            future = self.service.submit_command(command, source="c2d")
            future.add_done_callback(
                lambda done: self._on_command_done(message, verb, done, received_at))
        except Exception as error:
            logger.exception("could not accept c2d command")
            print(f"[iotc] could not accept {message.command_raw!r}: {error}")

    def _on_command_done(self, message, verb: str, future: Future,
                         received_at: float = 0.0) -> None:
        """Command-worker thread: finish the cloud-only part of the command, then acknowledge.

        The elapsed time logged here is ours alone - from the SDK handing us the message to the ack
        going out. Anything the dashboard shows on top of it is delivery and the web UI's own
        refresh, which is worth knowing when a command "feels slow".
        """
        try:
            result = future.result()
            is_ok, text = result.is_ok, result.message
            if is_ok and verb == commands.DESCRIBE_SCENE:
                text = "Scene described."  # the description itself went out as the scene attribute
            elif is_ok and verb in (commands.AGENT, commands.DESCRIBE_ALERT, commands.CLEAR_ALERT):
                # The beginning of the answer, not a stand-in for it: a tooltip saying "Answered."
                # leaves the dashboard unable to tell one reply from another. The whole thing went
                # out as the `answer` attribute a moment ago - which is also why `alert-describe`
                # is here: an event log in prose is several sentences, and a tooltip is one.
                text = shorten(text, MAX_AGENT_ACK_CHARS)
        except Exception as error:
            logger.exception("c2d %s failed", verb)
            is_ok, text = False, f"Failed: {type(error).__name__}"
        self._send_ack(message, is_ok, text)
        print(f"[iotc] ack {message.command_name} {'ok' if is_ok else 'failed'} "
              f"in {(monotonic() - received_at) * 1000:.0f} ms on the device: {text}")

    def upload_capture(self) -> str:
        """Put capture.jpg in the device's S3 bucket, tagged with what we think is in it.

        The snapshot handler calls this - `main.py` hands it over as a plain callable, so `app.py`
        never learns what S3 is. It follows a handler's contract rather than this file's: the
        sentence to say when it works, `CommandError` when it does not. That is what makes the
        failure readable whether it came from the dashboard, from voice or from the LLM's tool.
        """
        if self._client is None or self._s3 is None:
            raise CommandError("Snapshot saved, but file upload is not enabled for this device.")
        if not self.capture_path.exists():
            raise CommandError("Snapshot saved, but the file is missing.")
        self._client.s3_upload(
            local_path=str(self.capture_path),
            custom_values={"cf": {"alarm": self.telemetry.get("alarm"),
                                  "objects": self.telemetry.get("objects")}},
        )
        print(f"[iotc] uploaded {self.capture_path} ({self.capture_path.stat().st_size} bytes)")
        return "Snapshot uploaded."

    def _send_ack(self, message, is_ok: bool, text: str) -> None:
        # No ack_id means the template marked the command as not needing one; no client means a
        # command outlived a reconnect, and there is nothing left to answer on.
        if message.ack_id is None or self._client is None:
            return
        status = C2dAck.CMD_SUCCESS_WITH_ACK if is_ok else C2dAck.CMD_FAILED
        self._client.send_command_ack(message, status, text[:MAX_ACK_CHARS])

    # --- the rest of the callbacks --------------------------------------------------------------

    def _on_ota(self, message) -> None:
        """We do not update ourselves yet - but say so out loud rather than leaving it hanging.

        The pieces are already here for when we do: `restart` brings the process back with the same
        arguments, and everything a visitor set is on disk.
        """
        url = message.urls[0] if message.urls else None
        print(f"[iotc] OTA offered: version {message.version} file "
              f"{url.file_name if url else '(none)'}")
        self._client.send_ota_ack(message, C2dAck.OTA_DOWNLOAD_FAILED, "OTA is not implemented")

    def _on_disconnect(self, reason: str, is_from_server: bool) -> None:
        print(f"[iotc] disconnected{' by the server' if is_from_server else ''}: {reason}")
        self.on_status("cloud: offline")


def shorten(text: str, limit: int) -> str:
    """`text` on one line, cut to `limit` characters with an ellipsis when it does not fit.

    The newlines matter as much as the length: a model's answer often arrives with them, and a
    dashboard tooltip renders the lot as one run-on line anyway.
    """
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


