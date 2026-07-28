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
          version, fps, alarm, objects, and `scene` when someone asked the VLM a question
    C2D   the nine commands in the alrmtheft device template, mapped to our verbs by `C2D_VERBS`;
          every one is acknowledged with the same sentence the demo would have spoken
    S3    `snapshot` writes capture.jpg and uploads it; /IOTCONNECT timestamps each version

Two conveniences worth knowing about. `snapshot` and `scene` get their ack **rewritten** here, and
only here: the cloud wants "Snapshot uploaded" (which is a different fact from "snapshot saved") and
a short "Scene described" rather than a paragraph - the paragraph goes to the `scene` attribute
instead, once. And KVS credentials are fetched even though nothing streams video yet: the next pilot
needs the signalling channel, and finding out at the booth that streaming was never enabled on the
template is the failure worth pre-empting.

Nothing here is required to run the demo. No config, no certificates, no SDK installed, or no
network - the pilot says so on the HUD and carries on being an anti-theft demo.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import Future
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import Callable

import commands
from commands import Command, CommandService
from telemetry import TelemetryState

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
    "scene": commands.DESCRIBE_SCENE,
    "snapshot": commands.SNAPSHOT,
    "restart": commands.RESTART,
}

TELEMETRY_INTERVAL_S = 4.0
MIN_SEND_GAP_S = 1.0        # a flapping alarm state must not turn into an MQTT flood
RECONNECT_WAIT_S = 30.0     # after a failure that took the client down entirely
CREDENTIALS_MARGIN_S = 120  # refresh AWS credentials this long before they expire
MAX_ACK_CHARS = 200         # acks are a status line, not a transcript


class IotcClient:
    """The /IOTCONNECT connection: one thread that publishes, and callbacks that hand work away."""

    def __init__(
        self, config_path: Path, service: CommandService, telemetry: TelemetryState,
        capture_path: Path, app_version: str, cert_path: Path | None = None,
        key_path: Path | None = None, on_status: Callable[[str], None] | None = None,
        interval_s: float = TELEMETRY_INTERVAL_S, is_verbose: bool = False,
    ) -> None:
        self.config_path = config_path
        self.service = service
        self.telemetry = telemetry
        self.capture_path = capture_path
        self.app_version = app_version
        self.cert_path, self.key_path = resolve_credentials(config_path, cert_path, key_path)
        self.on_status = on_status or (lambda status: None)
        self.interval_s = interval_s
        self.is_verbose = is_verbose

        self._client: Client | None = None
        self._s3 = None
        self._kvs = None
        self._last_sent = 0.0
        self._stop = Event()
        self._thread = Thread(target=self._run, name="iotc", daemon=True)

    # --- lifecycle ----------------------------------------------------------------------------

    def start(self) -> None:
        """Connect and publish on our own thread: the identity REST call alone takes a second."""
        self.on_status("cloud: connecting")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.telemetry.wake()  # so the publisher is not still sitting in its 4-second wait
        self._thread.join(timeout=10.0)
        if self._client is not None:
            self._client.disconnect()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._connect()
                self._publish_loop()
            except Exception as error:  # the cloud going away must never take the demo with it
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
            callbacks=Callbacks(command_cb=self._on_c2d, ota_cb=self._on_ota,
                                disconnected_cb=self._on_disconnect, vs_cb=self._on_stream_event),
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
        """Fetch S3 and KVS credentials once at connect, and say what the account actually allows.

        KVS is not used yet. It is fetched anyway because "Streaming is not enabled on the template"
        is a five-minute fix on the day the next pilot needs it and a lost afternoon at a booth.
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
        else:
            self._kvs.obtain_credentials()
            print(f"[iotc] KVS channel {self._kvs.get_signaling_channel_arn()} "
                  f"(auto-start {self._kvs.is_auto_start()}, streaming {self._kvs.is_streaming()}), "
                  f"credentials good for {self._kvs.get_secs_to_expiry() / 60:.0f} min")

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
            if is_ok and verb == commands.SNAPSHOT:
                is_ok, text = self._upload_capture()
            elif is_ok and verb == commands.DESCRIBE_SCENE:
                text = "Scene described."  # the description itself went out as the scene attribute
        except Exception as error:
            logger.exception("c2d %s failed", verb)
            is_ok, text = False, f"Failed: {type(error).__name__}"
        self._send_ack(message, is_ok, text)
        print(f"[iotc] ack {message.command_name} {'ok' if is_ok else 'failed'} "
              f"in {(monotonic() - received_at) * 1000:.0f} ms on the device: {text}")

    def _upload_capture(self) -> tuple[bool, str]:
        """Put capture.jpg in the device's S3 bucket, tagged with what we think is in it."""
        if self._s3 is None:
            return False, "Snapshot saved, but file upload is not enabled for this device"
        if not self.capture_path.exists():
            return False, "Snapshot file is missing"
        self._client.s3_upload(
            local_path=str(self.capture_path),
            custom_values={"cf": {"alarm": self.telemetry.get("alarm"),
                                  "objects": self.telemetry.get("objects")}},
        )
        print(f"[iotc] uploaded {self.capture_path} ({self.capture_path.stat().st_size} bytes)")
        return True, "Snapshot uploaded."

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

    def _on_stream_event(self, kvs) -> None:
        """Video streaming start/stop from the cloud. Nothing streams yet; keep the credentials
        fresh and record the request so the next pilot has something to act on."""
        print(f"[iotc] video streaming requested: streaming={kvs.is_streaming()}")
        try:
            self._refresh_credentials()
        except Exception as error:  # this runs on the MQTT thread; it must not raise into paho
            logger.exception("KVS credential refresh failed")
            print(f"[iotc] KVS refresh failed: {error}")


def resolve_credentials(config_path: Path, cert_path: Path | None,
                        key_path: Path | None) -> tuple[Path, Path]:
    """The certificate pair beside the device config, unless the caller says otherwise.

    /IOTCONNECT names a downloaded pair after the device (`nik-imx95-01-crt.pem`), so that is the
    convention followed here; the duid comes out of the config file, which is always present.
    """
    duid = json.loads(config_path.read_text()).get("uid", "") if config_path.exists() else ""
    return (cert_path or config_path.parent / f"{duid}-crt.pem",
            key_path or config_path.parent / f"{duid}-key.pem")


def is_configured(config_path: Path) -> bool:
    """True when there is something to connect with. Checked before the client is even built."""
    if not IS_SDK_AVAILABLE or not config_path.exists():
        return False
    cert_path, key_path = resolve_credentials(config_path, None, None)
    return cert_path.exists() and key_path.exists()
