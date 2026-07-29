#!/usr/bin/env python3
"""Standalone /IOTCONNECT first-aid: can this device talk to the cloud at all?

The equivalent of `mic-check.py` for the network half. No camera, no models, no eIQ payload - just
the SDK, the device config and the certificate pair sitting in this directory. Run it once after
deploying to a new board, and again whenever the demo's HUD says `cloud: offline` and you want to
know whether the problem is the board, the account or the device template.

    ./iotc-check.py                 # connect, report S3/KVS, send one telemetry message
    ./iotc-check.py --listen 60     # ... then print C2D commands for a minute (acks them as failed)
    ./iotc-check.py --upload        # ... and upload capture.jpg, proving the S3 path end to end
    ./iotc-check.py --webrtc        # ... and open the signalling channel, proving the WebRTC path

It deliberately does *not* import anything else from the pilot: if this works and `./run.sh` does
not, the difference is the demo, not the connection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import sleep

from iotc import resolve_credentials

from avnet.iotconnect.sdk.lite import Callbacks, Client, ClientSettings, DeviceConfig
from avnet.iotconnect.sdk.lite import __version__ as SDK_VERSION
from avnet.iotconnect.sdk.sdklib.mqtt import C2dAck


def report_provider(name: str, provider) -> None:
    """One line per AWS credential provider, and the endpoint the next pilot will need."""
    if provider is None:
        print(f"{name:4s}: not enabled for this device (check the device template)")
        return
    provider.obtain_credentials()
    credentials = provider.get_credentials()
    print(f"{name:4s}: credentials obtained, expire in {provider.get_secs_to_expiry() / 60:.0f} min "
          f"(key {credentials.access_key_id[:8]}...)")
    print(f"      endpoint {provider.credentials_endpoint}")


def check_webrtc(kvs_client) -> None:
    """Prove the WebRTC half as far as it can be proved without a camera or a viewer.

    Three things go wrong here and each fails differently: aiortc missing (an install problem), the
    template having no WebRTC channel (a web-UI problem), and the signed URL being rejected (a
    clock, region or permissions problem). Doing this on its own means the demo's `stream:` line
    never has to be the first place you find out.
    """
    import webrtc

    if not webrtc.IS_WEBRTC_AVAILABLE:
        print("WebRTC: aiortc is not installed -- pip install -r requirements.txt")
        return
    if kvs_client is None or not kvs_client.get_signaling_channel_arn():
        print("WebRTC: no signalling channel -- enable Video Streaming -> WebRTC in the template")
        return

    import asyncio  # imported late: only reachable once the packages above are known to be there

    import websockets

    streamer = webrtc.WebRtcStreamer()
    streamer.configure(kvs_client.get_signaling_channel_arn(), kvs_client.get_credentials)
    print(f"WebRTC: region {streamer.region}, signing a MASTER url ...")
    url = streamer.build_signed_url()
    print(f"        endpoint {streamer.endpoints['WSS']}")
    print(f"        ice servers: {len(streamer.fetch_ice_servers())} (stun + kvs turn relays)")

    async def open_once() -> None:
        async with websockets.connect(url):
            print("        signalling channel OPEN as master -- a viewer could connect now")

    asyncio.run(open_once())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="iotcDeviceConfig.json", help="the device config JSON")
    parser.add_argument("--cert", default="", help="device certificate (default: <duid>-crt.pem)")
    parser.add_argument("--key", default="", help="device private key (default: <duid>-key.pem)")
    parser.add_argument("--listen", type=float, default=0.0, help="seconds to wait for C2D commands")
    parser.add_argument("--upload", action="store_true", help="also upload capture.jpg to S3")
    parser.add_argument("--webrtc", action="store_true",
                        help="also sign and open the KVS signalling channel (no camera needed)")
    parser.add_argument("--capture", default="capture.jpg", help="the file --upload sends")
    parser.add_argument("--quiet", action="store_true", help="do not print every MQTT packet")
    args = parser.parse_args()

    config_path = Path(args.config)
    cert_path, key_path = resolve_credentials(config_path, Path(args.cert) if args.cert else None,
                                              Path(args.key) if args.key else None)
    print(f"SDK {SDK_VERSION}")
    for what, path in (("config", config_path), ("cert", cert_path), ("key", key_path)):
        print(f"{what:6s}: {path} {'' if path.exists() else '  <-- MISSING'}")
    if not (config_path.exists() and cert_path.exists() and key_path.exists()):
        print("\nDownload the config and certificate pair from the device's info panel in "
              "/IOTCONNECT and put them here.")
        return 1

    def on_command(message) -> None:
        print(f"\nC2D  : {message.command_raw!r} (args {message.command_args})")
        if message.ack_id is not None:
            client.send_command_ack(message, C2dAck.CMD_FAILED, "iotc-check is running, not the demo")

    config = DeviceConfig.from_iotc_device_config_json_file(
        device_config_json_path=str(config_path), device_cert_path=str(cert_path),
        device_pkey_path=str(key_path))
    client = Client(config=config, callbacks=Callbacks(command_cb=on_command),
                    settings=ClientSettings(verbose=not args.quiet))
    client.connect()
    if not client.is_connected():
        print("Could not connect. Check the network, the clock and that the device is not disabled.")
        return 2
    print(f"\nconnected as {client.get_duid()} (thing {client.get_client_id()})")

    s3_client = client.get_s3_client()
    report_provider("S3", s3_client)
    if s3_client is not None:
        buckets = ", ".join(bucket.bucket_name for bucket in s3_client.get_buckets()) or "(none)"
        print(f"      buckets {buckets}")
    kvs_client = client.get_kvs_client()
    report_provider("KVS", kvs_client)
    if kvs_client is not None:
        print(f"      channel {kvs_client.get_signaling_channel_arn()} "
              f"auto-start {kvs_client.is_auto_start()}")
    if args.webrtc:
        check_webrtc(kvs_client)

    client.send_telemetry({"sdk_version": SDK_VERSION, "version": "iotc-check", "alarm": "disarmed",
                           "objects": "none", "fps": 0})
    print("\nsent one telemetry message -- it should appear in the device's Live Data within seconds")

    if args.upload:
        capture_path = Path(args.capture)
        if not capture_path.exists():
            print(f"{capture_path} does not exist -- run the demo and take a snapshot first")
        else:
            client.s3_upload(local_path=str(capture_path), custom_values={"cf": "iotc-check"})
            print(f"uploaded {capture_path} -- look under Telemetry Files")

    if args.listen > 0:
        print(f"\nlistening for C2D commands for {args.listen:.0f}s -- send one from the dashboard")
        sleep(args.listen)

    client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
