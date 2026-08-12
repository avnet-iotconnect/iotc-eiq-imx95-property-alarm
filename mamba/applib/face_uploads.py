"""Faces that arrive from the cloud: the device's own S3 folder, polled, downloaded and registered.

Drop `Nick Markovic.jpg` into the device's `faces/` folder in /IOTCONNECT and the next poll
registers that person, without them standing in front of the camera. The file name is the name.

`iotc.py` hands over the bucket, the device's path in it and a callable returning current
credentials - the same handover the WebRTC streamer gets. This file knows S3 and nothing about
/IOTCONNECT; `iotc.py` knows /IOTCONNECT and nothing about faces. boto3 comes with the SDK's
`aws-s3` extra, so it is not a new dependency.

**Each version of each picture is dealt with exactly once, permanently.** Beside every downloaded
`Nick Markovic.jpg` sits a `Nick Markovic.status` holding the S3 ETag it was last handled at and a
word for how it went; a poll skips any file whose ETag still matches. The record is on disk rather than in
memory, so the hourly restart changes nothing - a picture with no face in it is not re-read and its
complaint is not republished, and unregistering somebody by voice is not undone twenty seconds
later by the folder they are still in.

Two ways to make it happen again, both of them things a person can do at a booth:

    re-upload the picture   a new ETag, so the next poll registers it
    delete faces/           everything is fetched and offered again from scratch

The ETag is the whole versioning scheme, so a listing that has none is **refused with a warning**
rather than guessed at - see `_poll`. Nothing here is fatal: a failed download writes no status and
is simply tried again on the next tick.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Event, Thread
from time import time
from typing import Callable

logger = logging.getLogger(__name__)

try:
    import boto3
    IS_BOTO3_AVAILABLE = True
except ImportError:
    IS_BOTO3_AVAILABLE = False

POLL_INTERVAL_S = 20.0
FACES_PATH = "faces/"  # the sub-path of the device's uploads directory that this file watches
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
STATUS_SUFFIX = ".status"  # the record written beside each picture: which ETag, and what happened


class FaceUploads:
    """Polls the device's `faces/` folder in S3 and hands each new picture to a callback."""

    def __init__(self, on_face_image: Callable[[str, Path], str], faces_dir: Path,
                 interval_s: float = POLL_INTERVAL_S) -> None:
        self.on_face_image = on_face_image
        self.faces_dir = faces_dir
        self.interval_s = interval_s
        self.bucket_name = ""
        self.role_arn: str | None = None
        self.prefix = ""
        self.get_credentials: Callable[[], object] = lambda: None
        self._client = None
        self._credentials = None
        self._warned: set[str] = set()  # keys already complained about, so a poll is not a flood
        self._stop = Event()
        self._thread = Thread(target=self._run, name="faces", daemon=True)

    def configure(self, bucket_name: str, role_arn: str | None, device_path: str,
                  get_credentials: Callable[[], object]) -> None:
        """Point this at a bucket without polling it. `role_arn` is set only for a bucket in
        another account, where the device's own credentials get no further than an assume-role."""
        self.bucket_name = bucket_name
        self.role_arn = role_arn
        self.prefix = device_path + FACES_PATH
        self.get_credentials = get_credentials

    def start(self, bucket_name: str, role_arn: str | None, device_path: str,
              get_credentials: Callable[[], object]) -> None:
        """Begin polling, once /IOTCONNECT has named the bucket and the device's path in it.

        Calling this again after a reconnect is a no-op: the bucket does not move.
        """
        if self._thread.is_alive():
            return
        if not IS_BOTO3_AVAILABLE:
            print("[faces] boto3 is not installed - uploaded faces will not be read")
            return
        self.configure(bucket_name, role_arn, device_path, get_credentials)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        print(f"[faces] watching s3://{bucket_name}/{self.prefix} every {self.interval_s:.0f}s "
              f"-> {self.faces_dir}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll()
            except Exception as error:
                logger.exception("could not read the uploaded faces")
                print(f"[faces] {type(error).__name__}: {error} - retrying in "
                      f"{self.interval_s:.0f}s")
                self._client = None  # credentials that went stale are rebuilt, not retried
            self._stop.wait(self.interval_s)

    def _poll(self) -> None:
        client = self._get_client()
        for key, etag, size in _list_images(client, self.bucket_name, self.prefix):
            if self._stop.is_set():
                return
            local_path = self.faces_dir / Path(key).name
            name = Path(key).stem.strip()
            if not etag:
                # The ETag is the only thing that says which upload this is. Without one, the file
                # would either be registered once every twenty seconds or never again, and neither
                # is a demo somebody can reason about - so it is left alone and said out loud.
                self._warn_once(key, "S3 REPORTED NO ETAG FOR IT, so there is no way to tell one "
                                     "upload of it from the next")
                continue
            if not name:
                self._warn_once(key, "there is no name in its file name")
                continue
            if _read_etag(local_path) == etag:
                continue
            try:
                self._fetch(client, key, local_path, size)
            except Exception as error:  # a network problem: no status, so it is tried again
                logger.exception("could not download %s", key)
                print(f"[faces] {key}: {type(error).__name__}: {error}")
                continue
            try:
                status = self.on_face_image(name, local_path)
            except Exception as error:  # a problem with the file itself: recorded, not retried
                logger.exception("could not register %s", key)
                status = "failed"
                print(f"[faces] {key}: {type(error).__name__}: {error}")
            _write_status(local_path, etag, status)

    def _fetch(self, client, key: str, local_path: Path, size: int) -> None:
        """Download the picture unless the copy on disk is already it."""
        if local_path.exists() and local_path.stat().st_size == size:
            return
        client.download_file(self.bucket_name, key, str(local_path))
        print(f"[faces] downloaded {key} ({size} bytes) -> {local_path}")

    def _warn_once(self, key: str, reason: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[faces] WARNING: {key} is being IGNORED because {reason}.")

    def _get_client(self):
        credentials = self.get_credentials()
        if credentials is None:
            raise ConnectionError("no S3 credentials available")
        if self._client is None or credentials is not self._credentials:
            self._client = self._create_client(credentials)
            self._credentials = credentials
        return self._client

    def _create_client(self, credentials):
        """A boto3 S3 client, assuming the cross-account role first when the bucket needs one -
        the reading half of what `Client.s3_upload` does to upload into somebody else's account."""
        if not self.role_arn:
            return boto3.client("s3", **_as_boto_credentials(credentials))
        assumed = boto3.client("sts", **_as_boto_credentials(credentials)).assume_role(
            RoleArn=self.role_arn,
            RoleSessionName=f"iotconnect-face-uploads-{int(time())}")["Credentials"]
        return boto3.client("s3",
                            aws_access_key_id=assumed["AccessKeyId"],
                            aws_secret_access_key=assumed["SecretAccessKey"],
                            aws_session_token=assumed["SessionToken"])


def _status_path(local_path: Path) -> Path:
    """`Nick Markovic.jpg` -> `Nick Markovic.status`, the record that sits beside it."""
    return local_path.with_suffix(STATUS_SUFFIX)


def _read_etag(local_path: Path) -> str:
    """The ETag this picture was last dealt with at, or '' when it never has been."""
    status_path = _status_path(local_path)
    if not status_path.exists():
        return ""
    try:
        return json.loads(status_path.read_text()).get("etag", "")
    except (OSError, ValueError):  # a truncated or hand-edited record means "do it again"
        return ""


def _write_status(local_path: Path, etag: str, status: str) -> None:
    """Record which upload this was and how it went. Deleting the file is the way to redo it.

    A word rather than a sentence - `registered`, `refused`, `failed`. What the demo has to know
    here is only whether this version of this file has been dealt with; the reason it was refused
    went to the console and to the dashboard when it happened.
    """
    _status_path(local_path).write_text(json.dumps({"etag": etag, "status": status}, indent=2))


def _list_images(client, bucket_name: str, prefix: str) -> list[tuple[str, str, int]]:
    """The images sitting directly in `prefix`, as (key, ETag, size). Sub-folders are ignored.

    One page, which is a thousand keys - far more than a folder somebody drops photographs into.
    """
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    images = []
    for entry in response.get("Contents", []):
        relative = entry["Key"][len(prefix):]
        if not relative or "/" in relative or not relative.lower().endswith(IMAGE_SUFFIXES):
            continue
        images.append((entry["Key"], entry.get("ETag", ""), entry["Size"]))
    return images


def _as_boto_credentials(credentials) -> dict[str, str]:
    """The SDK's credential object, in the shape boto3's client constructor wants."""
    return {"aws_access_key_id": credentials.access_key_id,
            "aws_secret_access_key": credentials.secret_access_key,
            "aws_session_token": credentials.session_token}
