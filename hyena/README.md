# hyena — the anti-theft demo, driven by voice and by /IOTCONNECT

Sixth pilot. `gazelle` put two earlier pilots in one process — `elephant`'s eyes (camera → YOLO11n
on Neutron → tracking → face recognition → alarm) and `falcon`'s ears and mouth (NXP's eIQ wake
word, speech-to-text, text-to-speech and SmolVLM, used as a **library**) — and replaced
`command.txt` with spoken commands. hyena connects that to the cloud.

Avnet's [/IOTCONNECT Python Lite SDK](https://github.com/avnet-iotconnect/iotc-python-lite-sdk) adds
a second way to drive the same demo and a way to watch it:

- **telemetry every 4 seconds** — alarm state, what is on screen, frames per second, versions — and
  immediately, not in four seconds, when the alarm state changes;
- **the same commands as C2D messages**, each acknowledged with the sentence the demo would have
  spoken back;
- **`capture.jpg` uploaded to S3** when the cloud asks for a snapshot;
- **`restart`**, because eIQ stops the models an hour after they load and a stand cannot depend on
  someone noticing.

## Say it out loud

Say **"Hey NXP"**, wait for the blip, then you have **3 seconds to start talking**:

| command | what happens | error you will hear |
|---|---|---|
| `register user Michael` | binds Michael to the largest face on screen | already registered / no face visible |
| `unregister user Michael` | forgets him, and the label disappears at once | I do not know anyone called Michael |
| `unregister last user` | undo for a registration that bound the wrong face | there are no registered users |
| `arm` / `arm the alarm` | arms the alarm | (never fails) |
| `disarm` / `disarm the alarm` | disarms it; the HUD says DISARMED in green | (never fails) |
| `lock the laptop` | **stub**: finds a laptop on screen and marks it guarded | I cannot see a laptop |
| `unlock the laptop` | releases it | the laptop is not locked |
| `describe the scene` | sends the frame, boxes and all, to SmolVLM and reads out the answer | no camera frame yet |
| `take a snapshot` | writes `capture.jpg` — the frame with boxes and HUD | no camera frame yet |

Every one of those answers is spoken. So is the transcript, before the answer — you hear what the
recogniser thought you said, which is the difference between debugging in five seconds and in five
minutes. `--no-repeat` turns that off once the booth is running.

`lock` / `unlock` are deliberately stubs: they prove an object can be *named and found* on screen,
which is the hard part of the voice slice. The 2s/5s theft timing that makes a locked object
actually trip the alarm is the next pilot.

## The pieces (and which future module each seeds)

| File | Role | Lineage |
|---|---|---|
| `main.py` | composition root + the 30 fps frame loop + the argument surface | main |
| `app.py` | `AntiTheftApp`: alarm state machine + every command handler | app |
| `commands.py` | sentence → `Command`, and the 4-thread pool that runs it | app |
| `voice.py` | wake word → blip → VAD capture → transcript → spoken reply | audiotext |
| `vocabulary.py` + `.json` | saying names right, and recognising them when STT mangles them | audiotext |
| `audio_devices.py` + `audio.json` | device naming and the mixer gain NXP's code does not set | audiotext |
| `iotc.py` | /IOTCONNECT: telemetry out, C2D in, snapshot to S3, KVS credentials | iotconnect |
| `telemetry.py` | what the cloud should know, written by anyone, read by the publisher | iotconnect |
| `scene.py` | `describe scene`: annotated frame → SmolVLM → a sentence | vlm |
| `state.py` | armed flag + locked objects, persisted across a restart | db |
| `registry.py` | face database (name ↔ embedding), presence queries | db |
| `face_worker.py` | face recognition off the frame loop, ~5 Hz | db/ml |
| `face.py` | YuNet detect + align + SFace embed on person crops | ml/face |
| `detector.py`, `yolo.py` | YOLO interpreter wrapper + the pre/post math | ml |
| `tracking.py` | `Tracker` + `Track` (identity, embedding, face box) | db |
| `camera.py`, `overlay.py` | GStreamer capture; Cairo HUD + the frame annotator the VLM needs | video |
| `mic-check.py` | standalone audio first-aid tool — no payload, no venv | — |
| `iotc-check.py` | standalone /IOTCONNECT first-aid: connect, S3, KVS, one message | — |
| `diag-face.py` | checks the board's `cv2` has the face classes | — |

## One command path, three ways in

This is the piece worth reading the code for. Voice is not wired to the app; it is wired to a
**command service** that anything can push a command into:

```
voice.py ──┐  a sentence, parsed
           ├─► CommandService (4 threads) ─► app.on_command ─► "Registered Michael."
iotc.py  ──┘  a verb + arguments             │                     │
              already structured             └─ raises ────────────┴─► "Michael is already
                                                 CommandError            registered."
```

A handler either returns the sentence to say or raises `CommandError` with a speakable reason.
Both come back as a `CommandResult` carrying `is_ok` **and text** — which is exactly the shape an
/IOTCONNECT C2D acknowledgement needs, and the reason the error channel is a string rather than an
exception that escapes.

gazelle predicted the cloud would be "one more caller of `service.submit(text)`". It is one more
caller of `submit_command(...)` instead: /IOTCONNECT sends `user-register Nick`, so the verb is
already known and running it through a parser built for mangled speech would only lose information.
The two entry points meet in `execute()`, one line later. Nothing in `app.py` or `voice.py` changed
to make room for the cloud, and `app.py` still does not import `iotc.py`.

Four worker threads, not one: `describe scene` occupies a thread for tens of seconds inside the VLM,
and an `arm` arriving meanwhile should not queue behind it. Four is also the ceiling that leaves the
A55s enough headroom for the camera and YOLO to hold 30 fps.

## /IOTCONNECT

### Set the device up

The demo needs three files in this directory, all of them yours and none of them committed:

```
iotcDeviceConfig.json      the cog icon in your device's info panel
<duid>-crt.pem             the certificate pair created with the device
<duid>-key.pem
```

`deploy.sh` pushes them along with the code if they are here. Point somewhere else with
`--iotc-config`, `--iotc-cert`, `--iotc-key`. Create the device from the **alrmtheft** template
(`files/alrmtheft.json` in this repo) — it declares the six telemetry attributes and the nine
commands below, and the names have to match. Two template settings are not in that file and have to
be ticked in the web UI:

| setting | what breaks without it |
|---|---|
| **File Support** | `snapshot` saves the JPEG but cannot upload it: "file upload is not enabled" |
| **Streaming** | no KVS credentials — harmless today, needed by the pilot after this one |

`./iotc-check.py` tells you where you stand in about five seconds, without starting the demo.

### What goes out

Every 4 seconds, and immediately whenever the alarm state changes:

| attribute | example | from |
|---|---|---|
| `alarm` | `disarmed` / `armed` / `alarm` | `app._set_state`, the single announce point |
| `objects` | `Nick, person, laptop` | the tracker, names filled in by face recognition |
| `fps` | `27.4` | the frame loop, twice a second |
| `scene` | the VLM's answer | **once**, after a `scene` command — not repeated afterwards |
| `version`, `sdk_version` | `hyena-1.0`, `1.3.0` | constants |

### What comes in

| command | argument | acknowledged with |
|---|---|---|
| `user-register` | a name | `Registered Nick.` / why not |
| `user-unregister` | a name | `Unregistered Nick.` / who is actually known |
| `object-lock`, `object-unlock` | a YOLO class name | `The laptop is locked.` / what is visible |
| `alarm-arm`, `alarm-disarm` | — | `Alarm armed.` |
| `scene` | optional question | `Scene described.` — the text itself goes to the `scene` attribute |
| `snapshot` | — | `Snapshot uploaded.` (S3) or the reason it was not |
| `restart` | — | `Restarting.`, then the process comes back three seconds later |

`scene` with no argument asks the model to describe what it sees. With one, the argument *is* the
question: `scene what is the person wearing`.

**Arguments from the cloud are matched exactly.** `user-unregister Michael` acts on a user called Michael
or is refused with "I do not know anyone called Michael. I know Marija" — it never guesses. Voice is
the opposite and has to be: the recogniser emits "money" for Marija, so spoken arguments go through
the phonetic matcher in `vocabulary.py`. The source states which it is (`Command.is_exact`), and the
difference is worth measuring: 31 µs for a typed name against 74 ms of espeak calls for a spoken one,
per command, growing with the number of registered users.

### How it is wired, in one paragraph

Producers never touch the network. `app.py` writes into `telemetry.TelemetryState` — a dict behind a
lock — and one publisher thread reads it, so the 30 fps loop never waits on MQTT and `app.py` has no
idea the cloud exists. The SDK's C2D callback runs on paho's MQTT thread, so `_on_c2d` only looks up
the verb and hands it to the command pool; the acknowledgement is sent when the future completes.
`telemetry.wake()` cuts the 4-second wait short for anything worth seeing now, with a one-second
floor so a flapping alarm cannot turn into an MQTT flood.

Nothing here is load-bearing for the demo. No config, no certificates, no SDK, no network: the HUD
says `cloud: unconfigured` or `cloud: offline` and the anti-theft demo runs exactly as before.

### Restarting

eIQ's models stop working 60 minutes after they load. `restart` (or `--restart-after 55`) sets a
flag, the frame loop finishes its frame, everything shuts down in order, and the process `exec`s
itself with the same arguments — one pid, no orphan holding `/dev/video4`. Registered faces, the
armed flag and the locked objects are all on disk, so the demo comes back as it was in ~15 s. The
three-second delay before it happens is there so the C2D acknowledgement reaches the cloud first.

## Threads, and why the video never stutters

| thread | does | cadence |
|---|---|---|
| main | camera → YOLO (Neutron) → tracking → overlay | 30 fps, blocks on nothing |
| face | YuNet + SFace on person crops | ~5 Hz, skips a tick if still busy |
| voice | loads eIQ, then wake word → STT → TTS | idle until spoken to |
| command ×4 | runs a command, speaks the result | on demand |
| iotc | connects, then publishes telemetry | every 4 s, or when woken |
| paho (SDK) | MQTT socket, delivers C2D | idle |

The voice thread **loads its own models**, ~15 s of `import torch` and decrypting weights. That
happens after the camera is already running, so the picture is live in about a second and the HUD's
`voice:` line tells you when the ears are ready. Nothing in eIQ touches Neutron, so the YOLO and
SFace offload is what buys the voice stack its CPU.

## Alarm

Starts **disarmed** (and stays however you left it — see below). Armed + a person on screen + nobody
recognised ⇒ **ALARM**, big red banner. A recognised user drops it back to armed, yellow.

A sighting counts for a moment after it happens — a person for 1 s, a recognised user for 2 s. The
tracker reports only what it saw *this* frame, and YOLO drops a person for the odd frame, so without
that the banner flickers several times a second. The real 2s/5s guarded-object timing is the next
pilot; this is only what the banner needs.

## What survives a restart

eIQ kills the process after **60 minutes** — the timeout is inside NXP's compiled modules and rides
along with the models, so at a trade show the demo *will* restart between visitors. Anything set by
voice therefore goes to disk: registered faces in `faces.json`, the armed flag and locked objects in
`state.json`. Startup is about 15 seconds. hyena can also do the restarting itself: see above.

## Install (on the board, in `~/hyena`)

```bash
./install.sh          # once: eIQ payload, espeak-ng into /opt/dm-eiq, then a venv
./run.sh --prefetch   # once, on a good network: pulls SmolVLM from Hugging Face
./iotc-check.py       # once: does this device reach /IOTCONNECT, S3 and KVS?
./run.sh              # the demo
```

If a previous pilot's venv is already on the board, copying it (`cp -a ~/gazelle/{nxp-lib,venv,models}
~/hyena/`) beats a torch re-download — but then install the SDK with **`./venv/bin/python -m pip`**,
not `./venv/bin/pip`: a copied venv's console scripts keep the absolute shebang they were created
with, and `pip` will happily install into the venv it came from while reporting success.

The vision models are **not** fetched by `install.sh`: they are converted on the host (the
neutron-converter does not run on the board) and pushed by `deploy.sh` along with the code.

```bash
bash hyena/deploy.sh                      # code + vision models + your /IOTCONNECT credentials
bash hyena/deploy.sh dm-eiq-<tag>.tgz     # ... and the 139 MB eIQ payload, when it changes
```

Useful ways to run it:

```bash
./run.sh --no-voice                     # video only: no eIQ payload, no pip packages needed
./run.sh --no-iotc                      # no cloud, same as having no credentials here
./run.sh --iotc-verbose                 # log every MQTT packet, both directions
./run.sh --restart-after 55             # beat eIQ's 60-minute timeout instead of waiting for it
./run.sh --model yolo11n_int8.tflite    # full-CPU baseline
./run.sh --preload-vlm                  # pay SmolVLM's 12 s at startup, not at the first question
./run.sh --no-vlm                       # drop 'describe scene' entirely
./run.sh --text-commands command.txt    # debug: type commands instead of saying them
./mic-check.py --loop                   # when the microphone misbehaves, start here
./iotc-check.py --listen 60             # ... and when the cloud does
```

`--text-commands` writes into the *same* command service voice uses, so a handler tested that way is
tested for real:

```bash
echo 'register user Michael' > command.txt
```

## Layout — ours vs. theirs

Everything at the top level of `hyena/` is our code, tracked and editable. Everything inside
`nxp-lib/` is NXP's, copied verbatim, **never edited**, and never committed — the licence forbids
modifying it and it is 139 MB of mostly encrypted weights. `main.py` puts exactly one directory on
`sys.path` (`nxp-lib/src`), which replaces the nine roots their `pip install -e .` would have set up.

```
hyena/
├── *.py  *.json  *.txt  *.sh                    ours, tracked
├── *.tflite  *.onnx     vision models, pushed by deploy.sh       gitignored
├── faces.json  state.json  scene.jpg  capture.jpg  runtime state gitignored
├── iotcDeviceConfig.json  *-crt.pem  *-key.pem  your device      gitignored
├── venv/  models/                                                gitignored
└── nxp-lib/            NXP payload — theirs, never edited        gitignored
```

## Vocabulary: `vocabulary.json`

Two different problems, one file.

**Saying a word.** TTS phonetises through espeak-ng, which reads by English spelling rules: "Marija"
comes out "ma-RID-zha". `say_as` hands TTS a respelling instead — `Mahria` → `mˈɑːɹiə`. Derive a new
one by ear: `espeak-ng -q --ipa -v en "Mahria"`.

**Hearing it.** moonshine-base is compiled and encrypted: no hot-word list, no grammar to bias. It
emits "money" for Marija and that is that. So we match what it *did* emit against the short list of
words we care about, on **phonemes** rather than letters ("county" vs "Kranti" is 0.36 as strings,
close as sounds). A `heard_as` entry always wins outright — when the booth catches a new mangling,
write it down and the guessing stops.

Three groups: `names` (users), `objects` (**keyed by YOLO class name** — `cell phone`, not `phone`,
with "phone" as a `heard_as`), and `terms` (plain substitutions, mostly acronyms).

Registering a name that is *not* in `vocabulary.json` works — that is the live demo — but matching
is stricter there than anywhere else, because a loose match would silently register Michael as
Marija. Add each new person afterwards to make the *next* recognition reliable.

## Audio: `audio.json`

```json
{"capture_device": "micfil", "playback_device": "mqs", "micfil_range": 10,
 "capture_gain": 1.0, "playback_gain": 1.0}
```

Devices may be named by alias (`micfil`, `c920`, `mqs`), by any substring of a card name in
`arecord -l` (use this for a new USB microphone), or by a full ALSA string. Command-line flags win
over the file.

**The gain trap, stated plainly**, because it cost real time: NXP's audio manager raises mixer
levels on startup only for the **wm8960/wm8962** codecs on their other EVKs. This board has
**micfil** and **mqs**, nothing matches, and every level stays at its quiet default —
that is the whole explanation for a barely audible loopback. `audio_devices.py` sets them before
the manager is built. micfil has a 0–15 hardware `Range` (we ship 10, worth ~15 dB RMS); **mqs has
no mixer controls at all**, so playback loudness is digital only — `playback_gain`, or `--gain`.

## Taking a screenshot

**The compositor will not give you one.** Weston 14 only lets a client *it* launched capture the
output (`weston_compositor_add_screenshot_authority`), so `weston-screenshooter` from a shell
answers `Output capture error: unauthorized`. Its own `Super+S` binding does launch an authorised
client — but that needs a real keyboard on the board, and this image has no `uinput` module to fake
the keypress with. `/dev/fb0` exists but is an unused emulation buffer: all zeros.

So the picture is re-rendered from the frame we already have, which is also higher quality than a
screen grab (no compositor scaling):

```bash
./run.sh --text-commands command.txt      # start with the file input enabled
echo snapshot > command.txt               # -> capture.jpg, overwritten each time
scp root@<board>:hyena/capture.jpg .      # collect it
```

or just say **"take a snapshot"**, or send the `snapshot` command from the dashboard — which does
the same thing and then uploads the file, so /IOTCONNECT ends up holding the history that the board
deliberately does not. Same boxes, names and HUD as the HDMI screen. `--capture` puts the file
somewhere else.

If you do plug a USB keyboard in, `Super+S` works and drops PNGs of the real composited screen into
weston's home, `/home/weston`.

## Camera framerate: the exposure trap

If `capture=` in the frame log is above 33 ms, the camera — not the board — is the bottleneck. The
C920 in Aperture Priority mode stretches its exposure in a dim room (`exposure_time_absolute=666`
is 66 ms, so 15 fps) and UVC lets it drop the framerate to do it:

```bash
v4l2-ctl -d /dev/video4 -c exposure_dynamic_framerate=0    # hold 30 fps, accept a darker image
v4l2-ctl -d /dev/video4 -c exposure_dynamic_framerate=1    # back to the default
v4l2-ctl -d /dev/video4 --list-ctrls                       # what it is doing now
```

More light beats both. Face recognition wants the light anyway.

## Things that will bite

- **`describe scene` is slow and that is fine.** Tens of seconds on the CPU. It runs on a command
  thread; the video loop never waits for it. The first one also pays ~12 s to load SmolVLM unless
  you passed `--preload-vlm`.
- **The boxes go into the pixels.** The Cairo overlay on the HDMI preview cannot be read back, so
  `describe scene` burns the same boxes into a copy of the frame (saved as `scene.jpg`) before
  sending it. The question also states how many boxes there are — if the answer describes two people
  where we drew one box, that mismatch is the signal.
- **The microphone hears the speaker.** The wake reader is re-synced after every reply, or the demo
  wakes itself on its own voice.
- **Readers come back from `register_reader()` disabled.** Miss the `enable()` and you get a silent
  zero-frame loop, not an error.
- **SmolVLM downloads on first use.** `--prefetch` pulls it ahead of time; nothing else except
  /IOTCONNECT reaches the network at run time.
- **A `scene` command from the dashboard takes tens of seconds too.** The acknowledgement arrives
  when the model is done, not when the command was received. That is the VLM, not the connection.
- **File Support and Streaming are template settings, not code.** Without them `snapshot` cannot
  upload and there are no KVS credentials — `./iotc-check.py` says which, in one line each.
- **The clock matters.** TLS rejects a certificate when the board's date is wrong. If a freshly
  booted board will not connect, check `date` before suspecting anything else.
