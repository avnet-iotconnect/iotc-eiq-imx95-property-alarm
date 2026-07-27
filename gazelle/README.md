# gazelle — the anti-theft demo, driven by voice

Fifth pilot, and the first that is two earlier pilots at once. `elephant` proved the eyes (camera →
YOLO11n on Neutron → tracking → face recognition → alarm). `falcon` proved the ears and the mouth
(NXP's eIQ wake word, speech-to-text, text-to-speech and SmolVLM, used as a **library** rather than
as their demo pipeline). gazelle runs both in one process: the same 30 fps video loop, now taking
spoken commands.

`command.txt` is gone as the way in. Commands arrive by voice, and everything they touch — the
alarm, the user database, the locked objects — persists to disk.

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
| `take a snapshot` | writes `snapshot-<date>.jpg` — the frame with boxes and HUD | no camera frame yet |

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
| `scene.py` | `describe scene`: annotated frame → SmolVLM → a sentence | vlm |
| `state.py` | armed flag + locked objects, persisted across a restart | db |
| `registry.py` | face database (name ↔ embedding), presence queries | db |
| `face_worker.py` | face recognition off the frame loop, ~5 Hz | db/ml |
| `face.py` | YuNet detect + align + SFace embed on person crops | ml/face |
| `detector.py`, `yolo.py` | YOLO interpreter wrapper + the pre/post math | ml |
| `tracking.py` | `Tracker` + `Track` (identity, embedding, face box) | db |
| `camera.py`, `overlay.py` | GStreamer capture; Cairo HUD + the frame annotator the VLM needs | video |
| `mic-check.py` | standalone audio first-aid tool — no payload, no venv | — |
| `diag-face.py` | checks the board's `cv2` has the face classes | — |

## One command path, three ways in

This is the piece worth reading the code for. Voice is not wired to the app; it is wired to a
**command service** that anything can push a sentence into:

```
voice.py ──┐
           ├─► CommandService (4 threads) ─► app.on_command ─► "Registered Michael."
C2D (next) ┘                                     │                     │
                                                 └─ raises ────────────┴─► "Michael is already
                                                    CommandError            registered."
```

A handler either returns the sentence to say or raises `CommandError` with a speakable reason.
Both come back as a `CommandResult` carrying `is_ok` **and text** — which is exactly the shape an
/IOTCONNECT C2D acknowledgement needs, and the reason the error channel is a string rather than an
exception that escapes. Adding the cloud later is one more caller of `service.submit(text)`; nothing
in `app.py` or `voice.py` changes.

Four worker threads, not one: `describe scene` occupies a thread for tens of seconds inside the VLM,
and an `arm` arriving meanwhile should not queue behind it. Four is also the ceiling that leaves the
A55s enough headroom for the camera and YOLO to hold 30 fps.

## Threads, and why the video never stutters

| thread | does | cadence |
|---|---|---|
| main | camera → YOLO (Neutron) → tracking → overlay | 30 fps, blocks on nothing |
| face | YuNet + SFace on person crops | ~5 Hz, skips a tick if still busy |
| voice | loads eIQ, then wake word → STT → TTS | idle until spoken to |
| command ×4 | runs a command, speaks the result | on demand |

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
`state.json`. Startup is about 15 seconds.

## Install (on the board, in `~/gazelle`)

```bash
./install.sh          # once: eIQ payload, espeak-ng into /opt/dm-eiq, then a venv
./run.sh --prefetch   # once, on a good network: pulls SmolVLM from Hugging Face
./run.sh              # the demo
```

The vision models are **not** fetched by `install.sh`: they are converted on the host (the
neutron-converter does not run on the board) and pushed by `deploy.sh` along with the code.

```bash
bash gazelle/deploy.sh                      # code + vision models
bash gazelle/deploy.sh dm-eiq-<tag>.tgz     # ... and the 139 MB eIQ payload, when it changes
```

Useful ways to run it:

```bash
./run.sh --no-voice                     # video only: no eIQ payload, no pip packages needed
./run.sh --model yolo11n_int8.tflite    # full-CPU baseline
./run.sh --preload-vlm                  # pay SmolVLM's 12 s at startup, not at the first question
./run.sh --no-vlm                       # drop 'describe scene' entirely
./run.sh --text-commands command.txt    # debug: type commands instead of saying them
./mic-check.py --loop                   # when the microphone misbehaves, start here
```

`--text-commands` writes into the *same* command service voice uses, so a handler tested that way is
tested for real:

```bash
echo 'register user Michael' > command.txt
```

## Layout — ours vs. theirs

Everything at the top level of `gazelle/` is our code, tracked and editable. Everything inside
`nxp-lib/` is NXP's, copied verbatim, **never edited**, and never committed — the licence forbids
modifying it and it is 139 MB of mostly encrypted weights. `main.py` puts exactly one directory on
`sys.path` (`nxp-lib/src`), which replaces the nine roots their `pip install -e .` would have set up.

```
gazelle/
├── *.py  *.json  *.txt  *.sh                    ours, tracked
├── *.tflite  *.onnx     vision models, pushed by deploy.sh    gitignored
├── faces.json  state.json  scene.jpg            runtime state  gitignored
├── venv/  models/                               gitignored
└── nxp-lib/            NXP payload — theirs, never edited     gitignored
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
echo snapshot > command.txt               # -> snapshot-20260727-141005.jpg
scp root@<board>:gazelle/snapshot-*.jpg . # collect them
```

or just say **"take a snapshot"**. Same boxes, names and HUD as the HDMI screen. `--snapshot-dir`
puts them somewhere else.

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
- **SmolVLM downloads on first use.** `--prefetch` pulls it ahead of time; nothing else here
  reaches the network at run time.
