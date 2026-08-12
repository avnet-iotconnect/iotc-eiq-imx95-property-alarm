# lion — the anti-theft demo, made fit for a booth

Ninth pilot: `koala`'s demo, with the rough edges taken off it and the source readied for release.
Nothing in `lion/` names a pilot any more — the code that ships should read as the product, not as
its own history — and the first thing to stop needing a manual is **the camera device**:

| | |
|---|---|
| `./run.sh` | finds the USB camera itself, whatever `/dev/videoN` this boot gave it |
| `./run.sh --device /dev/video52` | still yours to override, and it says which one it took |

See **[Picking the camera](#picking-the-camera)** below and `applib/camera_devices.py`.

Everything koala proved is unchanged. `iguana` had the whole demo bar one thing: everything a
visitor could ask for had to be a command somebody had written down — a spoken phrase the parser
knew, or a button on the dashboard.

**koala adds the `agent` command.** Type a sentence into the /IOTCONNECT dashboard and it is
answered by **Qwen2.5-7B-Instruct running on the Ara-240 DNPU**, whose tools are the demo's own
command handlers:

| you type | what happens |
|---|---|
| `agent please disarm the alarm` | the model calls the `disarm_alarm` tool — the alarm really disarms |
| `agent register another user Michael` | `register_user("Michael")`, against the face on screen |
| `agent is the alarm on, and who can you see?` | `get_status`, then an answer in plain English |
| `agent what time is it?` | `get_time` — the model has no clock of its own |
| `agent take a screenshot` | `take_screenshot` — saved and uploaded, same as the button |

Three things are worth saying up front, because they are the design:

- **The model is in another process, on another chip.** `connector/` is NXP's eIQ AAF Connector,
  installed on the board with its own venv, keeping the 7B resident on the Ara-240 behind an
  OpenAI-shaped REST endpoint. koala needs `strands-agents` and `openai` and no ML packages at all.
- **The model does not decide anything.** It picks tools; `app/watchdog.py` still decides when the
  alarm goes off, in deterministic code. A model that hallucinates a theft is a worse demo than no
  model.
- **The tools *are* the command handlers.** `disarm_alarm` runs the same code the dashboard button
  and the spoken phrase run, gets the same refusals, and cannot do anything they could not.

Everything iguana proved is still here: YOLO11n + face recognition on Neutron at ~29 fps, voice,
SmolVLM scene description, /IOTCONNECT telemetry and C2D, and the HDMI display streamed to the
dashboard over KVS WebRTC.

## Say it out loud

Say **"Hey NXP"**, wait for the blip, then you have **3 seconds to start talking**:

| command | what happens | error you will hear |
|---|---|---|
| `register user Michael` | counts you down, then binds Michael to the closest face | already registered / no good look at a face |
| `unregister user Michael` | forgets him, and the label disappears at once | I do not know anyone called Michael |
| `unregister last user` | undo for a registration that bound the wrong face | there are no registered users |
| `arm` / `arm the alarm` | arms the alarm | (never fails) |
| `disarm` / `disarm the alarm` | disarms it, and clears everything the watchdog holds | (never fails) |
| `what happened` / `describe the alert` | reads the event log out loud | (never fails) |
| `clear the alert` | empties the log — the acknowledgement, without disarming | (never fails) |
| `lock the laptop` | guards it, anchored to where it is standing right now | I cannot see a laptop |
| `unlock the laptop` | releases it | the laptop is not locked |
| `describe the scene` | sends the frame, boxes and all, to SmolVLM and reads out the answer | no camera frame yet |
| `take a snapshot` | writes `capture.jpg` — the frame with boxes and HUD | no camera frame yet |

Every one of those answers is spoken. So is the transcript, before the answer — you hear what the
recogniser thought you said, which is the difference between debugging in five seconds and in five
minutes. `--no-repeat` turns that off once the booth is running.

Locking is its own switch: a locked object is watched whether or not the alarm is armed, because
locking a laptop is already an explicit act and should not also require remembering to arm. Saying
`lock the laptop` again is not a mistake — it is how you say "it lives *here* now" after something
has been moved. See **Alarm** below.

## Registering a face

`register user Michael` does not grab a face the instant it is understood. The screen counts you
down instead:

```
                    Registering Michael - look at the camera  2
```

Two seconds, along the bottom of the picture. It is there because a spoken command arrives *late* —
the wake word, the recogniser, the transcript being read back to you — and what the camera used to
get was somebody looking at the screen to see whether anything had happened, not at the lens.

When the count runs out the face is measured before it is stored, and a poor look is not stored at
all. The countdown starts again, a second longer, with the reason in the same place:

```
                    Please come closer to the camera - registering Michael  3
```

Three goes, then it gives up and says why. The five things it measures, and what it wants
(`applib/face.py`):

| measure | wanted | what a failure means |
|---|---|---|
| face size | ≥ 60 px | too far from the camera to have any detail to embed |
| straightness | ≥ 0.50 | the head is turned — the nose has slid towards one eye |
| YuNet confidence | ≥ 0.90 | something is in the way, or only part of the face is in shot |
| sharpness | ≥ 60 | motion blur, or a face that was upscaled from too far away |
| consistency | ≥ 0.75 | the two looks 200 ms apart disagree: still moving, or a different person |

**Why bother.** The vector stored here is what every later match is measured against, and a bad one
does not simply fail — it drifts. Embedding one face twice, once held still and once smeared by
movement, gives two vectors 0.55 alike, while a *stranger* sits at 0.06 from the still one and 0.14
from the smeared one. The gap identity is decided on falls from 0.94 to 0.41, and the match
threshold is 0.363. That is the mechanism behind "it keeps calling me by her name".

Matching is deliberately **not** gated the same way: it gets a fresh look five times a second and
can afford a bad one. Only registration is once-and-for-all.

The thresholds above are starting points, measured off photographs scaled to the size this camera
sees. Every attempt logs what it measured, so tuning them is a matter of registering somebody a few
times and reading the console:

```
[app] register 'Michael' 1/3: size 96px  straight 0.81  score 0.96  sharpness 210  steady 0.94  nearest Marija 0.12  ->  registered
```

The screen gets the countdown and nothing else — the numbers are debug, and debug belongs in the log.

`nearest` is the odd one out: it is not a quality measure, it is **which already-registered user
this new face is most like**. A new person scoring high there means the embedder cannot tell two
people apart, which no threshold above can fix. It is the first thing to read when a name lands on
the wrong face — and it is how the Neutron SDK 3.0.0 defect above was caught, where it read 1.00 for
everybody.

### Registering somebody from a photograph

The other way in, and it needs neither the person nor the camera: upload `Nick Markovic.jpg` to the
device's **`faces/` folder** in /IOTCONNECT. The demo looks there every 20 seconds and registers
whoever the file is named after — so the people you already know you want recognised can be loaded
the night before, and the demo opens with names on the screen instead of an empty database.

The picture is judged the same way a live look is: too small a face, a turned head or a blurred
photograph is refused, and the reason is sent to the dashboard. A face already registered under a
different name is refused too, rather than replacing it — there is nobody standing there to say
which name they meant.

**Each upload is dealt with once, permanently.** Beside every downloaded picture the demo writes a
`Nick Markovic.status` file — which upload it was, and one word for how it went:

```json
{ "etag": "\"23907bab7bfff062d76014cd12e6d30d\"", "status": "registered" }
```

So a restart does not redo any of it, and unregistering somebody by voice is not undone twenty
seconds later by the folder they are still in. Why a picture was refused is not in there — that is
on the console and in the dashboard. Two ways to make it happen again:

```
re-upload the picture          a new version, so the next poll registers it
rm -rf faces/                  start over: everything is fetched and offered again
```

A picture S3 reports no ETag for is **ignored**, with a warning on the console — without one there
is no way to tell one upload of a file from the next.

## The pieces

The directories say who each file is for. `app/` is the owner's domain — what the demo *does*, and
the two files most worth reading. `applib/` is everything it does that with. `agenttools/` is
diagnostics nobody runs at a booth, and is expected to be thrown away before this reaches `main`.

| File | Role | Future module |
|---|---|---|
| `main.py` | composition root + the 30 fps frame loop + the argument surface | main |
| **`app/app.py`** | `AntiTheftApp`: every command handler, and what the overlay is told | app |
| **`app/watchdog.py`** | arming, alert and recording: when the alarm has a reason | app |
| **`app/agent.py`** | the LLM's tools, its prompt, and one question at a time | app |
| `applib/commands.py` | sentence → `Command`, and the 4-thread pool that runs it | app |
| `applib/voice.py` | wake word → blip → VAD capture → transcript → spoken reply | audiotext |
| `applib/vocabulary.py` + `config/vocabulary.json` | saying names right, and recognising them when STT mangles them | audiotext |
| `applib/audio_devices.py` + `config/audio.json` | device naming and the mixer gain NXP's code does not set | audiotext |
| `applib/iotc.py` | /IOTCONNECT: telemetry out, C2D in, snapshot to S3, KVS credentials | iotconnect |
| `applib/webrtc.py` | KVS signalling, one peer per viewer, H.264 pass-through | iotconnect |
| `applib/telemetry.py` | what the cloud should know, written by anyone, read by the publisher | iotconnect |
| `applib/scene.py` | `describe scene`: annotated frame → SmolVLM → a sentence | vlm |
| `applib/state.py` | armed flag + locked objects and their anchors, persisted across a restart | db |
| `applib/guard.py` | has a locked object moved, or gone? Anchors, debounce, re-anchoring | db |
| `applib/eventlog.py` | the first and most recent event of each kind, read back as prose | db |
| `applib/registry.py` | face database (name ↔ embedding), presence queries | db |
| `applib/face_worker.py` | face recognition off the frame loop, ~5 Hz | db/ml |
| `applib/face.py` | YuNet detect + align + SFace embed on person crops | ml/face |
| `applib/detector.py`, `applib/yolo.py` | YOLO interpreter wrapper + the pre/post math | ml |
| `applib/tracking.py` | `Tracker` + `Track` (identity, embedding, face box) | db |
| `applib/camera.py`, `applib/overlay.py` | GStreamer capture; Cairo HUD + the frame annotator the VLM needs | video |
| `connector/` | NXP's eIQ AAF Connector: the LLM's home, its own venv, installed separately | — |
| `agenttools/mic-check.py` | standalone audio first-aid tool — no payload, no venv | — |
| `agenttools/iotc-check.py` | standalone /IOTCONNECT first-aid: connect, S3, KVS, one message | — |
| `agenttools/agent-probe.py` | the `agent` path with a fake demo behind it — runs from the host | — |
| `agenttools/diag-face.py` | checks the board's `cv2` has the face classes | — |
| `benchmarks/ara/` | what the Ara-240 costs per token, and per question | — |

## One command path, four ways in

This is the piece worth reading the code for. Voice is not wired to the app; it is wired to a
**command service** that anything can push a command into — and as of koala, one of those things is
a language model:

```
voice.py ──┐  a sentence, parsed
           ├─► CommandService (4 threads) ─► app.on_command ─► "Registered Michael."
iotc.py  ──┘  a verb + arguments             │      ▲              │
              already structured             │      │              │
                                             │   app/agent.py ─────┤  the LLM's tools, from
                                             │   (a tool call)     │  inside a running 'agent'
                                             └─ raises ────────────┴─► "Michael is already
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
`agent` does the same inside the LLM, and an `arm` arriving meanwhile should not queue behind either.
Four is also the ceiling that leaves the A55s enough headroom for the camera and YOLO to hold 30 fps.

**The one re-entrant command.** Every handler runs under a single lock, so no two commands touch the
registry at once — except `agent`, which is deliberately held *outside* it. A question sits inside
the model for tens of seconds and its tools come back in through `on_command`, so taking the lock
around it would deadlock the demo against itself. `agent.py` allows one question at a time instead,
which costs nothing: the Ara generates serially anyway.

## Agent: plain English, on the Ara-240

```
dashboard: agent please disarm the alarm
   │
   ├─ iotc.py ─► CommandService ─► app._agent ─► agent.py ─ HTTP ─► connector (venv, port 3000)
   │                                                                 │
   │                                                            Ara-240 DNPU, 7B resident
   │                                                                 │
   │                        app.on_command(DISARM) ◄── tool call ◄───┘
   └─◄ ack: the first 40 characters, and the whole answer, once, as the `answer` attribute
```

**Read the answer in telemetry, not in the ack.** The dashboard renders an acknowledgement as a
tooltip, where anything longer than a few words is unreadable, so the ack carries the *beginning*
of the answer — enough to tell one reply from another — and the whole text is sent once as the
`answer` attribute. Voice has no such limit: if `agent` is ever wired to the microphone, the entire
answer is what gets spoken.

Eleven tools are offered: `get_time`, `get_status`, `take_screenshot`, `arm_alarm`, `disarm_alarm`,
`clear_alert`, `describe_alert`, `register_user`, `unregister_user`, `lock_object`, `unlock_object`.
Adding a twelfth is a docstring and a one-line call in `app/agent.py` — but read the limits first:

- **4096 tokens total, prompt plus generation**, compiled into the model; the Ara's 16 GB holds
  weights, not context. Every tool schema is part of *every* prompt, and nine of them cost roughly
  600 tokens. This is also why each question builds a **fresh agent**: strands would otherwise keep
  the conversation, and a booth running all day would overflow the window by mid-morning. Keep tool
  descriptions — and what a tool *returns* — short for the same reason.
- **~5 tokens/second.** Measured: 11–12 s for "please disarm the alarm", 37 s when the model felt
  chatty about the date. The `agent:` line on the HUD says `thinking` while it is working.
- **Temperature must never be 0.0.** The Ara samples on-device and cannot sample from a degenerate
  distribution; NXP's shipped default of 0.0 makes every request fail with an HTTP 500 whose
  traceback points at transformers and is entirely misleading.
- **The connector needs a patch** for OpenAI tool-call conformance, or strands drops every tool call
  and the model reissues it forever. `connector/install.sh` applies it. See `connector/README.md`.

The LLM is optional in every direction. No connector, no `strands-agents`, or `--no-agent`: the demo
runs, the HUD says so, and `agent` refuses politely.

To try the whole path without a camera, models or even a board, from the host:

```bash
.venv/bin/python agenttools/agent-probe.py --url http://<board>:3000/v1 \
    "please disarm the alarm"
```

## /IOTCONNECT

### Set the device up

Create the device from the **alrmtheft** template (`files/alrmtheft.json`, at the top of this repo —
it configures the cloud, and nothing on the board reads it). It declares the seven telemetry
attributes and the ten commands below, and the names have to match. koala added two of them: the
`agent` command and the `answer` attribute.
Two template settings are not in that file and have to be ticked in the web UI:

| setting | what breaks without it |
|---|---|
| **File Support** | `snapshot` saves the JPEG but cannot upload it: "file upload is not enabled" |
| **Streaming → WebRTC** | no KVS credentials and no channel: the HUD says `stream: not enabled` |

`agenttools/iotc-check.py --webrtc` tells you where you stand in about five seconds, without
starting the demo — including signing a master URL and opening the signalling channel, which is the
step most likely to fail for a reason that has nothing to do with the demo.

### What goes out

Every 4 seconds, and immediately whenever the alarm state changes:

| attribute | example | from |
|---|---|---|
| `alarm` | `disarmed` / `armed` / `alarm` | `app._set_state`, the single announce point |
| `objects` | `Nick, person, laptop` | the tracker, names filled in by face recognition |
| `fps` | `27.4` | the frame loop, twice a second |
| `scene` | the VLM's answer | **once**, after a `scene` command — not repeated afterwards |
| `answer` | the LLM's answer, or the event log | **once** — after `agent`, `alert-describe`, or *any* clearing of the log |
| `version`, `sdk_version` | `1.0.0`, `1.3.0` | constants |

`alarm` is the one word a dashboard has room for, so it is a **rollup**, and its `alarm` value means
what REC means on the screen: something is happening this minute. The screen itself never says
ALARM — it has room to show the switch, the alert line and REC separately.

A one-shot attribute is **left out of every other packet**, not sent as `null`: the back end treats
an absent field and a null one differently, and "nobody asked a question this tick" is the absent
case. `telemetry.collect()` is where that is dropped.

One-shots **queue** rather than overwrite. Two can be produced inside one publishing interval — ask
the LLM to clear the alert and the handler publishes the event log (which is about to be destroyed)
just before the model's own reply arrives — and both matter, so they go out in order, one per
message. `telemetry.set_once` is the queue.

### What comes in

| command | argument | acknowledged with |
|---|---|---|
| `user-register` | a name | `Registered Nick.` / why not |
| `user-unregister` | a name | `Unregistered Nick.` / who is actually known |
| `object-lock`, `object-unlock` | a YOLO class name | `The laptop is locked.` / what is visible |
| `alarm-arm`, `alarm-disarm` | — | `Alarm armed.` |
| `alert-clear` | — | `Alert cleared.` — the log goes to `answer` first, because this destroys it |
| `alert-describe` | — | the first 60 characters; the whole log goes to the `answer` attribute |
| `scene` | optional question | `Scene described.` — the text itself goes to the `scene` attribute |
| `snapshot` | — | `Snapshot uploaded.` (S3) or the reason it was not |
| `restart` | — | `Restarting.`, then the process comes back three seconds later |
| `agent` | a question, in words | the answer's first 40 characters — the whole of it goes to the `answer` attribute. 10–40 s |

`scene` with no argument asks the model to describe what it sees. With one, the argument *is* the
question: `scene what is the person wearing`.

`agent` is the one command whose argument is a whole sentence rather than a name. The SDK hands
arguments over already split on whitespace, so joining them back is what reconstructs the question —
which also means the question survives, but its double spaces do not.

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

**The cloud is required.** koala connects *before* it opens the camera or loads a model, on the main
thread, and nothing catches the failure: a missing config, a certificate that is not the device's, a
disabled device or an unreachable back end all stop the demo right there, printing whatever the SDK
said. Guarding for those cases here would only reword the SDK's own error, worse. The deliberate way
to run without a dashboard is `./run.sh --no-iotc`, which also takes the WebRTC stream with it —
the channel ARN comes from /IOTCONNECT.

Once connected, a network that comes and goes is *not* fatal: the publisher retries every 30 s and
the HUD says `cloud: offline` in the meantime.

### Restarting

eIQ's models stop working 60 minutes after they load. `restart` (or `--restart-after 55`) sets a
flag, the frame loop finishes its frame, everything shuts down in order, and the process `exec`s
itself with the same arguments — one pid, no orphan holding `/dev/video4`. Registered faces, the
armed flag and the locked objects are all on disk, so the demo comes back as it was in ~15 s. The
three-second delay before it happens is there so the C2D acknowledgement reaches the cloud first.

## Watching it: KVS WebRTC

Press play on the device's dashboard. The board is the **master** on a Kinesis Video Streams
signalling channel; /IOTCONNECT hands over the channel ARN and temporary AWS credentials at connect,
and one signed WebSocket stays open for the whole run. Every viewer arrives on it as an SDP offer.

The `stream:` line on the HUD says where things stand — `ready`, `2 watching`, `not enabled`,
`offline`. It is drawn *into* the picture being streamed, so a viewer can see their own arrival
change it.

### The pipeline is the design

```
v4l2src ─┬─ videoconvert ─ appsink                          clean RGB for YOLO
         └─ videoconvert ─ cairooverlay ─┬─ waylandsink     HDMI
                            (the OSD)    ├─ v4l2h264enc ─ appsink   WebRTC
                                         └─ appsink        capture.jpg
```

The second `tee` is the trick. Everything below `cairooverlay` has the overlay burned in, so the
stream needs no second renderer, and `capture.jpg` stopped being a re-render and became the actual
screen. YOLO still gets clean pixels because its branch comes off the *first* tee — inference must
never see last frame's boxes.

Two details are load-bearing and both were found the hard way:

- **The format is pinned to BGRx, not negotiated.** `cairooverlay` accepts BGRx, ARGB or RGB16 and
  will settle on whichever the sinks prefer; `read_display` unpacks BGRx. Pinning makes that
  assumption true instead of lucky.
- **There is exactly one `videoconvert` in front of the overlay.** An earlier version had one on
  every branch, which is a full-frame software colour conversion each, and the demo ran at
  **10 fps**. `v4l2h264enc` takes BGRx natively, so the VPU is fed with no conversion at all.

### Why the predecessor lagged, and this does not

An earlier project of ours streamed at inference resolution and drifted up to ten seconds behind.
The cause was the timestamps: frames were stamped `pts += 1` at `time_base = 1/30`. If the pipeline
actually delivers 24 fps, one wall-clock second produces 0.8 s of media time, and the viewer falls
*cumulatively* further behind for as long as it runs — a frame counter cannot show this, because no
frames are lost.

koala stamps each access unit with the real time it arrived, at 90 kHz, so media time and wall
clock cannot diverge however the frame rate moves.

### What aiortc does and does not do

`RTCRtpSender` looks at what `recv()` returned: a raw `Frame` gets encoded, anything else gets
`pack()`ed — packetised only. We return an `av.Packet`, so aiortc never encodes. Two things follow,
and `webrtc.py` handles both because aiortc cannot:

- **Keyframes.** `force_keyframe` is only consulted on the encode path, so a viewer joining
  mid-stream would wait out the encoder's IDR interval. We ask GStreamer for one when a peer
  connects, and again when a viewer falls behind. `h264parse config-interval=-1` repeats SPS/PPS
  before every IDR so a late joiner can decode at all.
- **Bitrate.** The RTCP `target_bitrate` hint is ignored; the rate is fixed in `camera.py`. Fine for
  a booth, and the first thing to revisit on a bad network.

A viewer that stops keeping up has its queue **flushed and resynchronised** rather than trickled
stale frames: dropping the middle of an H.264 GOP corrupts everything until the next IDR, so
skipping to now and asking for a keyframe is the honest recovery.

## Threads, and why the video never stutters

| thread | does | cadence |
|---|---|---|
| main | camera → YOLO (Neutron) → tracking → overlay | 30 fps, blocks on nothing |
| face | YuNet + SFace on person crops | ~5 Hz, skips a tick if still busy |
| voice | loads eIQ, then wake word → STT → TTS | idle until spoken to |
| command ×4 | runs a command, speaks the result | on demand |
| iotc | connects, then publishes telemetry | every 4 s, or when woken |
| paho (SDK) | MQTT socket, delivers C2D | idle |
| webrtc | asyncio: signalling, and one peer per viewer | idle until somebody watches |

The streamed video is on none of them. GStreamer's own threads compose the overlay, drive the
hardware encoder and deliver finished packets; our webrtc thread only addresses and sends them.

The voice thread **loads its own models**, ~15 s of `import torch` and decrypting weights. That
happens after the camera is already running, so the picture is live in about a second and the HUD's
`voice:` line tells you when the ears are ready. Nothing in eIQ touches Neutron, so the YOLO and
SFace offload is what buys the voice stack its CPU.

## Alarm

Three flags, not one state — `app/watchdog.py`, and it is worth reading:

| flag | what it means | on screen |
|---|---|---|
| **arming** | your own switch, and *only* that. Persisted, so an hourly restart comes back armed if you left it armed | `alarm: ARMED` |
| **alert** | the event log is not empty. There is no second flag — an alert *is* a log with something in it | `alert: the laptop was moved, 30 seconds ago` |
| **recording** | a *timed* flag: a screenshot to the cloud every 3 s, lasting 5 s past whatever last refreshed it | **REC** in the corner |

**There is no ALARM banner.** *Alarm* is an input — the switch you arm and disarm — and conflating
it with what the demo has caught is what made the old screen unreadable: a recognised user could
stand in front of a camera that still said ALARM, with nothing on screen saying why. REC does that
job now, and it is honest about tense: REC means *happening*, the alert line means *happened*.

The cloud still gets one rolled-up `alarm` value (`disarmed` / `armed` / `alarm`), because a
dashboard tile has room for one word — and there, `alarm` means the same thing REC does.

It **ships disarmed** and stays however you left it. Disarming is not "off": a locked object is still
watched. It means *people are nobody's business* — the demo stops caring who is in the room. Arming
is what you do on the way out, and it adds the room to what is guarded.

**An unknown person is fine as long as a known one is there** — armed or disarmed, and whether the
stranger is standing next to them or picking the laptop up. One rule, one window (`USER_GRACE_S`,
5 s), and it is what makes a booth workable: the visitors are strangers, the person showing them
round is not.

Three things raise an alert, and each starts a recording, adds one line to the event log and
uploads the screenshot that goes with it:

- an **unrecognised person** in view while armed, for 3 seconds. The grace is for the face
  recogniser — it needs a moment and a look at a face before it can say "that is Nick". While they
  stay, the recording keeps being refreshed, so a visitor who lingers is photographed throughout;
- a **locked object moved** by nobody the camera recognises;
- a **locked object gone** from view for 2 seconds. What went missing is remembered by name, because
  once it is gone there is nothing on screen left to point at.

A locked object is an **anchor**: the box it occupied when you locked it, kept in `state.json`. It
counts as moved when its centre drifts more than half the object's own size — fractions of the
object, so one threshold fits a laptop and a phone — and only if that holds for half a second, since
YOLO reacquires a box in odd places for a frame at a time. **Who moved it decides what happens.**
With a registered user on screen (or seen in the last 5 s) the anchor simply follows the object,
silently. With nobody recognised, it is an event — and then the anchor moves anyway, because the
object is not going back by itself and an anchor it can never satisfy again would fire on every
frame for the rest of the day. Keep carrying it and it reports again every few seconds.

**Two ways to clear it.** `clear the alert` empties the log and nothing else — the acknowledgement,
for when you have read what happened and want the screen back. **Disarming** is the bigger reset: the
log, the recording, and every locked object re-anchored where it now stands. Disarming works when the
alarm is already disarmed, because a locked object raises an alert either way.

The event log keeps the **first and the most recent event of each kind**. Not the two most recent —
the *first* is the one that cannot be recovered later, and "the laptop has been moved, and it started
four minutes ago" is a different fact from "the laptop was moved a second ago". `what happened` reads
it back, with ages rather than timestamps, and so does `agent what has happened?`.

The log lives **only in RAM**, so anything that destroys it — `clear the alert`, `disarm`, or the LLM
calling the `clear_alert` tool — publishes a copy to the cloud as `answer` first.

A sighting counts for a moment after it happens — a person for 1 s, a recognised user for 2 s. The
tracker reports only what it saw *this* frame, and YOLO drops a person for the odd frame, so without
that an empty room flickers several times a second.

## What survives a restart

eIQ kills the process after **60 minutes** — the timeout is inside NXP's compiled modules and rides
along with the models, so at a trade show the demo *will* restart between visitors. Anything set by
voice therefore goes to disk: registered faces in `faces.json`, the armed flag and locked objects in
`state.json`. Startup is about 15 seconds. koala can also do the restarting itself: see above.

Photographs pulled from the cloud survive it too — the pictures and their `.status` records are in
`faces/`, so a restart re-downloads nothing and re-registers nobody.

# The Neutron SDK, and why the demo brings its own

**Drop `eiq-neutron-sdk-linux-<version>.zip` beside `install.sh` before installing.** It is
NXP-licensed, so it cannot be committed or hosted with the demo — download it from nxp.com and put
it here. `install.sh` unpacks it to `imx-eiq-neutron-sdk/`.

The BSP already ships a Neutron delegate in `/usr/lib` and its firmware in `/lib/firmware`, and this
demo deliberately uses neither. On the image this was written against, the SDK release those came
from **miscompiled the face embedder**: SFace returned a vector that barely depended on its input, so
three different people embedded to a cosine of 1.000 of each other and every face matched every
registered user. The same release also converted YOLO into nine Neutron graphs instead of two —
12.9 ms of inference where a good build does 5.9 ms.

Nothing is replaced to fix that, so falling back costs nothing:

| piece | how it is redirected |
|---|---|
| delegate | loaded by path — `applib/neutron.py` picks the SDK's copy when it exists, else `/usr/lib` |
| firmware | `run.sh` writes the SDK's directory into `/sys/module/firmware_class/parameters/path`, which the kernel searches **before** `/lib/firmware` |
| models | converted by the matching `neutron-converter`, by `scripts/package-models.sh` |

Delete `imx-eiq-neutron-sdk/` and you are back on the BSP's runtime, unmodified.

**Three pieces, one release.** Converter, delegate and firmware must all come from the same SDK. A
mismatch does not raise — it returns wrong numbers. (A delegate from one release on another's
firmware detected *nothing at all*, silently.) So `package-models.sh` stamps the SDK it used, and its
firmware's md5, into `models/version.txt`, and the demo checks that at startup:

```
==============================================================================
WARNING: The models were converted by eiq-neutron-sdk-linux-3.1.3, but the Neutron firmware
about to run them is the BSP's own /lib/firmware, which is a different build. Expect wrong
results rather than errors - faces that all match, or nothing detected at all.
==============================================================================
```

To check by hand which firmware is live: `dmesg | grep "Booting fw"` — the size identifies the
build. Note the setting does not survive a reboot, which is why `run.sh` applies it every launch.

# Running

Useful ways to run it:

```bash
./run.sh --no-voice                            # video only: no eIQ payload, no pip packages needed
./run.sh --no-agent                            # no LLM; 'agent' then refuses politely
./run.sh --ara-url http://<other-board>:3000/v1  # the connector somewhere else
./run.sh --no-iotc                             # no cloud, same as having no credentials here
./run.sh --iotc-verbose                        # log every MQTT packet, both directions
./run.sh --restart-after 55                    # beat eIQ's 60-minute timeout, don't wait for it
./run.sh --model models/yolo11n_int8.tflite    # full-CPU baseline
./run.sh --preload-vlm                         # pay SmolVLM's 12 s at startup, not at the first ask
./run.sh --no-vlm                              # drop 'describe scene' entirely
./run.sh --text-commands command.txt           # debug: type commands instead of saying them
agenttools/mic-check.py --loop                 # when the microphone misbehaves, start here
agenttools/iotc-check.py --listen 60           # ... and when the cloud does
```

`--text-commands` writes into the *same* command service voice uses, so a handler tested that way is
tested for real:

```bash
echo 'register user Michael' > command.txt
```

## Layout

Directories carry the meaning here, and the rule is one-way: `main.py` → `app/` → `applib/`.
`applib/` never imports `app/`, and nothing imports `main.py`. That is the whole architecture.

```
koala/
├── main.py                 composition root: builds everything, wires everything
├── install.sh  run.sh      the two hooks, board-side
├── app/                    what the demo does — the owner's domain
│   ├── app.py              every command handler, and what the overlay is told
│   ├── watchdog.py         arming, alert, recording — when the alarm has a reason
│   └── agent.py            the LLM's tools and prompt
├── applib/                 how it does it — cameras, models, faces, audio, cloud, WebRTC
├── config/                 audio.json, vocabulary.json
├── agenttools/             checks for when something is wrong; not part of the demo
├── benchmarks/ara/         what the Ara-240 costs, with numbers
├── connector/              NXP's LLM server — its own venv, own installer   (venv/ not committed)
├── models/                 converted on the host, copied over               not committed
├── nxp-lib/                NXP's eIQ payload — theirs, never edited         not committed
├── venv/                                                                    not committed
├── faces.json state.json scene.jpg capture.jpg   runtime state              not committed
└── iotcDeviceConfig.json device-crt.pem device-key.pem   your device       not committed
```

The /IOTCONNECT **device template** is not here: `files/alrmtheft.json` at the top of the repo
configures the cloud side, and nothing on the board ever reads it. Neither is a `.gitignore` — what
this directory should not commit is described once, in the repo's own, because everything here gets
copied to a board where a `.gitignore` means nothing.

**Ours vs. theirs.** Everything above except `nxp-lib/` and `connector/{source,venv}` is our code,
tracked and editable. NXP's eIQ payload is copied verbatim, **never edited** and never committed —
the licence forbids modifying it and it is 139 MB of mostly encrypted weights. `main.py` puts
exactly one directory on `sys.path` (`nxp-lib/src`), which replaces the nine roots their
`pip install -e .` would have set up.

**Two venvs, and they must stay apart.** `venv/` is built with `--system-site-packages` so the BSP's
`tflite_runtime`, `gi`, `cv2` and `numpy` stay visible. `connector/venv/` is built *without* it,
because optimum-ara pins its own torch and would fight every one of those. Neither is `.venv`, so
both stay visible in a plain `ls` on the board.

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

**The compositor still will not give you one.** Weston 14 only lets a client *it* launched capture
the output (`weston_compositor_add_screenshot_authority`), so `weston-screenshooter` from a shell
answers `Output capture error: unauthorized`. Its own `Super+S` binding does launch an authorised
client — but that needs a real keyboard on the board, and this image has no `uinput` module to fake
the keypress with. `/dev/fb0` exists but is an unused emulation buffer: all zeros.

**But we no longer need it.** hyena had to re-render the boxes in OpenCV to produce `capture.jpg`;
koala pulls the composed frame off the display branch of the pipeline, so the JPEG *is* the screen,
down to the font. The re-render survives as the fallback for `--no-preview --no-webrtc`, where
nothing is composing, and for the VLM, which wants boxes drawn on the inference frame instead:

```bash
./run.sh --text-commands command.txt      # start with the file input enabled
echo snapshot > command.txt               # -> capture.jpg, overwritten each time
scp root@<board>:koala/capture.jpg .       # collect it
```

or just say **"take a snapshot"**, or send the `snapshot` command from the dashboard — which does
the same thing and then uploads the file, so /IOTCONNECT ends up holding the history that the board
deliberately does not. Same boxes, names and HUD as the HDMI screen. `--capture` puts the file
somewhere else.

If you do plug a USB keyboard in, `Super+S` works and drops PNGs of the real composited screen into
weston's home, `/home/weston`.

## Picking the camera

`/dev/videoN` is not a property of the camera. This board's own IP blocks claim fifty-odd video
nodes before a USB camera gets one — `neoisp` alone takes `/dev/video2` through `/dev/video51` —
and how many they take depends on the BSP. The same C920 that was `/dev/video4` on one image is
`/dev/video52` on the next, and a hard-coded default turns that into a demo that starts, streams a
black screen and stops with "camera returned no frame" on hardware that is working perfectly.

So the device is **found at startup**, and the line it prints is the one to read:

```
Camera     : /dev/video52  HD Pro Webcam C920 (usb-ci_hdrc.0-1)
```

`v4l2-ctl --list-devices` groups nodes under the device that owns them and names the bus each is on:

```
neoisp (platform:4ae00000.isp):
        /dev/video2
        ...
HD Pro Webcam C920 (usb-ci_hdrc.0-1):
        /dev/video52
        /dev/video53
```

Two rules pick from that, and both are about *not* knowing what the booth plugged in:

- **The bus decides, not the name.** Excluding the names we happen to know (`neoisp`, `mxc-jpeg`,
  `wave6-*`) works on this image and breaks quietly on the next one that adds an IP block. Every
  built-in block is on `platform:`; a USB webcam — any USB webcam, not just the C920 — is on
  `usb-`. So: the first device on a USB bus.
- **The node must actually capture.** A UVC camera exposes two: `/dev/video52` lists `YUYV`,
  `/dev/video53` is a metadata node and enumerates no formats at all. Open the wrong one and the
  pipeline negotiates happily and then never produces a frame, which is a nasty half-hour. Each
  candidate is asked `v4l2-ctl --list-formats` and the first that answers wins.

No camera at all is fatal at startup rather than five seconds later, and the error prints what
`v4l2-ctl` did report, so "the hub dropped it" and "it is on a bus I did not expect" look different.
`--device /dev/videoN` skips the whole thing — two USB cameras, or a node that is not first.

## Camera framerate: the exposure trap

If `capture=` in the frame log is above 33 ms, the camera — not the board — is the bottleneck. The
C920 in Aperture Priority mode stretches its exposure in a dim room (`exposure_time_absolute=666`
is 66 ms, so 15 fps) and UVC lets it drop the framerate to do it (use the device the startup line
named):

```bash
v4l2-ctl -d /dev/video52 -c exposure_dynamic_framerate=0    # hold 30 fps, accept a darker image
v4l2-ctl -d /dev/video52 -c exposure_dynamic_framerate=1    # back to the default
v4l2-ctl -d /dev/video52 --list-ctrls                       # what it is doing now
```

More light beats both. Face recognition wants the light anyway.

## Things that will bite

- **`describe scene` is slow and that is fine.** Tens of seconds on the CPU. It runs on a command
  thread; the video loop never waits for it. The first one also pays ~12 s to load SmolVLM unless
  you passed `--preload-vlm`.
- **The boxes go into the pixels.** `describe scene` burns the boxes into a copy of the *inference*
  frame (saved as `scene.jpg`) before sending it. The question also states how many boxes there are
  — if the answer describes two people where we drew one box, that mismatch is the signal. Note
  this is a different picture from the display feed, which is why the OpenCV re-render survives.
- **Two `Gst` typelibs in one process.** NXP's eIQ payload imports `gi` and GStreamer for its own
  audio backend. With voice enabled there are two live `Gst` wrappers, and passing a `MapInfo`
  produced by one to the other fails with `TypeError: argument info: Expected Gst.MapInfo, but got
  gi.repository.Gst.MapInfo`. It is only reachable in the *full* configuration, so gazelle and
  hyena never hit it — they were only ever run with `--no-voice`. `camera.py` uses
  `buffer.extract_dup()` instead of `map`/`unmap`, which never makes a `MapInfo` at all.
- **One `videoconvert` per branch will cost you two thirds of your frame rate.** Software colour
  conversion is per-frame and full-resolution. The composed branch converts once, up front, and
  pins BGRx so that every consumer downstream takes it natively.
- **The microphone hears the speaker.** The wake reader is re-synced after every reply, or the demo
  wakes itself on its own voice.
- **Readers come back from `register_reader()` disabled.** Miss the `enable()` and you get a silent
  zero-frame loop, not an error.
- **SmolVLM downloads on first use.** `--prefetch` pulls it ahead of time; nothing else except
  /IOTCONNECT reaches the network at run time.
- **A `scene` command from the dashboard takes tens of seconds too.** The acknowledgement arrives
  when the model is done, not when the command was received. That is the VLM, not the connection.
- **File Support and Streaming are template settings, not code.** Without them `snapshot` cannot
  upload and there is no signalling channel — `agenttools/iotc-check.py --webrtc` says which, in one
  line each.
- **A cloud that will not connect stops the demo**, by design, before the camera is opened. Read the
  SDK's error: it names the file or the reason. `--no-iotc` runs the vision half alone, and takes
  the WebRTC stream with it, since the channel ARN comes from /IOTCONNECT. The HDMI display is
  unaffected either way.
- **`X-Amz-ClientId` is for viewers only.** A master signs its WebSocket URL with the channel ARN
  alone; include a ClientId and KVS rejects the signed request.
- **The clock matters.** TLS rejects a certificate when the board's date is wrong. If a freshly
  booted board will not connect, check `date` before suspecting anything else.
- **One connection per device id.** AWS IoT Core evicts an existing MQTT session when a second
  client connects with the same client id, so running `./iotc-check.py` — or any host-side tool
  using the same certificates — **while the demo is running will disconnect the board**. It shows up
  as `[iotc] disconnected: Unspecified error` and the publisher reconnects on its next tick, stealing
  the session back. Harmless, briefly confusing, and not a bug in either program.
