# falcon — bringing NXP's eIQ GenAI Flow in as a library

Fourth pilot. Where `dolphin` proved the Neutron YOLO path and `elephant` added face recognition,
`falcon` proves we can call NXP's voice and vision-language stack **as a library**, from our own loop,
without adopting their turn-taking demo pipeline.

Each capability is its own stage, so a failure names exactly which piece broke. **`voice` is the real
flow** — the others exist to isolate faults.

| stage | what you do | what it proves |
|---|---|---|
| `devices` | make noise | the microphone is delivering signal |
| `wake` | say "Hey NXP" | VIT detects the wake word |
| `stt` | speak a sentence | moonshine-base transcribes it |
| `tts` | **nothing, just listen** | a sentence comes out of the speaker |
| `vlm` | nothing | SmolVLM describes the reference doorbell image |
| **`voice`** | **say "Hey NXP", then a command** | **wake → blip → transcribe → match name → speak back** |
| `prefetch` | nothing | everything is downloaded and decrypted, ready to run offline |
| `pronounce` | listen and pick | auditions respellings so you can tune a name by ear |

## Layout — ours vs. theirs

The split is the point. Everything at the top level of `falcon/` is our code, tracked and editable.
Everything inside `nxp-lib/` is NXP's, copied verbatim, **never edited**, and never committed — the
licence forbids modifying it and it is 139 MB of mostly encrypted weights.

```
falcon/
├── main.py  audio_devices.py  vocabulary.py  mic-check.py    ours, tracked
├── audio.json  vocabulary.json  requirements.txt             ours, tracked
├── run.sh  install.sh                                        ours, tracked
├── venv/                                                     gitignored
├── models/            SmolVLM weights, pulled at first run   gitignored
└── nxp-lib/           NXP payload -- theirs, never edited    gitignored
    ├── src/           the six packages we import
    ├── testdata/      reference wavs + doorbell image
    └── earcons/       the three blips
```

**The payload lives in two places, on purpose.** On the board it is `falcon/nxp-lib/`, fetched by
`install.sh`, because the pilot has to be self-contained there. On a development PC it is
`nxp-lib/` at the repo root, fetched by `scripts/local-setup.sh`, so one copy serves every pilot and
the eventual main app rather than being duplicated per pilot. Both are gitignored.

`main.py` puts exactly one directory on `sys.path` — `nxp-lib/src` beside it — which replaces the
nine roots their `pip install -e .` would have set up. On the PC the same root is registered with
your venv through a `.pth` file, so imports resolve identically in the editor.

What is inside `nxp-lib/src`, and what each package needs beyond it:

| package | form | needs |
|---|---|---|
| `vit/` | one `.so` + readable `vit.py` + **unencrypted** model | `shared_utils` |
| `speech_to_text/` | 15 `.so` + encrypted moonshine-base weights | `shared_utils`, `silero-vad` (pulls torch) |
| `tts/` | 10 `.so` + encrypted quantised weights | `shared_utils`, **espeak-ng on the system** |
| `vlm/` | plain source, no weights | `transformers` + `torch`; weights come from Hugging Face |
| `audio_manager/` | plain source, both backends | `gi` (GStreamer, what we drive) |
| `shared_utils/` | two `.so` | — |

**The compiled modules are `cpython-313-aarch64`.** They pin Python 3.13 and only load on the board.
Importing them on an x86 development machine fails, and that is expected — the PC copy exists so
editors can resolve the plain-Python parts and you can read what you are calling into.

## Setting it up

**On your PC**, so the editor can resolve imports and you can read NXP's sources:

```bash
source .venv/bin/activate      # your venv, however you make it
scripts/local-setup.sh         # fetches the payload into ./nxp-lib and registers it
pip install -r falcon/requirements.txt   # optional, for third-party imports to resolve
```

**On the board**, in `~/falcon` — `install.sh` fetches the payload itself, so this is the whole setup:

```bash
./install.sh        # payload, espeak-ng into /opt/dm-eiq, then a --system-site-packages venv
./run.sh prefetch   # once: pull SmolVLM while the network is good
./run.sh voice      # say "Hey NXP", then a command
```

**Every time you change our code**, from the PC:

```bash
scp falcon/*.py falcon/*.sh falcon/*.json falcon/*.txt root@<board>:falcon/
```

Our code and the payload move separately on purpose: ours changes constantly and is a few kilobytes,
the payload is 129 MB and changes only when eIQ does.

**To cut a new payload** (only when eIQ moves) — standalone, no arguments, no checkout needed:

```bash
scripts/dm-eiq-package.sh      # clones release/v3.0, prunes, writes dm-eiq-genai-flow-lib.tgz
```

Rename it for hosting; `VERSION.txt` inside records the upstream commit. Then update the URL in
`falcon/install.sh` and `scripts/local-setup.sh`.

Individual stages, and the options that matter when something misbehaves:

```bash
falcon/run.sh wake --file                              # reference recording, no microphone involved
falcon/run.sh devices --mic plughw:CARD=micfilaudio,DEV=0
falcon/run.sh wake --gain 4 --seconds 20               # digital boost on top of the mixer gain
falcon/run.sh tts --text "Alarm armed." --voice 24
falcon/run.sh vlm --image some.jpg --question "Describe the person."
falcon/run.sh stt --verbose                            # show the library's own logging
```

`--file` is the important one. It runs wake and stt against NXP's own reference recordings, which
separates "the model is broken" from "the microphone is too quiet" — and on this board that second
failure is the likely one.

## Configuring audio: `audio.json`

One file holds the device choices and levels. Everything else reads it, and any command-line flag
overrides it.

```json
{
  "capture_device": "micfil",
  "playback_device": "mqs",
  "micfil_range": 10,
  "capture_gain": 1.0,
  "playback_gain": 1.0
}
```

Devices may be named three ways, so swapping in a USB microphone later needs no knowledge of ALSA:

| written as | means |
|---|---|
| `micfil`, `c920`, `mqs` | an alias |
| `Fifine` | any substring of a card name in `arecord -l` — this is the one to use for a new USB mic |
| `plughw:CARD=C920,DEV=0` | a full ALSA device string, passed through untouched |

`falcon/mic-check.py --list` prints every device the board has, in all three forms.

## Tuning the microphone: `mic-check.py`

Start here whenever audio misbehaves. It is standalone — pure GStreamer and numpy, no payload and no
venv — so it works before anything else is installed and cannot be broken by the rest of the pilot.
It is meant to stay in the demo permanently, as the booth's first-aid tool.

```bash
falcon/mic-check.py --list                     # what hardware this board has
falcon/mic-check.py                            # record, meter, play back, using audio.json
falcon/mic-check.py --loop                     # record, listen, adjust, repeat
falcon/mic-check.py --mic Fifine --range 8     # try another mic and another hardware gain
falcon/mic-check.py --gain 4                   # digital boost on playback only
```

It records, prints a 10 Hz level meter with a labelled scale, reports peak and RMS, then plays the
audio back so you hear the whole chain end to end.

**Reading the meter.** The bar is loudness *right now*, from silence at the left to digital full
scale at the right:

```
   -60          -40           -20      -6  0  dBFS
  [################################........]  -11.0
```

Aim to keep peaks between **−20 and −6 dBFS while speaking**. Below −40 the wake word struggles;
above −1 the signal clips, which hurts recognition more than being slightly quiet does.

Measured on this board:

| micfil `Range` | ambient meter | RMS | peak |
|---|---|---|---|
| 6 (stock) | −54 dBFS | −35.6 | −10.7 |
| 10 (current default) | −30 dBFS | −20.2 | −11.0 |
| 12 | −23 dBFS | −15.6 | −4.7 |

A 30 dB spread. 12 was tried first and put speech peaks at −3.9 dBFS, close enough to clipping to be
worth backing off, so `audio.json` ships 10. Retune if the booth is louder than the lab.

## The audio trap on this board

This cost real time, so it is worth stating plainly.

NXP's audio manager raises capture and playback mixer levels on startup, in
`set_audio_device_config.py`. It only recognises the **wm8960 and wm8962 codecs** fitted to their
other EVKs, where it pushes `Capture` to 60 and `Headphone` to 110. The i.MX95 EVK has **micfil** for
capture and **mqs** for playback. Neither matches, the function's final branch writes a debug log,
and every level stays at its quiet default. That is the entire explanation for the barely audible
`arecord | aplay` loopback on this hardware.

We cannot edit the payload, so `audio_devices.py` sets the levels before the audio manager is built:

- **micfil** exposes a 0–15 `Range` control per channel, shipped at 6. We set 10, worth about **+15 dB
  RMS** measured on the board.
- **The C920** is already at its 15/15 maximum, so there is nothing to gain there.
- **mqs exposes no mixer controls at all** — zero. Output loudness can only be raised digitally, by
  scaling samples before they reach the sink, which is what `--gain` does on the `tts` stage.

Also worth knowing: the **3.5 mm headset microphone does not enumerate as a capture device**. Only
micfil and the C920's USB microphone appear in `arecord -l`, so a headset mic is not an option here.

## Why GStreamer and not pyalsaaudio

NXP's audio manager ships two interchangeable backends, and their own `auto` mode tries GStreamer
first. Both end at the same `libasound` — GStreamer's `alsasrc`/`alsasink` are ALSA clients — so this
was never "ALSA vs. not ALSA", only which Python binding gets there.

GStreamer wins because `gi` ships with the BSP, while `pyalsaaudio` is a source distribution needing
ALSA headers the image lacks. NXP's installer reacts to that by building all of alsa-lib into
`/usr/local`, leaving a second `libasound.so.2` beside the BSP's that audio programs can silently
bind to. Choosing GStreamer deletes that whole hazard. Verified on the board: 495 of ~500 frames read
in a 15-second window, with `pyalsaaudio` absent.

`audio_manager_alsa.py` still ships in the payload — 15 KB — so `pip install pyalsaaudio` is the
escape hatch if the GStreamer path ever misbehaves.

## Things that will bite

**Everything here runs on the CPU.** Nothing in eIQ touches Neutron, so our YOLO and SFace offload is
what buys the voice stack its headroom.

**One-hour session timeout.** NXP enforces it inside the compiled modules and it rides along with the
models. The demo must restart well inside the hour, so alarm state and registered users have to
persist to disk.

**VLM downloads on first run.** SmolVLM weights come from Hugging Face. `main.py` repoints
`models_dir` at `falcon/models/` so they do not land inside the payload, which we replace wholesale
on every refresh.

## Status

Verified on the board:

- wake word against the reference recording — 5 of 5 detections, energy ≈ −37 dB
- wake word loading with **zero pip packages installed**
- GStreamer live capture feeding VIT — 495/500 frames
- the micfil gain fix — +12 dB RMS

Not yet run: `stt`, `tts`, `vlm`, all of which were blocked behind the venv and the espeak-ng build.
`whisper/feature.so` is still in the payload on the suspicion that `model_config.so` enumerates every
model at import; drop it once an STT run disproves that.
