#!/usr/bin/env python3
"""falcon -- first contact with NXP's eIQ GenAI Flow, one capability at a time.

Exercises wake word, speech-to-text, text-to-speech and the vision-language model as four
independent stages, so a failure names exactly which piece broke. None of NXP's turn-taking pipeline
is involved: we construct their libraries directly and drive them from our own loop, which is the
whole point of treating dm-eiq as a library rather than a demo.

    ./main.py devices          # what the board has, and how loud it is
    ./main.py wake             # say "Hey NXP"
    ./main.py stt              # speak after the prompt
    ./main.py voice            # say "Hey NXP", then a command -- the real flow
    ./main.py tts              # hear a sentence, say nothing
    ./main.py vlm              # describe an image
    ./main.py all

The single-capability stages each load their own model on entry, so they misrepresent startup cost.
`voice` is the honest one: it preloads everything before listening, the way the finished demo will.

Add --file to run wake and stt against NXP's reference recordings instead of a microphone, which
separates "the model is broken" from "the microphone is too quiet" -- the failure mode this board
is prone to. See audio_devices.py for why.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# NXP's library lives inside the pilot, but under its own directory: everything in nxp-lib/ is
# theirs and never edited, everything beside it is ours. One sys.path root replaces the nine that
# their `pip install -e .` would have set up.
PAYLOAD = Path(__file__).resolve().parent / "nxp-lib"
sys.path.insert(0, str(PAYLOAD / "src"))

import numpy as np  # noqa: E402

import audio_devices  # noqa: E402
import vocabulary  # noqa: E402

logger = logging.getLogger("falcon")

WAKE_WORD_REFERENCE = PAYLOAD / "testdata" / "HeyNXP_en.wav"
SPEECH_REFERENCE = PAYLOAD / "testdata" / "sample_en.wav"
IMAGE_REFERENCE = PAYLOAD / "testdata" / "delivery.jpg"
VIT_MODEL = PAYLOAD / "src" / "vit" / "models" / "VIT_Model_en.bin"
EARCONS = PAYLOAD / "earcons"
VLM_WEIGHTS = Path(__file__).resolve().parent / "models"


def create_audio(args, capture_format: str = "S16LE"):
    """Build NXP's audio manager on the GStreamer backend, with the mixer already turned up.

    GStreamer rather than ALSA because `gi` ships with the BSP while `pyalsaaudio` is a source
    distribution needing ALSA headers the image does not have. Both end at the same libasound.
    """
    from audio_manager.audio_factory import create_audio_manager
    from audio_manager.audio_manager_base import CaptureConfig, PlaybackConfig

    audio_devices.prepare_capture_device(args.mic, args.range)

    return create_audio_manager(
        backend="gstreamer",
        capture_config=CaptureConfig(
            capture_device=args.mic,
            sample_rate=16000,
            channels=1,
            format=capture_format,
            frame_duration_ms=30,
        ),
        playback_config=PlaybackConfig(
            playback_device=args.speaker,
            sample_rate=16000,
            channels=1,
            format="F32LE",
        ),
        start_glib_loop=True,
    )


def read_wav(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path)) as handle:
        if handle.getframerate() != 16000 or handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16 kHz 16-bit")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def stage_devices(args) -> bool:
    """Report the audio hardware and measure what the microphone actually delivers."""
    from audio_manager.audio_factory import print_backend_info
    from audio_manager.audio_manager_base import ReaderConfig

    print_backend_info()

    audio = create_audio(args)
    reader = audio.register_reader("level", ReaderConfig(channels=1, format="S16LE", channel_indices=[0]))
    audio.start_capture()
    reader.enable(sync_to_current=True)

    print(f"\nMeasuring {args.mic} for 5 s -- make some noise\n")
    peak = np.zeros(1, dtype=np.int16)
    deadline = time.time() + 5
    while time.time() < deadline:
        samples = reader.read(480, blocking=True, timeout=0.03)
        if samples is not None and len(samples):
            peak = np.maximum(peak, np.abs(samples).max())

    audio.stop_capture()
    level = audio_devices.get_peak_dbfs(peak)
    print(f"peak: {level:.1f} dBFS")
    print("  above -20 dBFS is healthy; below -40 dBFS the wake word will struggle")
    return level > -40


def stage_wake(args) -> bool:
    """Detect "Hey NXP". The only piece here with an unencrypted model and no pip dependencies."""
    from audio_manager.audio_manager_base import ReaderConfig
    from vit.vit import VIT

    detector = VIT(model_path=str(VIT_MODEL))

    if args.file:
        samples = read_wav(WAKE_WORD_REFERENCE)
        hits = 0
        for start in range(0, len(samples) - VIT.SAMPLES_PER_FRAME, VIT.SAMPLES_PER_FRAME):
            frame = samples[start:start + VIT.SAMPLES_PER_FRAME].copy()
            kind, info = detector(frame, frame)
            if kind == "wakeword":
                hits += 1
                print(f"  {start / 16000:6.2f}s  {info['name']}  energy={info['energy']:.1f} dB")
        print(f"\n{hits} detections in the reference recording (expected 5)")
        return hits == 5

    audio = create_audio(args)
    reader = audio.register_reader("VIT", ReaderConfig(channels=1, format="S16LE", channel_indices=[0]))
    audio.start_capture()
    reader.enable(sync_to_current=True)

    print(f'\nSay "Hey NXP" -- listening {args.seconds} s on {args.mic}\n')
    hits = 0
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        samples = reader.read(VIT.SAMPLES_PER_FRAME, blocking=True, timeout=0.03)
        if samples is None or len(samples) != VIT.SAMPLES_PER_FRAME:
            continue
        samples = audio_devices.apply_digital_gain(samples, args.gain)
        kind, info = detector(samples, samples)
        if kind == "wakeword":
            hits += 1
            print(f"  detected: {info['name']}  energy={info['energy']:.1f} dB")

    audio.stop_capture()
    print(f"\n{hits} detections")
    return hits > 0


def load_timed(what: str, build):
    """Load something slow, saying so first and reporting how long it took.

    Every model here is encrypted and decrypts on load, so startup is seconds, not milliseconds.
    Silence during that is indistinguishable from a hang.
    Printed on its own line rather than inline, because the libraries write to stdout mid-load and
    would otherwise split the timing across their output.
    """
    print(f"  loading {what} ...", flush=True)
    started = time.time()
    result = build()
    print(f"    done in {time.time() - started:.1f}s")
    return result


def speak(audio, synthesiser, samplerate: int, text: str, gain: float, vocab=None) -> None:
    """Say `text` out loud and wait for it to finish.

    Names and acronyms are respelled on the way in, because espeak-ng reads "Marija" as
    "ma-RID-zha" and "NXP" as a word -- see vocabulary.py.

    mqs has no hardware volume control, so `gain` is the only loudness lever there is.
    """
    if vocab is not None:
        text = vocabulary.say_as(text, vocab)
    audio_data = synthesiser.generate(text)
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.concatenate(list(audio_data))
    audio_data = audio_devices.apply_digital_gain(audio_data.astype(np.float32), gain)
    audio.play_audio_async(audio_data, sample_rate=samplerate)
    time.sleep(len(audio_data) / samplerate + 1.0)
    audio.stop_playback()


def play_earcon(audio, name: str, gain: float) -> None:
    """Play one of NXP's short blips and wait for it to finish.

    Blocking on purpose. The caller syncs the speech reader *after* this returns, so the blip is
    never captured back through the microphone and mistaken for the user talking.
    """
    import soundfile as sf

    path = EARCONS / f"{name}.wav"
    if not path.exists():
        return
    samples, samplerate = sf.read(str(path), dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    audio.play_audio_async(audio_devices.apply_digital_gain(samples, gain), sample_rate=samplerate)
    time.sleep(len(samples) / samplerate + 0.2)
    audio.stop_playback()


def listen_until_silence(reader, voice_activity, gain: float, max_seconds: float):
    """Collect speech and stop as soon as the speaker does, rather than running a fixed timer.

    VAD marks the start and end of an utterance. We buffer a little audio *before* the start mark so
    the first syllable is not clipped -- VAD needs a moment of speech before it fires, and without
    the pre-roll "register" becomes "egister".

    `max_seconds` is only a backstop for someone who never stops talking.
    """
    from collections import deque

    pre_roll = deque(maxlen=voice_activity.pre_vad_samples)
    collected: list[np.ndarray] = []
    is_speaking = False
    deadline = time.time() + max_seconds

    while time.time() < deadline:
        samples = reader.read(voice_activity.required_samples, blocking=True, timeout=0.05)
        if samples is None or len(samples) != voice_activity.required_samples:
            continue
        samples = audio_devices.apply_digital_gain(samples, gain)
        marks = voice_activity(samples)

        if not is_speaking:
            if marks and "start" in marks:
                is_speaking = True
                print("  hearing speech ...", flush=True)
                collected.append(np.array(pre_roll, dtype=samples.dtype))
                collected.append(samples)
            else:
                pre_roll.extend(samples)
        else:
            collected.append(samples)
            if marks and "end" in marks:
                print("  speech ended", flush=True)
                break

    if not collected:
        return None
    return np.concatenate(collected)


def transcribe(recogniser, audio) -> str:
    """ Feed the recogniser in chunks and return the final transcript.

    Two traps here, both learned the hard way:

    **The recogniser streams cumulatively.** Each yield is the whole transcript so far, not a delta,
    so concatenating them produces 'AndAnd xAnd xpAnd xp hello...'. Keep the last yield, not the sum.

    **`audio_chunk_length` does not exist on this build.** NXP's own file path uses it
    (`torch.split(audio, stt.audio_chunk_length)`), so `python -m speech_to_text -f <wav>` is broken
    for moonshine in their tree. Their microphone path uses `current_chunk_length` and re-reads it
    every iteration, because the recogniser varies its chunk size as decoding proceeds.
    """

    text = ""
    position = 0
    while position < len(audio):
        length = recogniser.current_chunk_length
        chunk = audio[position:position + length]
        position += length
        for piece in recogniser(chunk, ending=position >= len(audio)):
            if piece:
                text = piece
    return text


def stage_stt(args) -> bool:
    """Transcribe speech with moonshine-base. Pulls torch in via silero-vad, so it is the slow import."""
    from audio_manager.audio_manager_base import ReaderConfig
    from speech_to_text.speech_to_text import SpeechToText
    from speech_to_text.vad import VAD

    recogniser = load_timed(
        "moonshine-base (encrypted)", lambda: SpeechToText("moonshine-base", language="English", task="transcribe")
    )
    detector = load_timed("VAD (silero)", VAD)

    if args.file:
        from speech_to_text.utils.utils import load_audio

        audio_input, _ = load_audio(str(SPEECH_REFERENCE), sample_rate=recogniser.sample_rate)
        audio_input = detector.process(audio_input[recogniser.audio_channel_index])
        if audio_input is None:
            print("VAD found no speech in the reference recording")
            return False
        text = transcribe(recogniser, audio_input)
        print(f"\ntranscript: {text.strip()!r}")
        return bool(text.strip())

    audio = create_audio(args, capture_format="F32LE")
    reader = audio.register_reader("STT", ReaderConfig(channels=1, format="F32LE", channel_indices=[0]))
    audio.start_capture()
    reader.enable(sync_to_current=True)

    print(f"\nSpeak now -- listening {args.seconds} s on {args.mic}\n")
    speech: list[np.ndarray] = []
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        samples = reader.read(detector.required_samples, blocking=True, timeout=0.05)
        if samples is None or len(samples) != detector.required_samples:
            continue
        speech.append(audio_devices.apply_digital_gain(samples, args.gain))

    audio.stop_capture()
    captured = np.concatenate(speech) if speech else np.zeros(0, dtype=np.float32)
    print(f"captured {len(captured) / 16000:.1f} s at {audio_devices.get_peak_dbfs(captured):.1f} dBFS peak")

    import torch

    trimmed = detector.process(torch.from_numpy(captured))
    if trimmed is None:
        print("VAD found no speech -- the microphone is probably too quiet")
        return False

    text = transcribe(recogniser, trimmed)
    print(f"\ntranscript: {text.strip()!r}")
    return bool(text.strip())


def stage_pronounce(args) -> bool:
    """Audition respellings of a name out loud, so you can tune pronunciation by ear.

    IPA on a screen only gets you close. Whether `Kranntee` lands as KRAN-tee or KRAYN-tee is a
    question about how the VITS voice renders those phonemes, and the only way to settle it is to
    listen. This speaks each candidate in turn with its IPA printed, so you can pick a winner and
    paste it into vocabulary.json as `say_as`.

        ./main.py pronounce --text Kranti
        ./main.py pronounce --text Kranti --candidates "Kranti,Kranntee,Krahntee,Krunty,Cronntee"
    """
    from tts.config import MultiSpeakerTTS16kHzQuantConfig
    from tts.model import TextToSpeech

    vocab = vocabulary.load_vocabulary()
    if args.candidates:
        candidates = [word.strip() for word in args.candidates.split(",") if word.strip()]
    else:
        candidates = [args.text] + [name.say_as for name in vocab.names if name.name == args.text]

    tts_config = MultiSpeakerTTS16kHzQuantConfig(speaker_id=args.voice)
    synthesiser = load_timed("TTS (encrypted, quantised)", lambda: TextToSpeech(tts_config))
    audio = load_timed("audio pipeline", lambda: create_audio(args))

    print(f"\nAuditioning {len(candidates)} spellings of {args.text!r}:\n")
    for candidate in candidates:
        print(f"  {candidate:<14} {vocabulary.get_phonemes(candidate)}", flush=True)
        # No vocabulary substitution here -- we are testing the raw spelling, not the mapping.
        speak(audio, synthesiser, tts_config.samplerate, candidate, args.gain)
        time.sleep(0.4)

    print(f"\nPick the one that sounded right and set it as say_as for {args.text!r} "
          f"in {vocabulary.VOCABULARY_PATH.name}.")
    return True


def are_vlm_weights_cached() -> bool:
    """True once SmolVLM has been pulled from Hugging Face into our tree.

    `ensure_local_or_download_hf()` decides per-file with `os.path.exists`, so the presence of the
    ONNX sessions is the honest signal -- and far better than telling the user we are downloading
    every single run.
    """
    return VLM_WEIGHTS.is_dir() and any(VLM_WEIGHTS.rglob("*.onnx"))


def stage_prefetch(args) -> bool:
    """Fetch and warm everything that would otherwise be fetched at first use.

    Run this once after installing, on a network you trust. A trade show floor is exactly the wrong
    place to discover that SmolVLM's weights are still sitting on Hugging Face, and the only piece
    here that touches the network is the VLM -- everything else ships in the tarball.

    It also decrypts every model once, which surfaces a corrupt payload now rather than mid-demo.
    """
    print("Importing torch, transformers and the eIQ libraries ...", flush=True)
    started = time.time()
    from speech_to_text.speech_to_text import SpeechToText
    from speech_to_text.vad import VAD
    from tts.config import MultiSpeakerTTS16kHzQuantConfig
    from tts.model import TextToSpeech
    from vit.vit import VIT
    print(f"    done in {time.time() - started:.1f}s\n")

    print("Warming every model (decrypting weights, downloading what is missing):")
    load_timed("wake word (VIT)", lambda: VIT(model_path=str(VIT_MODEL)))
    load_timed("moonshine-base", lambda: SpeechToText("moonshine-base", language="English", task="transcribe"))
    load_timed("VAD (silero)", VAD)
    load_timed("TTS (encrypted, quantised)", lambda: TextToSpeech(MultiSpeakerTTS16kHzQuantConfig(speaker_id=args.voice)))
    stage_vlm(args)

    print(f"\nAll models present. VLM weights cached in {VLM_WEIGHTS} "
          f"({sum(f.stat().st_size for f in VLM_WEIGHTS.rglob('*') if f.is_file()) / 1e6:.0f} MB).")
    print("Nothing else reaches the network at run time.")
    return are_vlm_weights_cached()


def stage_voice(args) -> bool:
    """The real flow: wait for "Hey NXP", then transcribe whatever is said next.

    Everything is loaded **before** any listening starts. That is the point of this stage as much as
    the handoff is: a visitor at the booth cannot wait ten seconds for moonshine to decrypt after
    saying the wake word, so the finished demo preloads once at startup and never pays that cost
    again. The individual stages each load their own model and are misleading about startup cost.

    Both readers sit on one capture stream. On detection the speech reader is synced to *now*, so it
    starts just after the wake word rather than replaying it -- the cheap version of the seek in
    `eiq_genai_flow.py:621-654`.
    """
    print("Importing torch, transformers and the eIQ libraries (slow, and before anything loads) ...", flush=True)
    started = time.time()
    from audio_manager.audio_manager_base import ReaderConfig
    from speech_to_text.speech_to_text import SpeechToText
    from speech_to_text.vad import VAD
    from tts.config import MultiSpeakerTTS16kHzQuantConfig
    from tts.model import TextToSpeech
    from vit.vit import VIT
    import torch
    print(f"    done in {time.time() - started:.1f}s\n")

    print("Preloading every model up front (this is what the demo will do at startup):")
    detector = load_timed("wake word (VIT)", lambda: VIT(model_path=str(VIT_MODEL)))
    recogniser = load_timed(
        "moonshine-base", lambda: SpeechToText("moonshine-base", language="English", task="transcribe")
    )
    voice_activity = load_timed("VAD (silero)", VAD)
    tts_config = MultiSpeakerTTS16kHzQuantConfig(speaker_id=args.voice)
    synthesiser = load_timed("TTS (encrypted, quantised)", lambda: TextToSpeech(tts_config))
    audio = load_timed("audio pipeline", lambda: create_audio(args))
    vocab = load_timed("vocabulary (espeak phonemes)", vocabulary.load_vocabulary)
    print(f"  total startup: {time.time() - started:.1f}s -- paid once, never again\n")

    wake_reader = audio.register_reader("VIT", ReaderConfig(channels=1, format="S16LE", channel_indices=[0]))
    speech_reader = audio.register_reader("STT", ReaderConfig(channels=1, format="F32LE", channel_indices=[0]))
    audio.start_capture()
    wake_reader.enable(sync_to_current=True)

    print(f'Say "Hey NXP" -- listening {args.seconds} s on {args.mic}')
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        samples = wake_reader.read(VIT.SAMPLES_PER_FRAME, blocking=True, timeout=0.03)
        if samples is None or len(samples) != VIT.SAMPLES_PER_FRAME:
            continue
        kind, info = detector(audio_devices.apply_digital_gain(samples, args.gain), samples)
        if kind == "wakeword":
            print(f"\n  woken by {info['name']} at {info['energy']:.1f} dB")
            break
    else:
        audio.stop_capture()
        print("\nNo wake word heard.")
        return False

    # Blip first, then sync the reader, so the blip itself is never recorded back.
    play_earcon(audio, "ww_earcon", args.gain)
    speech_reader.enable(sync_to_current=True)

    print(f"  now say a command (stops when you do, {args.command_seconds} s at most)")
    captured = listen_until_silence(speech_reader, voice_activity, args.gain, args.command_seconds)
    audio.stop_capture()

    if captured is None:
        print("\nNo speech heard after the wake word.")
        speak(audio, synthesiser, tts_config.samplerate, "Sorry, I did not catch that.", args.gain, vocab)
        return False

    # Second blip: heard you, now thinking. Without it the pause before TTS reads as a failure.
    play_earcon(audio, "intent_earcon", args.gain)
    print(f"  captured {len(captured) / 16000:.1f}s at {audio_devices.get_peak_dbfs(captured):.1f} dBFS"
          f" -- transcribing ...", flush=True)

    started = time.time()
    text = transcribe(recogniser, torch.from_numpy(captured)).strip()
    print(f"\ncommand: {text!r}   ({time.time() - started:.1f}s to transcribe)")

    # "Register user <name>" is the command the demo actually cares about, and the one the
    # recogniser reliably mangles -- it hears "money" for Marija. Match the tail against the
    # registered names rather than trusting the transcript verbatim.
    heard_name = vocabulary.extract_name(text)
    if heard_name:
        matched, score = vocabulary.match_name(heard_name, vocab)
        if matched:
            print(f"  heard {heard_name!r} -> matched registered user {matched} ({score:.2f})")
            reply = f"Registered {matched}"
        else:
            print(f"  heard {heard_name!r} -> no registered user close enough ({score:.2f})")
            reply = "I did not recognise that name. Please try again."
    else:
        reply = f"You said: {text}"

    # Closing the loop out loud is the whole point: wake -> understand -> answer, all on the board.
    speak(audio, synthesiser, tts_config.samplerate, reply, args.gain, vocab)
    return bool(text)


def stage_tts(args) -> bool:
    """Speak a sentence. Needs espeak-ng on the system for the phonetizer."""
    from tts.config import MultiSpeakerTTS16kHzQuantConfig
    from tts.model import TextToSpeech

    config = MultiSpeakerTTS16kHzQuantConfig(speaker_id=args.voice)
    synthesiser = load_timed("TTS (encrypted, quantised)", lambda: TextToSpeech(config))

    started = time.time()
    audio_data = synthesiser.generate(args.text)
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.concatenate(list(audio_data))
    elapsed = time.time() - started
    duration = len(audio_data) / config.samplerate
    print(f"generated {duration:.1f} s of speech in {elapsed:.1f} s ({duration / elapsed:.1f}x realtime)")

    # mqs has no hardware volume control, so loudness is entirely ours to set here.
    audio_data = audio_devices.apply_digital_gain(audio_data.astype(np.float32), args.gain)

    audio = create_audio(args)
    audio.play_audio_async(audio_data, sample_rate=config.samplerate)
    time.sleep(duration + 1.5)
    audio.stop_playback()
    print(f"\nplayed at {audio_devices.get_peak_dbfs(audio_data):.1f} dBFS peak on {args.speaker}")
    return True


def stage_vlm(args) -> bool:
    """Describe an image with SmolVLM. Weights come from Hugging Face on first run."""
    import vlm.modeling_vlm as modeling_vlm
    from vlm.modeling_vlm import make_VLM

    # models_config.py points models_dir inside the payload, which we treat as read-only and
    # replace wholesale on every refresh. Redirect the download to our own tree instead.
    VLM_WEIGHTS.mkdir(exist_ok=True)
    modeling_vlm.models_dir = str(VLM_WEIGHTS) + "/"


    class Params:
        n_threads = -1
        use_neutron = False
        max_new_tokens = 96
        assistant_prompt = "How can I help you?"

    if are_vlm_weights_cached():
        print(f"  SmolVLM weights already cached in {VLM_WEIGHTS}")
    else:
        print(f"  SmolVLM weights NOT cached -- downloading from Hugging Face into {VLM_WEIGHTS}.")
        print("  Run './main.py prefetch' once on a good network to avoid this at the booth.")
    model = load_timed(
        "smolvlm-256M q8", lambda: make_VLM("smolvlm-256M", "q8", user_params=Params(), fixed_image=args.image)
    )
    model.image_features = load_timed("vision encoder pass", lambda: model.run_vision(model.image_inputs))

    started = time.time()
    description = "".join(model.process_message(args.question))
    print(f"\n{description.strip()}\n\n({time.time() - started:.1f} s)")
    return bool(description.strip())


STAGES = {
    "devices": stage_devices,
    "wake": stage_wake,
    "stt": stage_stt,
    "voice": stage_voice,
    "tts": stage_tts,
    "vlm": stage_vlm,
    "prefetch": stage_prefetch,
    "pronounce": stage_pronounce,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*STAGES, "all"], help="which capability to exercise")
    parser.add_argument("--mic", help="capture device, overriding audio.json (alias, substring, or ALSA name)")
    parser.add_argument("--speaker", help="playback device, overriding audio.json")
    parser.add_argument("--range", type=int, help="micfil hardware gain 0-15, overriding audio.json")
    parser.add_argument("--gain", type=float, help="digital gain on samples, overriding audio.json")
    parser.add_argument("--seconds", type=int, default=15, help="how long to listen")
    parser.add_argument("--command-seconds", type=int, default=6,
                        help="how long to listen for a command after the wake word, in the voice stage")
    parser.add_argument("--file", action="store_true", help="use NXP's reference recordings, not a microphone")
    parser.add_argument("--text", default="Alarm armed. The laptop is now being guarded.", help="text for tts")
    parser.add_argument("--voice", type=int, default=24, help="TTS speaker id, 1-904")
    parser.add_argument("--candidates", default="", help="comma-separated respellings to audition in the pronounce stage")
    parser.add_argument("--image", default=str(IMAGE_REFERENCE),
                        help="image for vlm; defaults to the doorbell-camera reference shot")
    parser.add_argument("--question", default="Describe the person at the door.", help="question for vlm")
    parser.add_argument("--verbose", action="store_true", help="show the library's own logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    # audio.json holds the defaults; anything given on the command line wins.
    config = audio_devices.load_audio_config()
    args.mic = audio_devices.resolve_device(args.mic, is_capture=True) if args.mic else config.capture_device
    args.speaker = audio_devices.resolve_device(args.speaker, is_capture=False) if args.speaker else config.playback_device
    args.range = args.range if args.range is not None else config.micfil_range
    args.gain = args.gain if args.gain is not None else config.playback_gain

    if not (PAYLOAD / "src" / "vit").is_dir():
        print(f"No payload at {PAYLOAD}/src -- unpack the tarball first (see nxp-lib/README.md)", file=sys.stderr)
        return 2

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    results = {}
    for name in stages:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        try:
            results[name] = STAGES[name](args)
        except Exception as error:
            logger.exception("stage %s raised", name)
            print(f"FAILED: {type(error).__name__}: {error}")
            results[name] = False

    print(f"\n{'=' * 70}")
    for name, passed in results.items():
        print(f"  {name:10s} {'ok' if passed else 'FAILED'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
