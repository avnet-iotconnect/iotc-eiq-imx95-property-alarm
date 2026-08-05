"""The voice control loop: "Hey NXP" -> blip -> a command -> a spoken answer. (the "audiotext" lineage)

This is falcon's `stage_voice` turned into something that runs forever on its own thread, next to a
30 fps video loop it must never disturb. What it does, once per utterance:

    wake word          VIT on 30 ms frames, the only piece with no pip dependencies
    blip               "I am listening" -- played before the speech reader is synced, so the blip
                       is never recorded back and mistaken for the user talking
    3 seconds          how long the user has to *start* speaking; silence means they were not
                       talking to us and we go back to waiting for the wake word
    VAD capture        stops when the speaker stops, rather than on a fixed timer
    blip               "heard you, thinking" -- without it the pause before TTS reads as a failure
    transcribe         moonshine-base, 0.3-0.4 s
    repeat + answer    say back what was heard, then hand the text to `on_command` and say the result

`on_command` is a plain `str -> str` callback, so this file knows nothing about what the commands
*are*: `main.py` points it at the command service. The reply comes back as text whether the command
succeeded or failed, which is why the error channel in `commands.py` is a string. The call blocks
this thread until the command finishes - deliberately: we are not listening for a wake word while
the answer is still being worked out, and `describe scene` takes tens of seconds. The video loop is
untouched throughout, which is the only deadline that matters.

**Everything is loaded on this thread, not before it.** Startup is ~8 s of `import torch` plus ~6 s
of decrypting models, and doing that before the camera starts would mean fifteen seconds of black
screen at a trade show. Instead the video comes up immediately and the HUD says `voice: loading`
until this thread announces itself.

Traps carried over from falcon, all learned on the board:
- **Readers come back from `register_reader()` disabled.** Miss the `enable()` and you get a silent
  zero-frame loop, not an error.
- **The recogniser streams cumulatively.** Each yield is the whole transcript; keep the last.
- **`audio_chunk_length` does not exist**; use `current_chunk_length`, re-read every iteration.
- **The microphone hears the speaker.** Re-sync the wake reader after every reply, or the demo
  wakes itself up on its own voice.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter, sleep
from typing import Callable

import numpy as np

from applib import audio_devices, vocabulary
from applib.audio_devices import AudioConfig

logger = logging.getLogger(__name__)

WAKE_WORD = "Hey NXP"


class VoiceLoop:
    """Owns the microphone, the speaker and the four eIQ models behind them."""

    def __init__(
        self, payload: Path, audio_config: AudioConfig, on_command: Callable[[str], str],
        on_status: Callable[[str], None] | None = None, speaker_id: int = 24,
        wake_timeout_s: float = 3.0, command_seconds: float = 6.0, is_repeating: bool = True,
    ) -> None:
        self.payload = payload
        self.audio_config = audio_config
        self.on_command = on_command
        self.on_status = on_status or (lambda status: None)
        self.speaker_id = speaker_id
        self.wake_timeout_s = wake_timeout_s
        self.command_seconds = command_seconds
        self.is_repeating = is_repeating

        self._stop = Event()
        self._speak_lock = Lock()  # command results and loop prompts can both want the speaker
        self._thread = Thread(target=self._run, name="voice", daemon=True)
        self._audio = None
        self._synthesiser = None
        self._samplerate = 16000
        self._vocabulary = None

    # --- lifecycle ----------------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._audio is not None:
            self._audio.stop_capture()

    def speak(self, text: str) -> None:
        """Say something out loud and wait for it to finish. Safe to call from any thread.

        Names and acronyms are respelled on the way in - espeak-ng reads "Marija" as "ma-RID-zha"
        and "NXP" as a word. `mqs` has no hardware volume control, so digital gain is the only
        loudness lever there is.
        """
        if self._synthesiser is None:
            print(f"[voice] (not loaded yet, would say) {text}")
            return
        with self._speak_lock:
            print(f"[voice] saying: {text}")
            spoken = vocabulary.say_as(text, self._vocabulary)
            audio_data = self._synthesiser.generate(spoken)
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.concatenate(list(audio_data))
            audio_data = audio_devices.apply_digital_gain(
                audio_data.astype(np.float32), self.audio_config.playback_gain)
            self._audio.play_audio_async(audio_data, sample_rate=self._samplerate)
            sleep(len(audio_data) / self._samplerate + 0.5)
            self._audio.stop_playback()

    # --- the loop -----------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._load()
        except Exception as error:  # a missing payload must not take the video down with it
            logger.exception("voice stack failed to load")
            self.on_status("voice: unavailable")
            print(f"[voice] disabled: {type(error).__name__}: {error}")
            return

        self.on_status("voice: ready")
        self.speak(f"Voice control ready. Say {WAKE_WORD}.")
        self._wake_reader.sync_to_current()

        while not self._stop.is_set():
            if not self._wait_for_wake():
                continue
            try:
                self._handle_utterance()
            except Exception:  # one bad utterance must not end voice control for the day
                logger.exception("voice: utterance failed")
                self.on_status("voice: ready")

    def _wait_for_wake(self) -> bool:
        """Block on 30 ms frames until VIT hears the wake word (or we are asked to stop)."""
        from vit.vit import VIT

        while not self._stop.is_set():
            samples = self._wake_reader.read(VIT.SAMPLES_PER_FRAME, blocking=True, timeout=0.1)
            if samples is None or len(samples) != VIT.SAMPLES_PER_FRAME:
                continue
            boosted = audio_devices.apply_digital_gain(samples, self.audio_config.capture_gain)
            kind, info = self._detector(boosted, samples)
            if kind == "wakeword":
                print(f"[voice] woken by {info['name']} at {info['energy']:.1f} dB")
                return True
        return False

    def _handle_utterance(self) -> None:
        """One wake-to-answer cycle. Everything after the wake word happens here."""
        self._play_earcon("ww_earcon")
        self._speech_reader.enable(sync_to_current=True)  # after the blip: never record the blip
        self.on_status("voice: listening")

        captured = self._listen_until_silence()
        self._speech_reader.disable()

        if captured is None:
            print("[voice] nothing said after the wake word")
            self.speak("I did not hear a command.")
        else:
            self._play_earcon("intent_earcon")
            self.on_status("voice: thinking")
            text = self._transcribe(captured)
            print(f"[voice] heard {text!r} ({len(captured) / 16000:.1f}s at "
                  f"{audio_devices.get_peak_dbfs(captured):.1f} dBFS)")
            if not text:
                self.speak("Sorry, I did not catch that.")
            else:
                if self.is_repeating:
                    self.speak(f"You said: {text}")
                self.speak(self.on_command(text))

        self.on_status("voice: ready")
        self._wake_reader.sync_to_current()  # drop everything the mic heard while we were talking

    def _listen_until_silence(self) -> np.ndarray | None:
        """Collect one utterance: up to `wake_timeout_s` to start, then until the speaker stops.

        VAD marks the start and end of speech. We buffer a little audio *before* the start mark so
        the first syllable is not clipped - VAD needs a moment of speech before it fires, and
        without the pre-roll "register" becomes "egister". `command_seconds` is only a backstop for
        someone who never stops talking.
        """
        voice_activity = self._voice_activity
        pre_roll = deque(maxlen=voice_activity.pre_vad_samples)
        collected: list[np.ndarray] = []
        is_speaking = False
        start_deadline = perf_counter() + self.wake_timeout_s
        end_deadline = perf_counter() + self.wake_timeout_s + self.command_seconds

        while perf_counter() < end_deadline and not self._stop.is_set():
            samples = self._speech_reader.read(voice_activity.required_samples,
                                               blocking=True, timeout=0.05)
            if samples is None or len(samples) != voice_activity.required_samples:
                continue
            samples = audio_devices.apply_digital_gain(samples, self.audio_config.capture_gain)
            marks = voice_activity(samples)

            if not is_speaking:
                if marks and "start" in marks:
                    is_speaking = True
                    collected.append(np.array(pre_roll, dtype=samples.dtype))
                    collected.append(samples)
                elif perf_counter() > start_deadline:
                    return None  # they were not talking to us
                else:
                    pre_roll.extend(samples)
            else:
                collected.append(samples)
                if marks and "end" in marks:
                    break

        return np.concatenate(collected) if collected else None

    def _transcribe(self, captured: np.ndarray) -> str:
        """Feed the recogniser in its own chunk sizes and keep the LAST yield, not the sum."""
        audio = self._torch.from_numpy(captured)
        text = ""
        position = 0
        while position < len(audio):
            length = self._recogniser.current_chunk_length
            chunk = audio[position:position + length]
            position += length
            for piece in self._recogniser(chunk, ending=position >= len(audio)):
                if piece:
                    text = piece
        return text.strip()

    def _play_earcon(self, name: str) -> None:
        """Play one of NXP's short blips and wait for it to finish. Blocking on purpose."""
        import soundfile as sf

        path = self.payload / "earcons" / f"{name}.wav"
        if not path.exists():
            return
        samples, samplerate = sf.read(str(path), dtype="float32")
        if samples.ndim > 1:
            samples = samples[:, 0]
        with self._speak_lock:
            self._audio.play_audio_async(
                audio_devices.apply_digital_gain(samples, self.audio_config.playback_gain),
                sample_rate=samplerate)
            sleep(len(samples) / samplerate + 0.2)
            self._audio.stop_playback()

    # --- loading ------------------------------------------------------------------------------

    def _load(self) -> None:
        """Import and build everything. ~15 s, paid once, on this thread while the video runs."""
        self.on_status("voice: loading (torch)")
        started = perf_counter()
        from audio_manager.audio_manager_base import ReaderConfig
        from speech_to_text.speech_to_text import SpeechToText
        from speech_to_text.vad import VAD
        from tts.config import MultiSpeakerTTS16kHzQuantConfig
        from tts.model import TextToSpeech
        from vit.vit import VIT
        import torch
        self._torch = torch
        print(f"[voice] imports done in {perf_counter() - started:.1f}s")

        self.on_status("voice: loading (models)")
        self._detector = _timed("wake word (VIT)", lambda: VIT(model_path=str(
            self.payload / "src" / "vit" / "models" / "VIT_Model_en.bin")))
        self._recogniser = _timed("moonshine-base", lambda: SpeechToText(
            "moonshine-base", language="English", task="transcribe"))
        self._voice_activity = _timed("VAD (silero)", VAD)
        tts_config = MultiSpeakerTTS16kHzQuantConfig(speaker_id=self.speaker_id)
        self._synthesiser = _timed("TTS", lambda: TextToSpeech(tts_config))
        self._samplerate = tts_config.samplerate
        self._vocabulary = _timed("vocabulary (espeak phonemes)", vocabulary.load_vocabulary)
        self._audio = _timed("audio pipeline", self._create_audio)

        # Two readers on one capture stream: VIT wants int16 frames, moonshine wants float32.
        self._wake_reader = self._audio.register_reader(
            "VIT", ReaderConfig(channels=1, format="S16LE", channel_indices=[0]))
        self._speech_reader = self._audio.register_reader(
            "STT", ReaderConfig(channels=1, format="F32LE", channel_indices=[0]))
        self._audio.start_capture()
        self._wake_reader.enable(sync_to_current=True)
        print(f"[voice] ready in {perf_counter() - started:.1f}s -- say \"{WAKE_WORD}\"")

    def _create_audio(self):
        """NXP's audio manager on the GStreamer backend, with the mixer already turned up.

        GStreamer rather than ALSA because `gi` ships with the BSP while `pyalsaaudio` needs ALSA
        headers the image does not have. Both end at the same libasound. The gain has to be set
        here because NXP's own code only knows the wm896x codecs on their other EVKs.
        """
        from audio_manager.audio_factory import create_audio_manager
        from audio_manager.audio_manager_base import CaptureConfig, PlaybackConfig

        audio_devices.prepare_capture_device(
            self.audio_config.capture_device, self.audio_config.micfil_range)
        return create_audio_manager(
            backend="gstreamer",
            capture_config=CaptureConfig(
                capture_device=self.audio_config.capture_device,
                sample_rate=16000, channels=1, format="S16LE", frame_duration_ms=30),
            playback_config=PlaybackConfig(
                playback_device=self.audio_config.playback_device,
                sample_rate=16000, channels=1, format="F32LE"),
            start_glib_loop=True,
        )


def _timed(what: str, build):
    """Load something slow, saying so first: every model here decrypts on load, so it takes seconds."""
    print(f"[voice] loading {what} ...", flush=True)
    started = perf_counter()
    result = build()
    print(f"[voice]   {what} in {perf_counter() - started:.1f}s")
    return result
