"""Saying names correctly, and recognising them when the recogniser mangles them.

These are two different problems, and only one of them is solvable head-on.

**Saying them — solvable.** TTS phonetises through espeak-ng, which reads names by English spelling
rules: "Marija" comes out "ma-RID-zha" (`m'a-r-I-dZ-@`). The fix is to hand TTS a respelling instead
of the real name. `Mahreeya` gives `mˈɑːɹiːjə` -- MAH-ree-ya, stress on the first syllable.

NXP supports this natively, via a config parameter rather than a file the library picks up on its
own -- documented only in their README, which is why it is easy to miss:

    MultiSpeakerTTS16kHzQuantConfig(speaker_id=24, pronunciation_json="tts/pronunciation.json")
    # {"english": {"Stephane": "Steffen", "CEO": "C E O", ...}}

We do the substitution here instead, so `vocabulary.json` stays the single source of truth for both
saying a name and recognising it. Switching to their parameter is a small change if it ever proves
better -- `write_tts_pronunciation_json()` emits their format from ours.

To derive a new `say_as`, try candidates against the phonetiser directly and read the IPA:

    espeak-ng -q --ipa -v en "Mahreeya"      ->  mˈɑːɹiːjə

**Hearing them — not solvable head-on.** moonshine-base is a compiled, encrypted model. There is no
vocabulary to extend, no grammar to bias, no hot-word list. We cannot make it emit "Marija"; it emits
"money", and that is that.

What we can do is match what it *did* emit against the short list of names we actually care about.
The trick is that we are not searching a dictionary -- we are picking the closest of two or three
registered users, so even a mediocre acoustic match wins comfortably. Matching runs on phonemes
rather than letters, because "county" and "Kranti" are far apart as strings (`SequenceMatcher` 0.36)
and close as sounds (`kˈaʊnti` vs `kɹˈantiː`).

Explicit `heard_as` entries are checked first and always win: once you catch the recogniser producing
a particular mangling, record it and the guessing stops.
"""

from __future__ import annotations

import difflib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

VOCABULARY_PATH = Path(__file__).resolve().parent / "vocabulary.json"

# Below this phoneme similarity we would rather admit we did not understand than register the wrong
# person. Tuned so "money" -> Marija passes while unrelated words do not.
MATCH_THRESHOLD = 0.45


@dataclass
class Name:
    name: str
    say_as: str
    heard_as: list[str] = field(default_factory=list)
    phonemes: str = ""


@dataclass
class Vocabulary:
    names: list[Name]
    terms: dict[str, str]


def get_phonemes(text: str) -> str:
    """Ask espeak-ng how a word sounds. Returns '' if espeak is unavailable."""
    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", "en", text],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("espeak-ng unavailable, falling back to letter matching: %s", error)
        return ""


def load_vocabulary(path: Path = VOCABULARY_PATH) -> Vocabulary:
    """Read vocabulary.json and precompute each name's phonemes once."""
    raw = json.loads(path.read_text())
    names = []
    for entry in raw.get("names", []):
        name = Name(
            name=entry["name"],
            say_as=entry.get("say_as", entry["name"]),
            heard_as=[heard.lower() for heard in entry.get("heard_as", [])],
        )
        name.phonemes = get_phonemes(name.say_as)
        names.append(name)
    terms = {key: value for key, value in raw.get("terms", {}).items() if not key.startswith("_")}
    return Vocabulary(names=names, terms=terms)


def say_as(text: str, vocabulary: Vocabulary) -> str:
    """Rewrite text so TTS pronounces names and acronyms correctly.

    Longest keys first, so 'i.MX95' is not half-eaten by a shorter rule.
    """
    replacements = {name.name: name.say_as for name in vocabulary.names}
    replacements.update(vocabulary.terms)
    for original in sorted(replacements, key=len, reverse=True):
        text = text.replace(original, replacements[original])
    return text


def match_name(heard: str, vocabulary: Vocabulary) -> tuple[str | None, float]:
    """Find which registered name the recogniser was probably trying to produce.

    Returns (name, confidence). `None` when nothing is close enough -- better to ask the user to
    repeat than to register the wrong person.
    """
    heard = heard.strip().lower()
    if not heard:
        return None, 0.0

    for name in vocabulary.names:
        if heard == name.name.lower() or heard in name.heard_as:
            return name.name, 1.0

    heard_phonemes = get_phonemes(heard)
    best_name, best_score = None, 0.0
    for name in vocabulary.names:
        if heard_phonemes and name.phonemes:
            score = difflib.SequenceMatcher(None, heard_phonemes, name.phonemes).ratio()
        else:
            score = difflib.SequenceMatcher(None, heard, name.name.lower()).ratio()
        if score > best_score:
            best_name, best_score = name.name, score

    if best_score < MATCH_THRESHOLD:
        return None, best_score
    return best_name, best_score


def write_tts_pronunciation_json(vocabulary: Vocabulary, path: Path) -> Path:
    """Emit our vocabulary in NXP's `pronunciation_json` format, for their built-in substitution.

    Only needed if we switch from doing the substitution ourselves to passing
    `pronunciation_json=` into the TTS config. Kept here so vocabulary.json stays authoritative
    either way.
    """
    mapping = {name.name: name.say_as for name in vocabulary.names}
    mapping.update(vocabulary.terms)
    path.write_text(json.dumps({"english": mapping}, indent=2))
    return path


# Words the recogniser sprinkles between "user" and the name -- "Register user of Maria",
# "Bridge the user a Kranti". They are never part of a name, and leaving them in drags the phonetic
# score down: "of Maria" matched Marija at 0.47 where "Maria" alone scores far higher.
FILLER_WORDS = {"of", "the", "a", "an", "is", "as", "to", "for", "and", "in", "it"}

# What the recogniser produces in place of "user". It rarely gets the whole phrase right --
# "Bridge the user" for "Register user" -- so we anchor on the last word before the name.
USER_ANCHORS = ("user", "used", "usar", "uses", "eraser")


def extract_name(command: str) -> str:
    """Pull the name out of 'register user <name>'. Returns '' if the phrase is not that shape.

    Anchors on the word before the name rather than matching the whole command, because the
    recogniser mangles the leading verb freely ("Bridge the user plantier") while getting the
    anchor right. Filler words after the anchor are dropped.
    """
    words = command.strip().rstrip(".").split()
    lowered = [word.lower() for word in words]
    for anchor in USER_ANCHORS:
        if anchor in lowered:
            tail = words[lowered.index(anchor) + 1:]
            while tail and tail[0].lower() in FILLER_WORDS:
                tail = tail[1:]
            return " ".join(tail)
    return ""
