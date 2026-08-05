"""Saying words correctly, and recognising them when the recogniser mangles them.

These are two different problems, and only one of them is solvable head-on.

**Saying them - solvable.** TTS phonetises through espeak-ng, which reads words by English spelling
rules: "Marija" comes out "ma-RID-zha" (`m'a-r-I-dZ-@`). The fix is to hand TTS a respelling instead
of the real word. `Mahria` gives `mˈɑːɹiə` - MAH-ree-a, stress on the first syllable. NXP supports
this natively through a `pronunciation_json=` config parameter; we do the substitution here instead,
so `vocabulary.json` stays the single source of truth for both saying a word and recognising it
(`write_tts_pronunciation_json()` emits their format from ours if we ever switch).

To derive a new `say_as`, try candidates against the phonetiser directly and read the IPA:

    espeak-ng -q --ipa -v en "Mahria"      ->  mˈɑːɹiə

**Hearing them - not solvable head-on.** moonshine-base is a compiled, encrypted model. There is no
vocabulary to extend, no grammar to bias, no hot-word list. We cannot make it emit "Marija"; it emits
"money", and that is that.

What we can do is match what it *did* emit against the short list of words we actually care about.
We are not searching a dictionary - we are picking the closest of a handful of known users or object
names, so even a mediocre acoustic match wins comfortably. Matching runs on phonemes rather than
letters, because "county" and "Kranti" are far apart as strings (`SequenceMatcher` 0.36) and close as
sounds (`kˈaʊnti` vs `kɹˈantiː`). Explicit `heard_as` entries are checked first and always win: once
you catch the recogniser producing a particular mangling, record it and the guessing stops.

Three groups, same machinery (falcon had only the first):
  `names`   - registered users. Matching decides *who* was meant.
  `objects` - guardable things, keyed by their **YOLO class name** ("cell phone"), with the words
              people actually say ("phone", "mobile") as `heard_as` and a `say_as` for TTS.
  `terms`   - plain substitutions applied before TTS. Acronyms are the main offender.
"""

from __future__ import annotations

import difflib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

VOCABULARY_PATH = Path(__file__).resolve().parent.parent / "config" / "vocabulary.json"

# Below this phoneme similarity we would rather admit we did not understand than act on the wrong
# word. Tuned so "money" -> Marija passes while unrelated words do not.
MATCH_THRESHOLD = 0.45

# Used when the answer creates something rather than finding it - registering a NEW user. Loose
# matching is right for "which of my two users did they mean"; it is wrong for "is this a name I
# already know", where a near miss silently registers Michael as Marija. An explicit `heard_as`
# still scores 1.0 and passes either threshold, which is the point of recording them.
STRICT_MATCH_THRESHOLD = 0.75


@dataclass
class Word:
    """One thing the demo has to say correctly and recognise reliably."""

    name: str  # the canonical form: a user's name, or the YOLO class name for an object
    say_as: str  # respelling handed to TTS instead of `name`
    heard_as: list[str] = field(default_factory=list)  # manglings STT has actually been seen to emit
    phonemes: str = ""  # IPA of `say_as`, computed once at load


@dataclass
class Vocabulary:
    names: list[Word]
    objects: list[Word]
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
    """Read vocabulary.json and precompute every word's phonemes once (one espeak call each)."""
    raw = json.loads(path.read_text())
    terms = {key: value for key, value in raw.get("terms", {}).items() if not key.startswith("_")}
    return Vocabulary(
        names=[_word_from(entry) for entry in raw.get("names", [])],
        objects=[_word_from(entry) for entry in raw.get("objects", [])],
        terms=terms,
    )


def _word_from(entry: dict) -> Word:
    word = Word(
        name=entry["name"],
        say_as=entry.get("say_as", entry["name"]),
        heard_as=[heard.lower() for heard in entry.get("heard_as", [])],
    )
    word.phonemes = get_phonemes(word.say_as)
    return word


def say_as(text: str, vocabulary: Vocabulary) -> str:
    """Rewrite text so TTS pronounces names, object names and acronyms correctly.

    Longest keys first, so 'i.MX95' is not half-eaten by a shorter rule.
    """
    replacements = {word.name: word.say_as for word in vocabulary.names + vocabulary.objects}
    replacements.update(vocabulary.terms)
    for original in sorted(replacements, key=len, reverse=True):
        text = text.replace(original, replacements[original])
    return text


def match_name(heard: str, vocabulary: Vocabulary,
               threshold: float = MATCH_THRESHOLD) -> tuple[str | None, float]:
    """Which known user was the recogniser probably trying to produce? (name, confidence)."""
    return match_word(heard, vocabulary.names, threshold)


def match_object(heard: str, vocabulary: Vocabulary,
                 threshold: float = MATCH_THRESHOLD) -> tuple[str | None, float]:
    """Which guardable object was meant? Returns the **YOLO class name**, e.g. 'cell phone'."""
    return match_word(heard, vocabulary.objects, threshold)


def match_word(heard: str, words: list[Word],
               threshold: float = MATCH_THRESHOLD) -> tuple[str | None, float]:
    """Core matcher: exact/`heard_as` first, then phonemes, then letters. `None` when nothing is close.

    Better to ask the user to repeat than to act on the wrong word.
    """
    heard = heard.strip().lower()
    if not heard or not words:
        return None, 0.0

    for word in words:
        if heard == word.name.lower() or heard in word.heard_as:
            return word.name, 1.0

    heard_phonemes = get_phonemes(heard)
    best_name, best_score = None, 0.0
    for word in words:
        if heard_phonemes and word.phonemes:
            score = difflib.SequenceMatcher(None, heard_phonemes, word.phonemes).ratio()
        else:
            score = difflib.SequenceMatcher(None, heard, word.name.lower()).ratio()
        if score > best_score:
            best_name, best_score = word.name, score

    return (None, best_score) if best_score < threshold else (best_name, best_score)


def match_choice(heard: str, candidates: list[str],
                 threshold: float = MATCH_THRESHOLD) -> tuple[str | None, float]:
    """Same matching against plain strings that are not in vocabulary.json.

    Needed because users get registered by voice: "unregister user Michael" has to find the Michael
    already in the face database, whether or not anyone wrote him into vocabulary.json.
    """
    return match_word(heard, [Word(name=candidate, say_as=candidate,
                                   phonemes=get_phonemes(candidate))
                              for candidate in candidates], threshold)


def write_tts_pronunciation_json(vocabulary: Vocabulary, path: Path) -> Path:
    """Emit our vocabulary in NXP's `pronunciation_json` format, for their built-in substitution.

    Only needed if we switch from doing the substitution ourselves to passing `pronunciation_json=`
    into the TTS config. Kept here so vocabulary.json stays authoritative either way.
    """
    mapping = {word.name: word.say_as for word in vocabulary.names + vocabulary.objects}
    mapping.update(vocabulary.terms)
    path.write_text(json.dumps({"english": mapping}, indent=2))
    return path
