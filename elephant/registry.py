"""The face database: names <-> face embeddings, persisted to a JSON file (the "DB" lineage).

Deliberately the simplest thing that works for a stand demo: a dict of name -> 128-float embedding,
saved to disk so registrations survive a restart. No index, no averaging over multiple shots - one
embedding per user is enough to show the flow. Swap in something fancier later without touching callers.

Matching uses cosine similarity, which is how SFace embeddings are meant to be compared. OpenCV's own
default "same identity" cosine threshold for SFace is ~0.363; we expose it so it's easy to tune live.

Two responsibilities, both DB-shaped:
  identify(tracks)            - turn each track's face embedding into a name (fills Track.identity)
  is_*_present(tracks)        - the questions the alarm logic in app.py asks each frame
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tracking import Track

SFACE_COSINE_THRESHOLD = 0.363  # OpenCV's default same-identity cosine similarity for SFace


class Registry:
    """Persistent store of registered users' face embeddings, plus the presence queries app.py needs."""

    def __init__(self, db_path: str, match_threshold: float = SFACE_COSINE_THRESHOLD) -> None:
        self.db_path = Path(db_path)
        self.match_threshold = match_threshold
        self._embeddings: dict[str, np.ndarray] = self._load()

    def register_user(self, name: str, embedding: np.ndarray) -> None:
        """Store (or overwrite) a user's face vector and persist immediately."""
        self._embeddings[name] = np.asarray(embedding, dtype=np.float32)
        self._save()

    def identify(self, tracks: list[Track]) -> None:
        """Fill Track.identity (matched name) and Track.match_score (best cosine, for the debug overlay)."""
        for track in tracks:
            if track.embedding is None:
                track.identity, track.match_score = None, None
                continue
            name, best_similarity = self._best_match(track.embedding)
            track.identity = name
            track.match_score = best_similarity

    def is_person_present(self, tracks: list[Track]) -> bool:
        return any(track.class_name == "person" for track in tracks)

    def is_registered_user_present(self, tracks: list[Track]) -> bool:
        return any(track.identity is not None for track in tracks)

    @property
    def user_names(self) -> list[str]:
        return sorted(self._embeddings)

    def _best_match(self, embedding: np.ndarray) -> tuple[str | None, float | None]:
        """Return (name if best cosine clears the threshold else None, best cosine or None if no users)."""
        if not self._embeddings:
            return None, None
        best_name, best_similarity = max(
            ((name, _cosine_similarity(embedding, stored)) for name, stored in self._embeddings.items()),
            key=lambda pair: pair[1],
        )
        return (best_name if best_similarity >= self.match_threshold else None), best_similarity

    def _load(self) -> dict[str, np.ndarray]:
        if not self.db_path.exists():
            return {}
        raw = json.loads(self.db_path.read_text())
        return {name: np.asarray(vector, dtype=np.float32) for name, vector in raw.items()}

    def _save(self) -> None:
        serializable = {name: vector.tolist() for name, vector in self._embeddings.items()}
        self.db_path.write_text(json.dumps(serializable, indent=2))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0
