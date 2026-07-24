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
        """Fill Track.identity for any track whose face embedding matches a registered user."""
        for track in tracks:
            track.identity = self._match(track.embedding) if track.embedding is not None else None

    def is_person_present(self, tracks: list[Track]) -> bool:
        return any(track.class_name == "person" for track in tracks)

    def is_registered_user_present(self, tracks: list[Track]) -> bool:
        return any(track.identity is not None for track in tracks)

    @property
    def user_names(self) -> list[str]:
        return sorted(self._embeddings)

    def _match(self, embedding: np.ndarray) -> str | None:
        """Best registered user above the cosine threshold, or None (unfamiliar face)."""
        best_name, best_similarity = None, self.match_threshold
        for name, stored in self._embeddings.items():
            similarity = _cosine_similarity(embedding, stored)
            if similarity >= best_similarity:
                best_name, best_similarity = name, similarity
        return best_name

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
