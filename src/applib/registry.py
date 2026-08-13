"""The face database: names <-> face embeddings, persisted to a JSON file (the "DB" lineage).

Deliberately the simplest thing that works for a stand demo: a dict of name -> 128-float embedding,
saved to disk so registrations survive a restart. One embedding per user is enough to show the flow.

Matching uses cosine similarity, how SFace embeddings are meant to be compared (OpenCV's default
"same identity" threshold for SFace is ~0.363; we expose it so it's easy to tune live).

The interesting piece is `resolve_identities`: it maps *registered users onto tracks* each face cycle,
which is what makes identity robust to two people crossing. Because a registered user is an anchor, we
just re-ask "which track's face is most like Joe?" every cycle - reassignment IS swap handling. Between
cycles identity is sticky (kept through turn-away) and only moves when a user's face shows up elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from applib.tracking import Track

SFACE_COSINE_THRESHOLD = 0.363  # OpenCV's default same-identity cosine similarity for SFace

Identity = tuple[str | None, float | None]  # (matched name or None, best cosine or None) per track


class Registry:
    """Persistent store of registered users' face embeddings, plus the queries app.py/worker need."""

    def __init__(self, db_path: str, match_threshold: float = SFACE_COSINE_THRESHOLD) -> None:
        self.db_path = Path(db_path)
        self.match_threshold = match_threshold
        self._embeddings: dict[str, np.ndarray] = self._load()

    def register_user(self, name: str, embedding: np.ndarray) -> None:
        """Store (or overwrite) a user's face vector and persist immediately."""
        self._embeddings[name] = np.asarray(embedding, dtype=np.float32)
        self._save()

    def unregister_user(self, name: str) -> bool:
        """Forget a user. False if that name was never registered."""
        if name not in self._embeddings:
            return False
        del self._embeddings[name]
        self._save()
        return True

    def unregister_last(self) -> str | None:
        """Forget whoever was registered most recently - the undo for a botched registration.

        "Most recent" is insertion order, which dicts keep and `json` preserves through save/load,
        so it survives a restart. `None` when nobody is registered.
        """
        if not self._embeddings:
            return None
        name = list(self._embeddings)[-1]
        del self._embeddings[name]
        self._save()
        return name

    def compare(self, a: np.ndarray, b: np.ndarray) -> float:
        """How alike two face vectors are, on the same 0..1 cosine scale as `match_threshold`.

        Registration asks this about two looks at the *same* face 200 ms apart, to find out whether
        the person was holding still. It lives here because comparing embeddings is this file's
        business - `face.py` produces vectors and deliberately knows nothing about matching them.
        """
        return _cosine_similarity(a, b)

    def resolve_identities(
        self, embeddings: dict[int, np.ndarray], current_track_ids: list[int], previous: dict[int, Identity]
    ) -> dict[int, Identity]:
        """Assign registered users to this cycle's tracks (greedy + sticky), returning track_id -> Identity.

        `embeddings` are the tracks that yielded a face this cycle; `current_track_ids` is every live person
        track (some had no readable face). `previous` is last cycle's result, for the sticky/swap rules.
        """
        matched = self._greedy_assign(embeddings)            # {track_id: (name, score)} confident only
        claimed_users = {name for name, _ in matched.values()}

        result: dict[int, Identity] = {}
        for track_id in current_track_ids:
            if track_id in matched:
                result[track_id] = matched[track_id]         # fresh confident match this cycle
                continue
            kept = self._sticky(previous.get(track_id), claimed_users)
            if kept is not None:
                result[track_id] = kept                      # keep known name through a bad/absent look
            elif track_id in embeddings:
                result[track_id] = (None, self._best_score(embeddings[track_id]))  # face seen, no match
        return result

    def _sticky(self, previous: Identity | None, claimed_users: set[str]) -> Identity | None:
        """Keep a prior identity only if it still names someone AND that name wasn't claimed elsewhere."""
        if previous is None or previous[0] is None or previous[0] in claimed_users:
            return None
        return previous

    def _greedy_assign(self, embeddings: dict[int, np.ndarray]) -> dict[int, Identity]:
        """Best-first bipartite match: each registered user to at most one track above the threshold."""
        candidates = sorted(
            (
                (_cosine_similarity(embedding, stored), track_id, name)
                for track_id, embedding in embeddings.items()
                for name, stored in self._embeddings.items()
            ),
            reverse=True,
        )
        assigned: dict[int, Identity] = {}
        used_names: set[str] = set()
        for score, track_id, name in candidates:
            if score < self.match_threshold:
                break
            if track_id in assigned or name in used_names:
                continue
            assigned[track_id] = (name, score)
            used_names.add(name)
        return assigned

    def find_nearest_user(self, embedding: np.ndarray) -> tuple[str | None, float | None]:
        """Which registered user this face is most like, and how much. (None, None) if none exist.

        Registration puts this on the screen, and it is the one diagnostic no threshold can
        replace: a face being registered as somebody *new* that already scores high against
        somebody *old* is the mix-up happening while you watch. It says nothing about quality -
        it says the embedder cannot tell these two people apart.
        """
        if not self._embeddings:
            return None, None
        name = max(self._embeddings,
                   key=lambda user: _cosine_similarity(embedding, self._embeddings[user]))
        return name, _cosine_similarity(embedding, self._embeddings[name])

    def _best_score(self, embedding: np.ndarray) -> float | None:
        """Highest cosine to any registered user (debug '?' label); None when nobody is registered."""
        return self.find_nearest_user(embedding)[1]

    def is_person_present(self, tracks: list[Track]) -> bool:
        return any(track.class_name == "person" for track in tracks)

    def is_registered_user_present(self, tracks: list[Track]) -> bool:
        return any(track.identity is not None for track in tracks)

    def get_visible_classes(self, tracks: list[Track]) -> list[str]:
        """Distinct YOLO class names on screen right now - what "lock object" can be asked to guard."""
        return sorted({track.class_name for track in tracks})

    @property
    def user_names(self) -> list[str]:
        return sorted(self._embeddings)

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
