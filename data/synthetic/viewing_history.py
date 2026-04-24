"""
OmniStack-RS — Phase 0a: Synthetic Viewing History Generator

Simulates Stage 1 of the 6-Stage Personalization Firewall:
user interactions captured locally (pauses, skips, rewinds, rewatches).

Five canonical user personas define taste manifolds in genre space.
Temporal decay (30-day half-life) ensures recent viewing signals
outweigh stale history — matching how real taste evolves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

# ── Canonical genre vocabulary (15 dimensions) ────────────────────────────
GENRES = [
    "horror", "thriller", "sci_fi", "romance", "comedy",
    "drama", "action", "documentary", "foreign", "animation",
    "crime", "mystery", "fantasy", "biography", "musical",
]
NUM_GENRES = len(GENRES)
GENRE_INDEX = {g: i for i, g in enumerate(GENRES)}

# ── User persona definitions ───────────────────────────────────────────────
# Each persona is a dict of genre → preference weight (unnormalized).
# Unlisted genres default to 0.02 (background noise).
PERSONAS: dict[str, dict[str, float]] = {
    "80s_horror_fan": {
        "horror": 0.90, "thriller": 0.60, "sci_fi": 0.30,
        "crime": 0.30, "mystery": 0.20, "action": 0.15,
        "romance": 0.05, "documentary": 0.05,
    },
    "korean_romcom_fan": {
        "romance": 0.90, "comedy": 0.70, "drama": 0.50,
        "foreign": 0.80, "fantasy": 0.20, "animation": 0.15,
        "musical": 0.10, "horror": 0.02,
    },
    "prestige_drama": {
        "drama": 0.90, "thriller": 0.50, "documentary": 0.40,
        "biography": 0.40, "crime": 0.35, "mystery": 0.30,
        "foreign": 0.30, "comedy": 0.10,
    },
    "action_blockbuster": {
        "action": 0.95, "sci_fi": 0.60, "thriller": 0.40,
        "fantasy": 0.35, "comedy": 0.30, "animation": 0.20,
        "drama": 0.10, "horror": 0.10,
    },
    "arthouse_cinephile": {
        "foreign": 0.90, "drama": 0.80, "documentary": 0.70,
        "biography": 0.50, "mystery": 0.40, "musical": 0.30,
        "animation": 0.25, "action": 0.05,
    },
}


@dataclass
class ViewingEvent:
    """One movie-watching session captured in Stage 1."""
    movie_id: str
    genre_vector: np.ndarray   # (NUM_GENRES,) float32, sums to 1.0
    watch_fraction: float      # 0=skipped, 1.0=finished, >1=rewatched
    pause_count: int           # proxy for emotional friction / engagement
    rewind_count: int          # proxy for scenes worth re-experiencing
    timestamp: datetime


def _build_persona_base_vector(persona: str) -> np.ndarray:
    """Convert persona weight dict to a normalized float32 genre vector."""
    weights = PERSONAS[persona]
    vec = np.full(NUM_GENRES, 0.02, dtype=np.float64)
    for genre, w in weights.items():
        vec[GENRE_INDEX[genre]] = w
    vec /= vec.sum()
    return vec.astype(np.float32)


def generate_user_history(
    persona: str,
    n_movies: int = 500,
    noise_std: float = 0.15,
    seed: int = 42,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> list[ViewingEvent]:
    """
    Generate a synthetic viewing history for one user of the given persona.

    Args:
        persona:    One of the keys in PERSONAS.
        n_movies:   Number of viewing events to generate.
        noise_std:  Gaussian std added to genre preferences per event.
                    Keeps users distinguishable but not perfectly separable.
        seed:       RNG seed for reproducibility.
        start_date: Earliest possible viewing timestamp.
        end_date:   Latest possible viewing timestamp (defaults to now).

    Returns:
        List of ViewingEvent, sorted ascending by timestamp.
    """
    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona '{persona}'. Choose from: {list(PERSONAS)}")

    if start_date is None:
        start_date = datetime(2020, 1, 1)
    if end_date is None:
        end_date = datetime.now()

    rng = np.random.RandomState(seed)
    base_vec = _build_persona_base_vector(persona)
    total_days = max(1, (end_date - start_date).days)

    events: list[ViewingEvent] = []
    for i in range(n_movies):
        # Perturb genre preferences with Gaussian noise, then re-normalize
        noise = rng.randn(NUM_GENRES).astype(np.float32) * noise_std
        genre_vec = np.clip(base_vec + noise, 1e-4, None)
        genre_vec /= genre_vec.sum()

        # Engagement metrics correlated with how well the movie fits the persona
        genre_match = float(np.dot(genre_vec, base_vec))  # cosine-like score
        watch_fraction = float(np.clip(rng.normal(genre_match * 1.2, 0.20), 0.0, 2.0))
        pause_count = int(rng.poisson(max(0.5, 3.0 + (1.0 - genre_match) * 5.0)))
        rewind_count = int(rng.poisson(max(0.1, genre_match * 4.0)))

        # Scatter timestamps uniformly within [start_date, end_date]
        day_offset = int(rng.uniform(0, total_days))
        timestamp = start_date + timedelta(days=day_offset)

        events.append(ViewingEvent(
            movie_id=f"mv_{persona[:4]}_{i:04d}_{seed}",
            genre_vector=genre_vec,
            watch_fraction=watch_fraction,
            pause_count=pause_count,
            rewind_count=rewind_count,
            timestamp=timestamp,
        ))

    events.sort(key=lambda e: e.timestamp)
    return events


def history_to_embedding(
    history: list[ViewingEvent],
    dim: int = 128,
    half_life_days: float = 30.0,
    projection_seed: int = 0,
    reference_time: Optional[datetime] = None,
) -> np.ndarray:
    """
    Convert a viewing history to a dense float32 user embedding.

    Two-step process:
      1. Temporally-decayed genre aggregation:
         Weight each event by exp(-ln(2) / half_life_days × days_ago) × engagement.
         At half_life_days ago, weight = 0.5. At 0 days ago, weight = 1.0.
         Engagement = watch_fraction + 0.10×rewind_count − 0.02×pause_count.
         This ensures recent taste outweighs stale history.

      2. Fixed random projection from NUM_GENRES → dim:
         A seeded orthonormal projection matrix expands the 15-dim genre
         summary into a high-dimensional space where user manifolds are
         more separable by the GrassmannianProjector.

    Args:
        history:         List of ViewingEvent (output of generate_user_history).
        dim:             Output embedding dimension (default 128).
        half_life_days:  Temporal decay half-life. 30 days is the default.
        projection_seed: Seed for the random projection matrix (shared across users).
        reference_time:  "Now" for decay computation. Defaults to datetime.now().

    Returns:
        float32 ndarray of shape (dim,), L2-normalized.
    """
    if not history:
        return np.zeros(dim, dtype=np.float32)

    if reference_time is None:
        reference_time = datetime.now()

    decay_rate = math.log(2.0) / half_life_days

    # ── Step 1: Temporally-decayed weighted aggregation ───────────────────
    aggregated = np.zeros(NUM_GENRES, dtype=np.float64)
    total_weight = 0.0

    for event in history:
        days_ago = max(0.0, (reference_time - event.timestamp).total_seconds() / 86400.0)
        time_weight = math.exp(-decay_rate * days_ago)

        # Engagement: rewatching/rewinding is a strong positive signal;
        # excessive pausing (disengagement) is a mild negative signal.
        engagement = event.watch_fraction + 0.10 * event.rewind_count - 0.02 * event.pause_count
        engagement = max(0.01, engagement)

        w = time_weight * engagement
        aggregated += w * event.genre_vector.astype(np.float64)
        total_weight += w

    if total_weight > 0.0:
        aggregated /= total_weight

    # ── Step 2: Fixed orthonormal random projection NUM_GENRES → dim ──────
    rng = np.random.RandomState(projection_seed)
    raw = rng.randn(max(dim, NUM_GENRES), NUM_GENRES).astype(np.float32)
    # QR decomposition gives an orthonormal basis: each row of Q is a unit vector
    Q, _ = np.linalg.qr(raw)          # Q shape: (max(dim, NUM_GENRES), NUM_GENRES)
    projection = Q[:dim, :]           # (dim, NUM_GENRES)

    embedding = projection @ aggregated.astype(np.float32)

    # L2 normalize so the Grassmannian projector works on the unit hypersphere
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding /= norm

    return embedding.astype(np.float32)
