"""Late fusion - STANDALONE module (pure Python, no imports from this package).

Copy this one file into any project that merges ranked lists from several retrievers.

Late fusion = retrieve separately per modality, THEN merge by a shared key (here the
page_id). The alternative, early fusion, mixes the vectors before searching; it needs
one model that embeds both modalities jointly.

Why not just add the raw scores? Each modality has its own score scale. Text-to-text
cosine from a sentence model often sits around 0.4-0.8, while CLIP text-to-image
cosine sits around 0.2-0.35 even for a perfect match. Summing raw scores lets one
modality drown the other. Two scale-free options are provided:

  rrf     Reciprocal Rank Fusion: score = sum_over_lists weight / (60 + rank).
          Uses ranks only, so score scales never matter. Robust default.
  minmax  Rescale each list's scores to 0..1, then weighted sum. Keeps score gaps,
          but one outlier hit can squash the rest of its list.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fused:
    key: str
    score: float
    parts: dict[str, tuple[int, float]] = field(default_factory=dict)  # list name -> (1-based rank, raw score)


def fuse(
    ranked: dict[str, list[tuple[str, float]]],
    method: str = "rrf",
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
) -> list[Fused]:
    """ranked: {"text": [(key, raw_score), ...best first], "image": [...]} -> merged, best first."""
    if method not in {"rrf", "minmax"}:
        raise ValueError("method must be 'rrf' or 'minmax'")
    weights = weights or {}
    fused: dict[str, Fused] = {}
    for name, items in ranked.items():
        w = weights.get(name, 1.0)
        scores = [s for _, s in items]
        lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
        for rank, (key, raw) in enumerate(items, start=1):
            if method == "rrf":
                contrib = w / (rrf_k + rank)
            else:
                contrib = w * ((raw - lo) / (hi - lo) if hi > lo else 1.0)
            f = fused.setdefault(key, Fused(key, 0.0))
            f.score += contrib
            f.parts[name] = (rank, raw)
    return sorted(fused.values(), key=lambda f: f.score, reverse=True)
