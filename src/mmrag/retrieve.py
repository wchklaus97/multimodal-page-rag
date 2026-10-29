"""Stage 5 - Retrieval: text-only, image-only, or late fusion; optional org filter."""

from __future__ import annotations

from dataclasses import dataclass, field

from .fusion import fuse
from .store import Hit, Store

MODES = ("text", "image", "fusion")


@dataclass
class FusedPage:
    page_id: str
    source: str
    page: int
    text: str
    image_path: str | None
    entity: str
    department: str
    score: float
    parts: dict[str, tuple[int, float]] = field(default_factory=dict)  # modality -> (rank, raw cosine)


def _check_embedder(store: Store, modality: str, embedder) -> None:
    built = store.index_info(modality)
    if built != f"{embedder.name}|{embedder.dim}":
        raise RuntimeError(
            f"{modality} index was built with {built!r} but the configured embedder is "
            f"{embedder.name!r}|{embedder.dim}. Re-ingest or fix the config."
        )


def retrieve(
    store: Store,
    cfg,
    question: str,
    mode: str,
    text_embedder=None,
    mm_embedder=None,
    department: str | None = None,
    entity: str | None = None,
) -> tuple[list[FusedPage], list[str]]:
    """Returns (pages, notes). `notes` explains any degradation, e.g. fusion falling back to text."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    notes: list[str] = []
    use_text = mode in {"text", "fusion"}
    use_image = mode in {"image", "fusion"}

    if use_image and not (store.has_index("image") and mm_embedder):
        if mode == "image":
            raise LookupError(
                "No usable image index. Ingest with index_mode=multimodal and an image embedder "
                "(pip install -e '.[images]'), or use --mode text."
            )
        notes.append("fusion -> text only: no image index / image embedder available.")
        use_image = False
    if use_text and not store.has_index("text"):
        if mode == "text":
            raise LookupError("No text index. Run `mmrag ingest <folder>` first.")
        use_text = False

    filtering = bool(department or entity)
    pool = cfg.fusion_pool * (5 if filtering else 1)  # post-filtering needs a deeper candidate list

    min_score = {"text": cfg.text_min_score, "image": cfg.image_min_score}

    def keep(h: Hit) -> bool:
        # RRF uses ranks only, so a near-zero "match" would still earn credit: cut those first.
        if h.score <= min_score[h.modality]:
            return False
        return (not department or h.department == department) and (not entity or h.entity == entity)

    by_modality: dict[str, list[Hit]] = {}
    if use_text:
        _check_embedder(store, "text", text_embedder)
        by_modality["text"] = [h for h in store.search("text", text_embedder.embed([question])[0], pool) if keep(h)]
    if use_image:
        _check_embedder(store, "image", mm_embedder)
        by_modality["image"] = [h for h in store.search("image", mm_embedder.embed_texts([question])[0], pool) if keep(h)]

    info = {h.page_id: h for hits in by_modality.values() for h in hits}
    merged = fuse(
        {m: [(h.page_id, h.score) for h in hits] for m, hits in by_modality.items()},
        cfg.fusion_method,
        {"text": cfg.text_weight, "image": cfg.image_weight},
    )
    pages = [
        FusedPage(f.key, info[f.key].source, info[f.key].page, info[f.key].text, info[f.key].image_path,
                  info[f.key].entity, info[f.key].department, f.score, f.parts)
        for f in merged[: cfg.top_k]
    ]
    return pages, notes
