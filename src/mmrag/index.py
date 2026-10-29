"""Stages 1-4 glue: Ingest (page split) -> Embed (text and/or image) -> Store."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .ingest import load_pages
from .store import Store


def pages_dir_for(db_path: str) -> Path:
    return Path(f"{db_path}.pages")


def build_index(
    folder,
    cfg,
    text_embedder,
    mm_embedder,
    store: Store,
    renderer=None,
    log: Callable[[str], None] = lambda _m: None,
    batch_size: int = 16,
) -> int:
    want_images = cfg.index_mode == "multimodal"
    if want_images and (mm_embedder is None or renderer is None):
        log("note: index_mode=multimodal but image rendering/embedding is unavailable -> indexing TEXT ONLY")
        want_images = False

    pages = load_pages(folder, renderer if want_images else None)
    log(f"[1/3 ingest+page split] {len(pages)} page units from {folder}")

    store.ensure_index("text", text_embedder.name, text_embedder.dim)
    if want_images:
        store.ensure_index("image", mm_embedder.name, mm_embedder.dim)
    out_dir = pages_dir_for(store.path)
    if want_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]
        with_text = [p for p in batch if p.text]
        tvecs = dict(zip((p.page_id for p in with_text), text_embedder.embed([p.text for p in with_text]))) if with_text else {}
        with_img = [p for p in batch if want_images and p.image]
        ivecs = dict(zip((p.page_id for p in with_img), mm_embedder.embed_images([p.image for p in with_img]))) if with_img else {}
        for p in batch:
            img_path = None
            if p.page_id in ivecs:
                img_path = str(out_dir / (p.page_id.replace("/", "__").replace("#", "_") + ".png"))
                Path(img_path).write_bytes(p.image)
            store.add_page(p, tvecs.get(p.page_id), ivecs.get(p.page_id), img_path)
    log(f"[2/3 embed]  text: {text_embedder.name}" + (f"  image: {mm_embedder.name}" if want_images else "  image: (skipped)"))
    log(f"[3/3 store]  {store.path}")
    return len(pages)
