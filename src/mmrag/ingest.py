"""Stage 1+2 - Ingest and the page split (the "cut unit").

THE KEY IDEA: what counts as one retrievable unit is decided HERE, upstream of
embedding. This demo cuts a PDF into pages: one page = one unit = one `page_id`.
Text and picture of the same page share that id, which is what lets us merge
text hits and image hits later (late fusion).

The embedding APIs never split anything. They receive a page's text, or a page's
PNG, and return one vector for it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    page_id: str  # "<source>#p<page>", shared by the text and image vectors
    source: str
    page: int  # 1-based
    text: str  # may be empty (scanned page / figure-only page)
    image: bytes | None  # PNG bytes of the rendered page, if rendering is on
    entity: str = ""  # company entity, from the folder structure (e.g. "hk-hq")
    department: str = ""  # department, from the folder structure (e.g. "hr")


def org_metadata(rel_path: str) -> tuple[str, str]:
    """<entity>/<department>/file.pdf -> (entity, department). Folders ARE the org chart."""
    parts = rel_path.split("/")
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return "", parts[0]
    return "", ""


Renderer = Callable[[Path, int], bytes]  # (pdf path, 0-based page index) -> PNG bytes


def pdfium_renderer(scale: float = 1.5) -> Renderer:
    """Render pages with pypdfium2 (pip wheel, no system poppler needed)."""
    import pypdfium2 as pdfium  # optional: pip install -e '.[images]'

    def render(path: Path, index: int) -> bytes:
        doc = pdfium.PdfDocument(str(path))
        try:
            buf = io.BytesIO()
            doc[index].render(scale=scale).to_pil().convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
        finally:
            doc.close()

    return render


def load_pages(folder: str | Path, renderer: Renderer | None = None) -> list[Page]:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a folder: {root}")
    pages: list[Page] = []
    for path in sorted(root.rglob("*.pdf")):
        rel = path.relative_to(root).as_posix()
        entity, department = org_metadata(rel)
        for i, page in enumerate(PdfReader(path).pages):
            text = (page.extract_text() or "").strip()
            image = renderer(path, i) if renderer else None
            if text or image:  # a text-only index drops figure-only/scanned pages: no OCR here
                pages.append(Page(f"{rel}#p{i + 1}", rel, i + 1, text, image, entity, department))
    return pages
