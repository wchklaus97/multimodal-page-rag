"""End-to-end with a stub multimodal embedder, so no model download is needed.

The stub's "page image" is just caption bytes (supplied by a fake renderer) and its image/text
embeddings share a hash space - enough to exercise indexing, image-only and fusion paths.
"""

from pathlib import Path

import pytest
from _minipdf import make_pdf

from mmrag.config import Config
from mmrag.embed import HashEmbedder, MultimodalEmbedder
from mmrag.index import build_index
from mmrag.ingest import load_pages, org_metadata
from mmrag.retrieve import retrieve
from mmrag.store import Store

EXAMPLES = Path(__file__).parent.parent / "examples" / "harbour-lane"


class StubMM(MultimodalEmbedder):
    name = "stub-mm"

    def __init__(self):
        self._h = HashEmbedder(64)

    def embed_images(self, images):
        return self._h.embed([b.decode() for b in images])

    def embed_texts(self, texts):
        return self._h.embed(texts)


def fake_renderer(path, index):  # "image" of a page = a caption
    captions = {"expense-claims.pdf": ["claims policy", "bar chart quarterly expense by department"]}
    return captions.get(path.name, ["page"] * 9)[index].encode()


def _build(tmp_path, mode, folder=EXAMPLES):
    cfg = Config(db_path=str(tmp_path / "t.db"), index_mode=mode, top_k=3)
    text, mm = HashEmbedder(), StubMM()
    store = Store(cfg.db_path)
    build_index(folder, cfg, text, mm, store, fake_renderer if mode == "multimodal" else None)
    return cfg, store, text, mm


def test_org_metadata_from_folders():
    assert org_metadata("hk-hq/hr/x.pdf") == ("hk-hq", "hr")
    assert org_metadata("hr/x.pdf") == ("", "hr")
    assert org_metadata("x.pdf") == ("", "")


def test_pages_are_the_cut_unit_with_shared_page_id():
    pages = load_pages(EXAMPLES)
    ids = [p.page_id for p in pages]
    assert "hk-hq/hr/leave-and-mpf-policy.pdf#p1" in ids and "hk-hq/hr/leave-and-mpf-policy.pdf#p2" in ids
    assert len(ids) == len(set(ids))


def test_text_mode_has_no_image_index(tmp_path):
    cfg, store, text, _ = _build(tmp_path, "text")
    assert store.has_index("text") and not store.has_index("image")
    with pytest.raises(LookupError):
        retrieve(store, cfg, "chart", "image", text, StubMM())


def test_text_mode_finds_hr_policy_and_org_filter_works(tmp_path):
    cfg, store, text, _ = _build(tmp_path, "text")
    pages, _ = retrieve(store, cfg, "how many days of annual leave", "text", text)
    assert pages[0].source.endswith("leave-and-mpf-policy.pdf") and pages[0].department == "hr"
    only_sz, _ = retrieve(store, cfg, "who approves orders above RMB 50,000", "text", text, entity="sz-subsidiary")
    assert only_sz and all(p.entity == "sz-subsidiary" for p in only_sz)
    # Shenzhen has no leave policy: zero-similarity pages are dropped instead of padding the result
    irrelevant, _ = retrieve(store, cfg, "how many days of annual leave", "text", text, entity="sz-subsidiary")
    assert irrelevant == []


def test_figure_page_has_text_vector_too_but_image_mode_finds_it_by_caption(tmp_path):
    cfg, store, text, mm = _build(tmp_path, "multimodal")
    pages, _ = retrieve(store, cfg, "bar chart quarterly expense by department", "image", mm_embedder=mm)
    assert pages[0].page_id == "hk-hq/finance/expense-claims.pdf#p2"
    assert set(pages[0].parts) == {"image"}


def test_fusion_merges_by_page_id(tmp_path):
    cfg, store, text, mm = _build(tmp_path, "multimodal")
    pages, notes = retrieve(store, cfg, "quarterly expense claims chart", "fusion", text, mm)
    assert not notes
    assert len({p.page_id for p in pages}) == len(pages)  # one row per page even if found twice
    assert any(set(p.parts) == {"text", "image"} for p in pages)


def test_fusion_degrades_to_text_when_no_image_index(tmp_path):
    cfg, store, text, _ = _build(tmp_path, "text")
    pages, notes = retrieve(store, cfg, "annual leave", "fusion", text, None)
    assert pages and "text only" in notes[0]


def test_figure_only_page_is_dropped_by_text_index_but_kept_with_images(tmp_path):
    folder = tmp_path / "scan" / "ops"
    folder.mkdir(parents=True)
    (folder / "scan.pdf").write_bytes(make_pdf([{"lines": [], "ops": "0 0 1 rg 100 100 50 50 re f\n"}]))
    assert load_pages(tmp_path / "scan") == []  # no text layer, no renderer: nothing to index
    assert len(load_pages(tmp_path / "scan", lambda p, i: b"img")) == 1


def test_changing_embedder_is_refused(tmp_path):
    cfg, store, text, mm = _build(tmp_path, "text")
    with pytest.raises(RuntimeError, match="built with"):
        retrieve(store, cfg, "leave", "text", HashEmbedder(128))
