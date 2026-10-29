"""Opt-in test with the REAL CLIP model (needs `uv sync --extra images`; downloads ~600 MB once).

    RUN_CLIP=1 uv run pytest tests/test_live_clip.py -v
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_CLIP"), reason="set RUN_CLIP=1 (needs .[images])")

HK = Path(__file__).parent.parent / "examples" / "harbour-lane"
CHART = "hk-hq/finance/expense-claims.pdf#p2"


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    from mmrag.config import Config
    from mmrag.embed import HashEmbedder, get_multimodal_embedder
    from mmrag.index import build_index
    from mmrag.ingest import pdfium_renderer
    from mmrag.store import Store

    cfg = Config(db_path=str(tmp_path_factory.mktemp("clip") / "t.db"), index_mode="multimodal", top_k=3)
    text, mm = HashEmbedder(), get_multimodal_embedder("clip")
    assert mm is not None, "CLIP not available - run: uv sync --extra images"
    store = Store(cfg.db_path)
    build_index(HK, cfg, text, mm, store, pdfium_renderer())
    return cfg, store, text, mm


def test_image_mode_finds_the_chart_page(index):
    from mmrag.retrieve import retrieve

    cfg, store, text, mm = index
    pages, _ = retrieve(store, cfg, "a bar chart", "image", mm_embedder=mm)
    assert pages[0].page_id == CHART
    assert 0.15 < pages[0].parts["image"][1] < 0.4  # CLIP text->image cosine lives on a low scale


def test_fusion_needs_a_noise_floor_for_rrf(index):
    """Without a floor, a noise-level text hit outranks the true chart page under RRF."""
    from dataclasses import replace

    from mmrag.retrieve import retrieve

    cfg, store, text, mm = index
    no_floor, _ = retrieve(store, replace(cfg, text_min_score=0.0), "a bar chart", "fusion", text, mm)
    with_floor, _ = retrieve(store, replace(cfg, text_min_score=0.1), "a bar chart", "fusion", text, mm)
    assert with_floor[0].page_id == CHART
    assert no_floor[0].page_id != CHART or no_floor[0].parts.get("text", (0, 1))[1] > 0.1
