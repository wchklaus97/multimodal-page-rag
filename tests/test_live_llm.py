"""Opt-in tests against a real LLM (OpenAI or any OpenAI-compatible API such as OpenRouter).

Skipped by default. Run with:
    RUN_LIVE_LLM=1 LIVE_MODEL=openai/gpt-4o-mini uv run pytest tests/test_live_llm.py
"""

import os
from pathlib import Path

import pytest

from mmrag.config import Config
from mmrag.embed import HashEmbedder
from mmrag.generate import generate
from mmrag.index import build_index
from mmrag.retrieve import retrieve
from mmrag.store import Store

pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE_LLM") and os.environ.get("OPENAI_API_KEY")),
    reason="live LLM test: set RUN_LIVE_LLM=1 and OPENAI_API_KEY",
)
MODEL = os.environ.get("LIVE_MODEL", "openai/gpt-4o-mini")
HK = Path(__file__).parent.parent / "examples" / "harbour-lane"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    cfg = Config(db_path=str(tmp_path_factory.mktemp("live") / "t.db"), index_mode="text", top_k=2)
    emb = HashEmbedder()
    store = Store(cfg.db_path)
    build_index(HK, cfg, emb, None, store)
    return cfg, store, emb


def test_answer_from_figure_page_text(setup):
    cfg, store, emb = setup
    q = "Which department had the highest quarterly expense claims?"
    pages, _ = retrieve(store, cfg, q, "text", emb)
    answer = generate(q, pages, "openai", MODEL)
    assert "Sales" in answer


def test_entity_filter_then_answer(setup):
    cfg, store, emb = setup
    q = "Who co-approves Shenzhen purchase orders above RMB 50,000?"
    pages, _ = retrieve(store, cfg, q, "text", emb, entity="sz-subsidiary")
    answer = generate(q, pages, "openai", MODEL)
    assert "Finance Manager" in answer
