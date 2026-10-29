from pathlib import Path

from test_pipeline import StubMM, fake_renderer

from mmrag.config import Config
from mmrag.embed import HashEmbedder
from mmrag.evaluate import Setting, default_settings, evaluate_setting, load_qa, to_markdown
from mmrag.index import build_index
from mmrag.ingest import load_pages
from mmrag.store import Store

ROOT = Path(__file__).parent.parent
HK = ROOT / "examples" / "harbour-lane"
QA = load_qa(ROOT / "eval" / "harbour_lane_pages_qa.jsonl")


def test_qa_set_points_at_real_pages_and_has_both_kinds():
    pages = {p.page_id for p in load_pages(HK)}
    assert all(q.page_id in pages for q in QA)
    assert {q.kind for q in QA} == {"text", "visual"} and len({q.id for q in QA}) == len(QA)


def test_default_settings_skip_image_modes_without_image_index():
    assert [s.mode for s in default_settings(False)] == ["text"]
    assert {s.mode for s in default_settings(True)} == {"text", "image", "fusion"}


def test_evaluate_setting_metrics_are_consistent(tmp_path):
    cfg = Config(db_path=str(tmp_path / "t.db"), index_mode="multimodal")
    text, mm = HashEmbedder(), StubMM()
    store = Store(cfg.db_path)
    build_index(HK, cfg, text, mm, store, fake_renderer)
    for s in default_settings(True):
        r = evaluate_setting(store, cfg, QA, s, text, mm, k=3)
        assert 0 <= r.hit1 <= r.mrr <= r.hitk <= 1
        assert set(r.hitk_by_kind) == {"text", "visual"}
    table = to_markdown([evaluate_setting(store, cfg, QA, Setting("text", "text"), text, mm)], 3, len(QA))
    assert "| text |" in table and "hit@3 visual" in table
