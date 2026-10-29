import pytest

from mmrag.fusion import fuse

TEXT = [("a", 0.80), ("b", 0.78), ("c", 0.40)]
IMAGE = [("c", 0.31), ("d", 0.30), ("a", 0.12)]  # CLIP-like scores live on a much lower scale


def test_rrf_page_found_by_both_modalities_wins():
    out = fuse({"text": TEXT, "image": IMAGE}, "rrf")
    assert out[0].key in {"a", "c"}  # both appear in both lists
    assert {f.key for f in out} == {"a", "b", "c", "d"}
    assert out[0].parts.keys() == {"text", "image"}


def test_raw_score_sum_would_be_dominated_by_text_but_minmax_is_not():
    raw = {k: sum(s for lst in (TEXT, IMAGE) for kk, s in lst if kk == k) for k in "abcd"}
    assert max(raw, key=raw.get) == "a"  # raw sum: text scale dominates, 'd' (image-only) is last
    mm = {f.key: f.score for f in fuse({"text": TEXT, "image": IMAGE}, "minmax")}
    assert mm["d"] > 0.5 * mm["b"]  # after normalisation the image-only hit is competitive


def test_weights_shift_ranking():
    heavy_image = fuse({"text": TEXT, "image": IMAGE}, "rrf", {"text": 0.1, "image": 1.0})
    assert heavy_image[0].key == "c"  # image rank 1


def test_single_list_and_empty():
    assert [f.key for f in fuse({"text": TEXT}, "rrf")] == ["a", "b", "c"]
    assert fuse({}, "rrf") == []
    assert fuse({"text": []}, "minmax") == []


def test_equal_scores_do_not_divide_by_zero():
    assert fuse({"text": [("a", 0.5), ("b", 0.5)]}, "minmax")[0].score == 1.0


def test_bad_method():
    with pytest.raises(ValueError):
        fuse({"text": TEXT}, "nope")
