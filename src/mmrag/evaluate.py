"""Evaluation: which retrieval mode / fusion setting finds the right PAGE?

The unit here is the page, so a hit = the retrieved page_id equals the expected page_id.
Questions are tagged `kind`: "text" (answer is in the page text) or "visual" (described
by what the page LOOKS like, e.g. "a bar chart") - that split is the whole point of
multimodal retrieval, so results are reported per kind as well as overall.

Settings compared (one index, many retrieval configs):
  text                      text index only
  image                     image index only (needs a multimodal embedder)
  fusion rrf / minmax       late fusion, with and without a text score floor
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from .retrieve import retrieve


@dataclass(frozen=True)
class PageQA:
    id: str
    lang: str
    kind: str  # "text" | "visual"
    q: str
    page_id: str


@dataclass
class Setting:
    label: str
    mode: str
    fusion_method: str = "rrf"
    text_min_score: float = 0.0


@dataclass
class Result:
    setting: str
    hit1: float
    hitk: float
    mrr: float
    hitk_by_kind: dict[str, float] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)


def load_qa(path: str | Path) -> list[PageQA]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [PageQA(**json.loads(line)) for line in lines if line.strip()]


def default_settings(has_image: bool, floor: float = 0.1) -> list[Setting]:
    settings = [Setting("text", "text")]
    if has_image:
        settings += [
            Setting("image", "image"),
            Setting("fusion rrf", "fusion", "rrf"),
            Setting(f"fusion rrf, text floor {floor}", "fusion", "rrf", floor),
            Setting("fusion minmax", "fusion", "minmax"),
            Setting(f"fusion minmax, text floor {floor}", "fusion", "minmax", floor),
        ]
    return settings


def evaluate_setting(store, cfg, qa: list[PageQA], setting: Setting, text_emb, mm_emb, k: int = 3) -> Result:
    run_cfg = replace(cfg, top_k=k, fusion_method=setting.fusion_method, text_min_score=setting.text_min_score)
    ranks: dict[str, int | None] = {}
    for item in qa:
        pages, _ = retrieve(store, run_cfg, item.q, setting.mode, text_emb, mm_emb)
        ids = [p.page_id for p in pages]
        ranks[item.id] = ids.index(item.page_id) + 1 if item.page_id in ids else None

    def share(items) -> float:
        items = list(items)
        return sum(1 for i in items if ranks[i.id]) / len(items) if items else 0.0

    return Result(
        setting=setting.label,
        hit1=sum(1 for i in qa if ranks[i.id] == 1) / len(qa),
        hitk=share(qa),
        mrr=sum(1 / ranks[i.id] for i in qa if ranks[i.id]) / len(qa),
        hitk_by_kind={kind: share(i for i in qa if i.kind == kind) for kind in sorted({i.kind for i in qa})},
        misses=[i.id for i in qa if not ranks[i.id]],
    )


def to_markdown(results: list[Result], k: int, n: int, header: str = "") -> str:
    kinds = sorted({kind for r in results for kind in r.hitk_by_kind})
    head = f"| setting | hit@1 | hit@{k} | MRR@{k} | " + " | ".join(f"hit@{k} {kd}" for kd in kinds) + " |"
    sep = "|" + "---|" * (4 + len(kinds))
    rows = [
        f"| {r.setting} | {r.hit1:.2f} | {r.hitk:.2f} | {r.mrr:.2f} | "
        + " | ".join(f"{r.hitk_by_kind.get(kd, 0):.2f}" for kd in kinds)
        + " |"
        for r in results
    ]
    return (header + "\n\n" if header else "") + f"{n} questions, top-k = {k}.\n\n" + "\n".join([head, sep, *rows]) + "\n"
