"""Configuration: one dataclass from mmrag.toml, overridden by env vars.

Nothing here is passed to an embedding API. The page split (the retrieval unit)
is decided by ingest.py; embedders only turn the units they are given into vectors.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    db_path: str = "mmrag.db"
    index_mode: str = "text"  # text | multimodal
    text_provider: str = "local"
    text_model: str = ""
    image_provider: str = "clip"
    image_model: str = ""
    top_k: int = 3
    fusion_method: str = "rrf"
    fusion_pool: int = 10
    text_weight: float = 1.0
    image_weight: float = 1.0
    text_min_score: float = 0.0  # drop hits at/below this cosine before fusion (RRF ignores scores)
    image_min_score: float = 0.0
    llm_provider: str = "auto"
    llm_model: str = ""


_ENV = {
    "MMRAG_DB": "db_path",
    "INDEX_MODE": "index_mode",
    "TEXT_PROVIDER": "text_provider",
    "TEXT_MODEL": "text_model",
    "IMAGE_PROVIDER": "image_provider",
    "IMAGE_MODEL": "image_model",
    "LLM_PROVIDER": "llm_provider",
    "LLM_MODEL": "llm_model",
    "TEXT_MIN_SCORE": "text_min_score",
    "IMAGE_MIN_SCORE": "image_min_score",
    "FUSION_METHOD": "fusion_method",
}


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("MMRAG_CONFIG", "mmrag.toml"))
    values = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in values.items() if k in known})
    for env, attr in _ENV.items():
        if os.environ.get(env):
            setattr(cfg, attr, type(getattr(cfg, attr))(os.environ[env]))
    if cfg.index_mode not in {"text", "multimodal"}:
        raise ValueError(f"index_mode must be 'text' or 'multimodal', got {cfg.index_mode!r}")
    return cfg
