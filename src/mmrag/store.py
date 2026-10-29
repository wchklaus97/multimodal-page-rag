"""Stage 4 - Vector store: one SQLite file + sqlite-vec.

Choice: SQLite + sqlite-vec (not Chroma): a single portable file, SQL for metadata,
and no server. Trade-off: no built-in hybrid search; fusion is done in fusion.py.

Tables
  meta(key, value)
  pages(id, page_id UNIQUE, source, page, text, image_path)   one row per page unit
  vec_text   vec0 (rowid = pages.id)    text embeddings, dimension of the text model
  vec_image  vec0 (rowid = pages.id)    image embeddings, dimension of the image model

Two vec tables, not one, because the modalities have different dimensions and live in
different vector spaces. `modality` on every Hit says which table it came from.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

MODALITIES = ("text", "image")


@dataclass(frozen=True)
class Hit:
    page_id: str
    source: str
    page: int
    text: str
    image_path: str | None
    modality: str  # "text" | "image"
    score: float  # cosine similarity within that modality only
    entity: str = ""
    department: str = ""


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS pages(
                id INTEGER PRIMARY KEY, page_id TEXT UNIQUE NOT NULL, source TEXT NOT NULL,
                page INTEGER NOT NULL, text TEXT NOT NULL, image_path TEXT,
                entity TEXT NOT NULL DEFAULT '', department TEXT NOT NULL DEFAULT '');
            """
        )

    # --- per-modality index, created on first use ---------------------------
    def ensure_index(self, modality: str, embedder_name: str, dim: int) -> None:
        sig = f"{embedder_name}|{dim}"
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (modality,)).fetchone()
        if row and row[0] != sig:
            raise RuntimeError(
                f"{modality} index was built with {row[0]!r}, you are using {sig!r}. "
                "Vectors from different models can't be compared; run `mmrag reset` and re-ingest."
            )
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?, ?)", (modality, sig))
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_{modality} USING vec0(embedding float[{dim}] distance_metric=cosine)"
        )
        self.db.commit()

    def has_index(self, modality: str) -> bool:
        return self.db.execute("SELECT 1 FROM meta WHERE key=?", (modality,)).fetchone() is not None

    def index_info(self, modality: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (modality,)).fetchone()
        return row[0] if row else None

    # --- write ---------------------------------------------------------------
    def delete_page(self, page_id: str) -> None:
        row = self.db.execute("SELECT id FROM pages WHERE page_id=?", (page_id,)).fetchone()
        if row:
            for m in MODALITIES:
                if self.has_index(m):
                    self.db.execute(f"DELETE FROM vec_{m} WHERE rowid=?", (row[0],))
            self.db.execute("DELETE FROM pages WHERE id=?", (row[0],))

    def add_page(self, page, text_vec: list[float] | None, image_vec: list[float] | None, image_path: str | None) -> None:
        """Insert one page unit; its text and image vectors share the same rowid / page_id."""
        self.delete_page(page.page_id)
        cur = self.db.execute(
            "INSERT INTO pages(page_id, source, page, text, image_path, entity, department) VALUES(?,?,?,?,?,?,?)",
            (page.page_id, page.source, page.page, page.text, image_path, page.entity, page.department),
        )
        for modality, vec in (("text", text_vec), ("image", image_vec)):
            if vec is not None:
                self.db.execute(f"INSERT INTO vec_{modality}(rowid, embedding) VALUES(?,?)", (cur.lastrowid, _pack(vec)))
        self.db.commit()

    # --- read ----------------------------------------------------------------
    def search(self, modality: str, vector: list[float], k: int) -> list[Hit]:
        if not self.has_index(modality):
            raise LookupError(f"no {modality} index in {self.path}")
        rows = self.db.execute(
            f"SELECT rowid, distance FROM vec_{modality} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_pack(vector), k),
        ).fetchall()
        hits = []
        for rid, dist in rows:
            page_id, source, page, text, image_path, entity, dept = self.db.execute(
                "SELECT page_id, source, page, text, image_path, entity, department FROM pages WHERE id=?", (rid,)
            ).fetchone()
            hits.append(Hit(page_id, source, page, text, image_path, modality, 1.0 - dist, entity, dept))
        return hits

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

    def org_summary(self) -> list[tuple[str, str, int]]:
        return self.db.execute(
            "SELECT entity, department, COUNT(*) FROM pages GROUP BY entity, department ORDER BY entity, department"
        ).fetchall()

    def close(self) -> None:
        self.db.close()


def reset_db(path: str | Path, pages_dir: str | Path | None = None) -> bool:
    removed = False
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()
            removed = True
    if pages_dir and Path(pages_dir).is_dir():
        for png in Path(pages_dir).glob("*.png"):
            png.unlink()
    return removed
