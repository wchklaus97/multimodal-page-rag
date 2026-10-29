"""Stage 3 - Embedding. Two kinds of embedder.

TextEmbedder       embed(list[str]) -> vectors                      (text space)
MultimodalEmbedder embed_images(list[bytes]) and embed_texts(list[str]) -> vectors
                   in ONE shared space, so a text query can be compared to page images.

Each embedder has its own vector space and dimension. A text-model vector and a
CLIP vector are NOT comparable, which is why the store keeps one index per modality
and fusion happens after retrieval ("late"), not by mixing vectors.

Offline default: text via local sentence-transformers (or the dependency-free
`hash` fallback); image embedding is skipped with a clear message.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import re
import subprocess
import sys

from ._http import post_json

DEFAULT_TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CLIP_MODEL = "clip-ViT-B-32"


class TextEmbedder:
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        if not hasattr(self, "_dim"):
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim


class HashEmbedder(TextEmbedder):
    """Hashed bag of words (+ CJK bigrams). Not semantic; for tests and offline demos."""

    _TOKEN = re.compile(r"[a-z0-9_]+|[一-鿿]")

    def __init__(self, dim: int = 384):
        self._dim = dim
        self.name = f"hash:{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        is_cjk = lambda t: "一" <= t <= "鿿"  # noqa: E731
        out = []
        for text in texts:
            toks = self._TOKEN.findall(text.lower())
            feats = toks + [a + b for a, b in zip(toks, toks[1:]) if is_cjk(a) and is_cjk(b)]
            vec = [0.0] * self._dim
            for f in feats:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
                vec[h % self._dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class SentenceTransformerEmbedder(TextEmbedder):
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.name = f"local:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def get_text_embedder(provider: str, model: str = "") -> TextEmbedder:
    if provider == "hash":
        return HashEmbedder()
    if provider == "local":
        try:
            return SentenceTransformerEmbedder(model or DEFAULT_TEXT_MODEL)
        except ImportError:
            print(
                "warning: sentence-transformers not installed; using the 'hash' text embedder "
                "(keyword-ish, not semantic). For real embeddings: pip install -e '.[local]'",
                file=sys.stderr,
            )
            return HashEmbedder()
    raise ValueError(f"unknown text_provider {provider!r} (local | hash)")


class MultimodalEmbedder:
    name: str

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        if not hasattr(self, "_dim"):
            self._dim = len(self.embed_texts(["dimension probe"])[0])
        return self._dim


class ClipEmbedder(MultimodalEmbedder):
    """CLIP via sentence-transformers: images and text land in one 512-d space."""

    def __init__(self, model: str):
        from PIL import Image  # noqa: F401  (fail early if pillow is missing)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.name = f"clip:{model}"

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        import io

        from PIL import Image

        pics = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        return self._model.encode(pics, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class VertexEmbedder(MultimodalEmbedder):
    """Vertex AI multimodalembedding@001 (1408-d, text+image in one space). Optional, untested in CI.

    Needs GOOGLE_CLOUD_PROJECT (+ optional GOOGLE_CLOUD_LOCATION) and a token from
    VERTEX_ACCESS_TOKEN or `gcloud auth print-access-token`.
    """

    def __init__(self, model: str):
        self._model = model
        self._project = os.environ["GOOGLE_CLOUD_PROJECT"]
        self._loc = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.name = f"vertex:{model}"

    def _token(self) -> str:
        return os.environ.get("VERTEX_ACCESS_TOKEN") or subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], text=True
        ).strip()

    def _predict(self, instance: dict) -> dict:
        url = (
            f"https://{self._loc}-aiplatform.googleapis.com/v1/projects/{self._project}"
            f"/locations/{self._loc}/publishers/google/models/{self._model}:predict"
        )
        return post_json(url, {"instances": [instance]}, {"Authorization": f"Bearer {self._token()}"})["predictions"][0]

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        return [
            self._predict({"image": {"bytesBase64Encoded": base64.b64encode(b).decode()}})["imageEmbedding"]
            for b in images
        ]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._predict({"text": t})["textEmbedding"] for t in texts]


def get_multimodal_embedder(provider: str, model: str = "") -> MultimodalEmbedder | None:
    """Returns None (with an explanation) when image embedding is unavailable."""
    if provider == "none":
        return None
    try:
        if provider == "clip":
            return ClipEmbedder(model or DEFAULT_CLIP_MODEL)
        if provider == "vertex":
            return VertexEmbedder(model or "multimodalembedding@001")
    except ImportError as e:
        print(f"note: image embedding skipped - {e.name} not installed. Install with: pip install -e '.[images]'", file=sys.stderr)
        return None
    except KeyError as e:
        print(f"note: image embedding skipped - env var {e.args[0]} is not set.", file=sys.stderr)
        return None
    raise ValueError(f"unknown image_provider {provider!r} (clip | vertex | none)")
