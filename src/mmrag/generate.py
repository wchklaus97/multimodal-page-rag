"""Stage 6 - Generation (context = the fused PAGES, text only; a vision LLM could also be sent the page PNGs).

Hand the retrieved chunks to an LLM inside a grounded prompt. With no LLM
configured, `generate` returns None and the CLI prints retrieval-only output.

Provider choice (llm_provider = "auto"): OPENAI_API_KEY -> openai,
GEMINI_API_KEY -> gemini, OLLAMA_MODEL -> ollama, otherwise none.
"""

from __future__ import annotations

import os

from ._http import post_json

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages. "
    "Cite passages like [1] or [2]. If the context does not contain the answer, "
    "say you don't know instead of guessing. Always reply in the same language as the Question, "
    "even when the context passages are in another language."
)


def format_context(hits) -> str:
    """Each hit is a whole PAGE (a fused page or a Hit): label it with org + location."""
    parts = []
    for n, h in enumerate(hits, start=1):
        org = "/".join(x for x in (getattr(h, "entity", ""), getattr(h, "department", "")) if x)
        where = f"{h.source} p.{h.page}" + (f", {org}" if org else "")
        body = h.text or "[page has no extractable text - image only]"
        parts.append(f"[{n}] ({where})\n{body}")
    return "\n\n".join(parts)


def build_prompt(question: str, hits) -> str:
    return f"Context:\n{format_context(hits)}\n\nQuestion: {question}"


def resolve_provider(provider: str) -> str:
    if provider != "auto":
        return provider
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OLLAMA_MODEL"):
        return "ollama"
    return "none"


def generate(question: str, hits, provider: str = "auto", model: str = "") -> str | None:
    provider = resolve_provider(provider)
    if provider == "none" or not hits:
        return None
    prompt = build_prompt(question, hits)

    if provider == "openai":
        url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
        data = post_json(
            url,
            {
                "model": model or "gpt-4o-mini",
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            },
            {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        )
        return data["choices"][0]["message"]["content"]

    if provider == "gemini":
        m = model or "gemini-2.5-flash"
        data = post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            },
            {"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        )
        return data["candidates"][0]["content"]["parts"][0]["text"]

    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        data = post_json(
            f"{host}/api/chat",
            {
                "model": model or os.environ["OLLAMA_MODEL"],
                "stream": False,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            },
        )
        return data["message"]["content"]

    raise ValueError(f"unknown llm_provider {provider!r} (auto | openai | gemini | ollama | none)")
