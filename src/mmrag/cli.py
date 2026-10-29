"""mmrag ingest <folder> | query "<q>" --mode text|image|fusion | stats | reset"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .embed import get_multimodal_embedder, get_text_embedder
from .generate import generate, resolve_provider
from .index import build_index, pages_dir_for
from .retrieve import MODES, retrieve
from .store import Store, reset_db


def cmd_ingest(args, cfg) -> int:
    text_emb = get_text_embedder(cfg.text_provider, cfg.text_model)
    mm_emb = renderer = None
    if cfg.index_mode == "multimodal":
        mm_emb = get_multimodal_embedder(cfg.image_provider, cfg.image_model)
        if mm_emb is not None:
            try:
                from .ingest import pdfium_renderer

                renderer = pdfium_renderer()
            except ImportError:
                print("note: pypdfium2 not installed - cannot render page images. pip install -e '.[images]'", file=sys.stderr)
    store = Store(cfg.db_path)
    n = build_index(args.folder, cfg, text_emb, mm_emb, store, renderer, log=print)
    print(f"Done: {n} pages processed, {store.count()} total in {cfg.db_path}")
    return 0


def cmd_query(args, cfg) -> int:
    store = Store(cfg.db_path)
    text_emb = get_text_embedder(cfg.text_provider, cfg.text_model) if args.mode in {"text", "fusion"} else None
    mm_emb = None
    if args.mode in {"image", "fusion"} and store.has_index("image"):
        mm_emb = get_multimodal_embedder(cfg.image_provider, cfg.image_model)
    if args.k:
        cfg.top_k = args.k
    pages, notes = retrieve(store, cfg, args.question, args.mode, text_emb, mm_emb, args.dept, args.entity)
    for note in notes:
        print(f"note: {note}")
    if not pages:
        print("No matching pages.")
        return 1

    print(f"mode={args.mode}  fusion={cfg.fusion_method if args.mode == 'fusion' else '-'}  pages used:\n")
    for n, p in enumerate(pages, start=1):
        org = "/".join(x for x in (p.entity, p.department) if x) or "-"
        parts = "  ".join(f"{m}: rank {r} cos {s:.3f}" for m, (r, s) in p.parts.items())
        print(f"[{n}] {p.page_id}  org={org}  score={p.score:.4f}")
        print(f"    {parts}")
        snippet = " ".join(p.text.split())
        print(f"    {snippet[:220] + ('...' if len(snippet) > 220 else '') if snippet else '(no extractable text - image only)'}")
        if p.image_path:
            print(f"    image: {p.image_path}")
        print()

    answer = generate(args.question, pages, cfg.llm_provider, cfg.llm_model)
    if answer is None:
        print("-- retrieval-only mode: no LLM configured (OPENAI_API_KEY / GEMINI_API_KEY / OLLAMA_MODEL) --")
    else:
        print(f"Answer ({resolve_provider(cfg.llm_provider)}):\n{answer}")
    return 0


def cmd_stats(_args, cfg) -> int:
    store = Store(cfg.db_path)
    print(f"db: {cfg.db_path}  pages: {store.count()}")
    for m in ("text", "image"):
        print(f"  {m} index: {store.index_info(m) or '(none)'}")
    for entity, dept, n in store.org_summary():
        print(f"  {n:3d} pages  entity={entity or '-'}  dept={dept or '-'}")
    return 0


def cmd_reset(args, cfg) -> int:
    print("Deleted index" if reset_db(cfg.db_path, pages_dir_for(cfg.db_path)) else "Nothing to delete")
    if args.rebuild:
        args.folder = args.rebuild
        return cmd_ingest(args, cfg)
    return 0


def cmd_eval(args, cfg) -> int:
    import tempfile
    from dataclasses import replace

    from .evaluate import default_settings, evaluate_setting, load_qa, to_markdown
    from .ingest import pdfium_renderer

    qa = load_qa(args.qa)
    text_emb = get_text_embedder(args.text, "")
    mm_emb = get_multimodal_embedder(cfg.image_provider, cfg.image_model)
    renderer = None
    if mm_emb is not None:
        try:
            renderer = pdfium_renderer()
        except ImportError:
            mm_emb = None
    with tempfile.TemporaryDirectory() as tmp:
        run_cfg = replace(cfg, db_path=str(Path(tmp) / "eval.db"), index_mode="multimodal" if mm_emb else "text")
        store = Store(run_cfg.db_path)
        build_index(args.folder, run_cfg, text_emb, mm_emb, store, renderer)
        settings = default_settings(store.has_index("image"), args.floor)
        results = [evaluate_setting(store, run_cfg, qa, s, text_emb, mm_emb, args.k) for s in settings]
        store.close()
    header = f"text: `{text_emb.name}` · image: `{mm_emb.name if mm_emb else 'none (image modes skipped)'}`"
    table = to_markdown(results, args.k, len(qa), header)
    print(table)
    if args.show_misses:
        for r in results:
            print(f"misses [{r.setting}]: {', '.join(r.misses) or '-'}")
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mmrag", description=__doc__)
    p.add_argument("--config")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="ingest a folder of PDFs (folder layout: <entity>/<department>/file.pdf)")
    s.add_argument("folder")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("query", help="ask a question")
    s.add_argument("question")
    s.add_argument("--mode", choices=MODES, default="text")
    s.add_argument("-k", type=int)
    s.add_argument("--dept", help="only pages from this department folder, e.g. hr")
    s.add_argument("--entity", help="only pages from this entity folder, e.g. hk-hq")
    s.set_defaults(fn=cmd_query)

    s = sub.add_parser("eval", help="score text / image / fusion retrieval on a page-level QA set")
    s.add_argument("--qa", default="eval/harbour_lane_pages_qa.jsonl")
    s.add_argument("--folder", default="examples/harbour-lane")
    s.add_argument("--text", default="hash", help="text embedder: hash | local")
    s.add_argument("--floor", type=float, default=0.1, help="text score floor tried in fusion rows")
    s.add_argument("-k", type=int, default=3)
    s.add_argument("--out")
    s.add_argument("--show-misses", action="store_true")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("stats")
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("reset", help="delete the index (and page images); --rebuild FOLDER re-ingests")
    s.add_argument("--rebuild", metavar="FOLDER")
    s.set_defaults(fn=cmd_reset)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    try:
        return args.fn(args, cfg)
    except (RuntimeError, LookupError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
