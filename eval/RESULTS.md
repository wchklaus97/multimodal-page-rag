# Page retrieval evaluation - Harbour Lane PDFs

## Weak text embedder

text: `hash:384` · image: `clip:clip-ViT-B-32`

18 questions, top-k = 3.

| setting | hit@1 | hit@3 | MRR@3 | hit@3 text | hit@3 visual |
|---|---|---|---|---|---|
| text | 0.44 | 0.44 | 0.44 | 0.62 | 0.00 |
| image | 0.33 | 0.61 | 0.44 | 0.62 | 0.60 |
| fusion rrf | 0.22 | 0.72 | 0.43 | 0.85 | 0.40 |
| fusion rrf, text floor 0.1 | 0.28 | 0.72 | 0.47 | 0.85 | 0.40 |
| fusion minmax | 0.44 | 0.83 | 0.60 | 0.92 | 0.60 |
| fusion minmax, text floor 0.1 | 0.44 | 0.72 | 0.57 | 0.77 | 0.60 |

## Strong (multilingual) text embedder

text: `local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` · image: `clip:clip-ViT-B-32`

18 questions, top-k = 3.

| setting | hit@1 | hit@3 | MRR@3 | hit@3 text | hit@3 visual |
|---|---|---|---|---|---|
| text | 0.89 | 1.00 | 0.94 | 1.00 | 1.00 |
| image | 0.33 | 0.61 | 0.44 | 0.62 | 0.60 |
| fusion rrf | 0.56 | 1.00 | 0.75 | 1.00 | 1.00 |
| fusion rrf, text floor 0.1 | 0.56 | 0.94 | 0.72 | 1.00 | 0.80 |
| fusion minmax | 0.72 | 0.94 | 0.82 | 1.00 | 0.80 |
| fusion minmax, text floor 0.1 | 0.72 | 0.94 | 0.83 | 1.00 | 0.80 |

## What the numbers say

1. **Fusion pays off when the text side is weak.** With the keyword `hash` text embedder, text-only retrieval finds **none** of the visual questions (0.00); CLIP image retrieval finds 0.60, and `fusion minmax` is best overall (hit@3 0.83, MRR 0.60).
2. **With a strong text model, adding images made results worse.** Multilingual text alone scores hit@3 1.00 / MRR 0.94 - *including* the visual questions - because the chart page carries text labels ("Figure 1 Quarterly expense claims by department"). Adding CLIP drops hit@1 from 0.89 to 0.56 (RRF) / 0.72 (min-max): a weaker modality adds noise. Image retrieval earns its place on pages **without** usable text (scans, photos, unlabelled diagrams), which this corpus barely covers.
3. **CLIP ViT-B/32 is English-only.** Image mode misses Chinese queries (e.g. `chart-zh`, `leave-days-zh`). A multilingual CLIP text encoder (e.g. sentence-transformers `clip-ViT-B-32-multilingual-v1`) is the obvious next experiment - untested here.
4. **The text score floor didn't generalise.** It was added after ONE query ("a bar chart") where RRF ranked noise first. Across all 18 questions the 0.1 floor is neutral or worse (hash min-max 0.83 → 0.72; multilingual RRF 1.00 → 0.94), so the default stays 0.0. Don't tune on an anecdote - that is what this eval is for.
5. **Small set:** 18 questions (one = 0.056); only 5 are visual (one = 0.20). Read differences accordingly.

Reproduce: `uv sync --extra images && uv run mmrag eval --text hash --show-misses` and `--text local`.
