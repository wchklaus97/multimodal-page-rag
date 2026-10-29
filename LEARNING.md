# LEARNING — 頁級多模態 RAG / Page-level multimodal RAG

情境：虛構嘅 Harbour Lane 集團（香港總部 + 深圳子公司），文件放喺 `公司實體/部門/檔案.pdf`。
Scenario: a fictional Hong Kong group with a Shenzhen subsidiary; files live in `entity/department/file.pdf`.

---

## 繁體中文（zh-HK）

### 1. 「切邊個單位」要喺 embedding 之前決定
你有一份有文字同圖表嘅 PDF。第一個問題唔係「用邊個 embedding」，而係「**一個可被檢索嘅單位係咩**」：一頁？一個章節？一張圖加圖註？
呢個決定喺 **ingest** 做（`ingest.py`）。Embedding API 淨係收你俾佢嗰段文字或者嗰張圖，傳返一個向量；**佢唔會幫你切頁、切段**。本 repo 揀「一頁一個單位」，每頁有一個 `page_id`。

### 2. 同一頁，兩種向量
- 文字：頁面文字 → 文字模型 → `vec_text`
- 圖像：頁面截圖 → 多模態模型（CLIP／Vertex）→ `vec_image`

兩個模型嘅向量空間同維度都唔同，**唔可以放入同一個索引、更加唔可以直接比較**。所以分開兩個索引，用同一個 `page_id` 連返埋一齊。

### 3. Late fusion（後期融合）
先**各自檢索**，再用 `page_id` **合併排名**。
- 為何唔直接加分數？文字對文字嘅 cosine 可以 0.6，CLIP 文字對圖像嘅 cosine 好似 0.3 已經好準。直接加，文字永遠壓過圖像。
- `rrf`：只用排名（1/(60+rank)），唔受分數尺度影響，係預設。
- `minmax`：每個清單先縮放到 0–1，再加權。保留分數差距，但一個極端值會壓扁其他。
- 對比 early fusion：先將兩種向量混合再搜尋，需要一個同時理解兩種模態嘅模型。

### 3b. 真 CLIP 實測：RRF 會獎勵雜訊
用真 CLIP 問「a bar chart」：image 模式正確將圖表頁排第一（cos 0.266）。但 fusion 模式反而將深圳供應商頁排第一，因為佢嘅文字分數只係 0.086（雜訊），但喺文字清單排第 1，RRF 只睇排名就照計分。設定 `text_min_score = 0.1` 先過濾雜訊，圖表頁就返到第一。教訓：**融合之前，每個模態要有自己嘅最低分數門檻。**

### 4. 公司結構變成 metadata
資料夾 `hk-hq/finance/` 自動變成 `entity=hk-hq`、`department=finance`。用 `--dept finance` 過濾。
**陷阱**：過濾喺向量搜尋**之後**做，若部門好冷門，top-k 可能得返少過 k 個結果。所以過濾時會將候選池放大 5 倍。

### 常見錯誤 / 失敗模式
- 掃描版 PDF 冇文字層、冇 OCR：純文字索引會直接丟棄；多模態索引保留，但只有圖像向量搵得到佢，細字 CLIP 睇唔清。
- 以為 embedding API 會切頁（唔會）。
- 一頁太長，超過文字模型 token 上限被截斷：考慮用章節做單位。
- 分數尺度唔同仲直接相加。
- 換咗模型但冇重建索引（本 repo 會拒絕）。
- 本示範嘅 PDF 用內建 Helvetica，只含英文；中文 PDF 需要內嵌 CJK 字型，真實文件就冇問題。

### 練習
1. 用 `--mode text`、`image`、`fusion` 問「quarterly expense claims chart」，比較結果同每個模態嘅 rank／cosine。
2. 將 `fusion_method` 改為 `minmax`，再將 `image_weight` 調到 3，睇排名點變。
3. 試 `--entity sz-subsidiary`，點解結果可能少過 k？

---

## English

### 1. Choose the cut unit before embedding
Given a PDF with text and figures, the first question isn't "which embedding model?" but "**what is one retrievable unit?**" A page, a section, an image with its caption? That decision lives in **ingest** (`ingest.py`). An embedding API takes the string or image you give it and returns one vector; **it never splits pages or text for you**. This repo uses one page per unit, each with a `page_id`.

### 2. Two vectors per page
- Text: page text → text model → `vec_text`
- Image: page render → multimodal model (CLIP / Vertex) → `vec_image`

Different models mean different vector spaces and dimensions, so they cannot share an index or be compared directly. Two indexes, joined by `page_id`.

### 3. Late fusion
Retrieve **separately**, then **merge rankings by `page_id`**.
- Why not add raw scores? Text→text cosine may be 0.6 while a great CLIP text→image match is ~0.3. Summed, text always wins.
- `rrf` uses ranks only (1/(60+rank)); immune to scale. Default.
- `minmax` rescales each list to 0–1 then takes a weighted sum. Keeps score gaps, but an outlier compresses the rest.
- Contrast with early fusion: mix modalities in one vector before search; needs one jointly trained model.

### 3b. Real CLIP run: RRF rewards noise
With real CLIP, "a bar chart" in image mode correctly ranks the chart page first (cos 0.266). In fusion mode the Shenzhen supplier page came first: its text score was only 0.086 (noise) but it was rank 1 in the text list, and RRF only sees ranks. Setting `text_min_score = 0.1` removes the noise and the chart page returns to #1. Lesson: **give each modality its own score floor before fusing.**

### 4. Company structure as metadata
`hk-hq/finance/` becomes `entity=hk-hq`, `department=finance`; filter with `--dept finance`.
**Pitfall:** the filter runs *after* vector search, so a rare department can leave fewer than k results. The candidate pool is widened ×5 when filtering.

### Common mistakes / failure modes
- Scanned PDFs with no text layer and no OCR: dropped by the text index; kept by the multimodal index but only findable through the image vector, and CLIP reads small print poorly.
- Believing an embedding API chunks pages (it doesn't).
- Pages longer than the text model's token limit get truncated; consider sections as the unit.
- Adding scores from different scales.
- Changing models without rebuilding (this repo refuses).
- The sample PDFs use built-in Helvetica (Latin text only); real Chinese PDFs embed CJK fonts, which is fine for ingestion.

### Exercises
1. Ask "quarterly expense claims chart" with `--mode text`, `image`, `fusion`; compare per-modality rank and cosine.
2. Switch `fusion_method` to `minmax` and set `image_weight = 3`; watch the ranking move.
3. Try `--entity sz-subsidiary`: why might you get fewer than k pages?
