# Hams.AI — Self-Hosted Arabic & English Hybrid RAG for an Enterprise Knowledge Bank

A fully self-hosted, bilingual (Modern Standard Arabic + English, incl. mixed and cross-lingual)
retrieval-augmented search system with three modes behind one API:

| Mode | What it does | Models involved |
|---|---|---|
| **Quick Search** | BM25 keyword search with Arabic/English analysis, ranked snippets | none (CPU only) |
| **Smart AI Search** | Hybrid retrieval (BM25 + fine-tuned dense) → RRF → fine-tuned cross-encoder → grounded answer with citations, or an explicit "not found" | fine-tuned embedder, fine-tuned reranker, local LLM |
| **Interactive AI Search** | Multi-turn chat with session memory, follow-up rewriting, language switching, suggested follow-ups | same backbone + query rewriting |

No closed or paid API is used anywhere at runtime. Everything runs on one machine.

> **Hardware note.** All numbers in this README were measured on the development laptop:
> NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), Intel i7-11800H (8C/16T), 16 GB RAM, Windows 11.
> The assessment's reference GPU (NVIDIA L4, 24 GB) was **not** available; L4 figures are not claimed.
> The whole stack uses < 4 GB of VRAM, i.e. well inside the L4 limit.

**Fine-tuned models on Hugging Face**
* Embedding model: [AbdelrahmanHassan/hams-e5-base-ar-en-retrieval](https://huggingface.co/AbdelrahmanHassan/hams-e5-base-ar-en-retrieval) (from [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base))
* Reranker: [AbdelrahmanHassan/hams-gte-reranker-ar-en](https://huggingface.co/AbdelrahmanHassan/hams-gte-reranker-ar-en) (from [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base))
* LLM (not fine-tuned): [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), GGUF from [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)

---

## 1. Architecture

```
                         ┌──────────────────────────── ingestion (scripts/ingest.py, POST /ingest) ─────────────────────────────┐
 PDF / DOCX / HTML / MD / TXT ─► loaders ─► NFKC + Arabic/English cleaning ─► structure-aware chunking (≈120 words, heading-aware)
                                   │  PDF: pdfplumber glyphs → visual→logical bidi repair, font-size headings, ruled tables
                                   │  tables → "header: value; header: value" rows
                                   ├─► SQLite FTS5 (analysed tokens: normalised + light-stemmed)          ── Quick Search
                                   └─► fine-tuned multilingual-e5-base → FAISS (inner product, L2-norm)   ── dense
 query ─► language detection ─► ┬─ BM25 top-50 ─────────┐
                                └─ dense top-50 ────────┴─► weighted RRF ─► top-20 ─► fine-tuned cross-encoder ─► abstain gate
                                                                                    │
                                     ┌──────────────────────────────────────────────┘
                                     ▼
                       top-4 chunks as numbered sources ─► llama.cpp (Qwen3.5, GGUF Q4_K_M, streaming)
                                     ─► answer in the question's language with [n] citations ─► citation validation ─► JSON / SSE
 Interactive: SQLite session memory ─► (gated) LLM query rewrite ─► same backbone ─► answer + suggested follow-ups (same LLM call)
```

One `RetrievalEngine` (`hamsrag/retrieval.py`) implements `lexical()`, `hybrid()` and `rerank()`; one
`SearchService` (`hamsrag/rag.py`) composes them into the three modes, so there is no duplicated pipeline.

### Component choices

| Component | Choice | Why | Rejected alternatives |
|---|---|---|---|
| Keyword engine | **SQLite FTS5** (`bm25()`), own analyser | Real inverted index, persistent, incremental inserts, zero extra service, ships with Python on Windows | rank_bm25 (pure Python, O(N) per query), Whoosh (unmaintained), Elasticsearch/OpenSearch (JVM service, heavy for 4 GB / Windows) |
| Vector store | **FAISS** (`IndexFlatIP`, HNSW above 50k vectors) | In-process, ms-level search at this scale, Windows wheels | Qdrant/Milvus (extra server), Chroma (heavier, no benefit here) |
| Embedder | **intfloat/multilingual-e5-base**, fine-tuned ([weights](https://huggingface.co/AbdelrahmanHassan/hams-e5-base-ar-en-retrieval)) | Strong multilingual/cross-lingual base, MIT, standard XLM-R (no custom code), trainable on 4 GB with frozen word embeddings | bge-m3 (568M: cannot be fully fine-tuned on 4 GB), jina-v3 (CC-BY-NC), Qwen3-Embedding-0.6B (LoRA-only on 4 GB) |
| Reranker | **Alibaba-NLP/gte-multilingual-reranker-base**, fine-tuned ([weights](https://huggingface.co/AbdelrahmanHassan/hams-gte-reranker-ar-en)) | 306M, Apache-2.0, multilingual incl. Arabic, trainable on 4 GB; strongest zero-shot of the candidates we could train (validation MRR@10 0.878 vs. 0.856 for mMiniLM) | bge-reranker-v2-m3 (568M: cannot be fine-tuned in 4 GB), jina-reranker-v2 (CC-BY-NC) |
| LLM | **Qwen3.5 small (GGUF Q4_K_M) via llama.cpp** | Apache-2.0, Arabic + English, hybrid DeltaNet architecture keeps KV memory small, fits next to the encoders in 4 GB | Qwen2.5-3B (non-commercial license), Llama-3.2-3B (Arabic not officially supported), 7B Arabic models (do not fit 4 GB) |
| Serving | **llama.cpp `llama-server`** (OpenAI-compatible) | Native Windows CUDA build, continuous batching, parallel slots, prompt cache, streaming | vLLM (Linux only, needs far more VRAM), Ollama (same engine, fewer controls) |
| API | FastAPI + Pydantic, SSE streaming | | |
| Sessions | SQLite | Self-hosted, durable, zero setup | Redis (extra service) |

Model licences, sizes and CPU/GPU support are listed in [`model_card.md`](model_card.md).

---

## 2. Language coverage

* Modern Standard Arabic and English, in both directions (AR question → EN document and EN → AR).
* Mixed Arabic/English queries and documents (Arabic text with English product names, codes, terms;
  code-switching such as `أبغى أعرف الـpricing حق الباقة المؤسسية`).
* Arabic-Indic (٠–٩) and Western (0–9) digits are interchangeable everywhere.
* Saudi dialect (bonus): partially covered — dialect queries are in training and evaluation data
  and reported as a separate group; the system is not tuned for other dialects.

### Arabic / English text processing (`hamsrag/text.py`)
* Unicode NFKC (folds Arabic presentation forms from PDFs), tashkeel and tatweel removal,
  invisible-mark removal, Eastern-Arabic → ASCII digits, Arabic decimal/thousands separators.
* For BM25 only: alef (أ إ آ ٱ → ا), ya (ى → ي), ta marbuta (ة → ه), hamza seats; script-boundary
  splitting (`الـpricing` → `ال` + `pricing`); Light10-style Arabic light stemmer (stem ≥ 3 letters to
  avoid `ضمان → ضم`); English Snowball stemming; small stopword lists; number canonicalisation
  (`١٬٥٠٠` = `1,500` = `1500`).
* Neural models receive the light cleanup only (they were pre-trained on real orthography).

### Arabic PDF extraction
Standard extractors scramble Arabic PDFs (reversed digit runs: `12,800 → ٠٠٨٬٢١`, broken lam-alef
ligatures, diacritics mapped to stray symbols). `hamsrag/ingest.py::load_pdf` reads glyphs with
pdfplumber, drops zero-width diacritic glyphs, reverses multi-letter ligature glyphs, then converts
each line from visual to logical order with the Unicode bidi algorithm (python-bidi) *before* NFKC.
Headings are detected from font size and ruled tables are extracted as header/value rows.
Every evidence phrase in the dataset is resolved against the ingested PDFs, so a regression in
extraction fails `prepare_data.py`.

---

## 3. Dataset

**Knowledge bank** (`data/corpus/`): 46 documents of a fictional Saudi furniture retailer and its B2B
workspace subscription arm ("Saraya Living" / "Saraya Business"): policies, price lists, product
catalogues, manuals, FAQs, HR/IT/finance policies. 20 Arabic, 17 English, 9 mixed; 11 PDF, 6 DOCX,
9 HTML, 20 Markdown; dates, SAR prices, names, locations, product codes, both digit systems.
Near-duplicate documents are planted on purpose as hard negatives (e.g. late-delivery compensation
vs. returns policy, Premium vs. Enterprise SLA, warranty policy vs. warranty claim procedure).

**Labels** (`data/qa_src/*.yaml`): facts, each with an *evidence phrase* (resolved to the gold chunk
after ingestion, so labels survive re-chunking), Arabic and English reference answers, and queries of
four types: MSA Arabic, English, code-switched, Saudi dialect. Cross-lingual pairs arise naturally:
every English query on an Arabic document (and vice versa) is cross-lingual.

**Splits — by document** (`data/processed/doc_splits.json`): 31 train / 5 validation / 10 test documents,
stratified by document language. No query about a validation/test document is ever trained on, and
hard negatives are mined from training documents only. `prepare_data.py` asserts there is no query or
document overlap.

| File | Content |
|---|---|
| `data/train.jsonl` | authored queries on training docs + cross-lingual answer augmentation (+ `synthetic_train.jsonl`) |
| `data/validation.jsonl` | authored queries on validation docs (model selection, RRF tuning, gate calibration) |
| `data/test.jsonl` | 138 authored queries on 10 held-out docs: AR 29, EN 24, mixed 32, cross-lingual 53 (incl. dialect) |
| `data/test_unanswerable.jsonl` | 40 plausible questions whose answer is *not* in the knowledge bank |
| `data/test_conversations.jsonl` | 15 three-turn conversations (45 turns) with follow-up references and language switches |

Generation method and limitations are documented in [`data/DATASET.md`](data/DATASET.md).

---

## 4. Fine-tuning

See [`model_card.md`](model_card.md) for full details and hyper-parameters.

**Embedding model** (`scripts/train_embeddings.py`) — full fine-tuning of all 12 transformer layers of
multilingual-e5-base with the 250k-token word-embedding matrix frozen (86M of 278M parameters
trainable). Loss: `CachedMultipleNegativesRankingLoss` (InfoNCE with GradCache) on
(query, positive, hard negative) triples, effective batch 64 with micro-batch 16, temperature 0.05.
Hard negatives are mined with the base model and filtered with a 0.95 × positive-score margin to drop
likely false negatives. Best epoch by validation nDCG@10.

**Reranker** (`scripts/train_reranker.py`) — gte-multilingual-reranker-base, word embeddings frozen,
binary cross-entropy on positives vs. negatives mined from the *fine-tuned* first stage, with
false-negative denoising (candidates the base reranker scores within 1 logit of the gold chunk are
dropped) and a very small learning rate (2e-6). The first attempt at a standard learning rate
collapsed the model; the diagnosis (forward/gradient checks, learning-rate sweep, control run on a
second architecture) and every run are reported in [`model_card.md`](model_card.md).
The best run is chosen on validation by `scripts/select_reranker.py`.

**Data augmentation** — reference answers are reused as additional queries in the *other* language
(back-translation-style cross-lingual pairs); optional LLM-generated queries for training documents
(`scripts/generate_synthetic_queries.py`, local Qwen + round-trip consistency filtering). **This optional
step was not used for the reported models**: they are trained only on the authored queries plus the
answer-based augmentation above.

The generator LLM is **not** fine-tuned (optional in the brief); grounding is enforced by prompting,
a retrieval-score abstention gate, and citation validation.

---

## 5. Setup and reproduction (Windows 10/11, NVIDIA GPU)

```bat
:: 1. environment (conda env in .env, torch CUDA 12.8, requirements, llama.cpp, base models)
tools\setup_env.bat
:: 2. data: ingest the knowledge bank, resolve labels, split by document
.env\python.exe scripts\prepare_data.py
:: 3. indexes
.env\python.exe scripts\build_index.py --lexical-only --fresh
.env\python.exe scripts\build_index.py --embedder e5-base
:: 4. fine-tuning
.env\python.exe scripts\train_embeddings.py
.env\python.exe scripts\build_index.py --embedder e5-base-ft
.env\python.exe scripts\train_reranker.py
:: 5. serve (llama-server + API + web UI at http://127.0.0.1:8010)
tools\start_all.bat
:: 6. inference from the CLI
.env\python.exe scripts\infer.py --mode quick_search --query "سياسة الاسترجاع"
.env\python.exe scripts\infer.py --mode smart_ai_search --query "What is the SLA for premium support?"
.env\python.exe scripts\infer.py --mode interactive --session new --query "كم رسوم التركيب؟"
:: 7. benchmarks
.env\python.exe scripts\benchmark_retrieval.py --embedder e5-base --reranker gte-rerank --tag before_finetuning
.env\python.exe scripts\benchmark_retrieval.py --embedder e5-base-ft --reranker reranker-ft --tag after_finetuning
.env\python.exe scripts\calibrate_gate.py --tag after --write
.env\python.exe scripts\benchmark_generation.py --tag after --judge
.env\python.exe scripts\benchmark_generation.py --tag before --embedder e5-base --reranker gte-rerank --gate-file results/gate_calibration_before.json --judge
.env\python.exe scripts\error_analysis.py --tag after
.env\python.exe scripts\benchmark_latency.py --tag after
.env\python.exe scripts\benchmark_concurrency.py --levels 1,2,4,8,16
.env\python.exe scripts\make_report.py
```

The same steps as double-clickable scripts, in order: `tools\run_A1_baseline.bat`, `run_B_train_embeddings.bat`,
`run_C5_reranker_final.bat`, `run_C4_finalize.bat`, `run_D2_generation.bat`, `run_E_latency.bat`, `run_E3_concurrency.bat`
(each writes a log to `tools\logs\`).

Docker is intentionally not used: `tools/setup_env.bat` + `tools/start_all.bat` are the self-hosted
reproduction path. The code itself is OS-independent (Python 3.11); on Linux, run the same Python
scripts and start `llama-server` with the flags in `tools/start_llm.bat`.

CPU-only Quick Search: `set HAMS_QUICK_ONLY=1 && .env\python.exe demo\app.py` loads no GPU model.

### Adding documents
`scripts/ingest.py --path <file-or-folder>` (or `POST /ingest {"path": ...}`, `POST /ingest/upload`)
loads PDF/DOCX/HTML/MD/TXT, chunks, indexes in FTS5 and appends vectors to FAISS. Re-ingesting a
document replaces its chunks.

---

## 6. API

`POST /search` — body `{"mode": "quick_search" | "smart_ai_search" | "interactive", "query": "...",
"session_id": "new" | "<id>", "top_k": 10, "stream": false}`; with `"stream": true` the AI modes return
Server-Sent Events (`token` deltas, then a `final` payload). Other endpoints: `GET /health`,
`POST /session/new`, `GET /session/{id}`, `POST /ingest`, `POST /ingest/upload`, web UI at `/`,
OpenAPI at `/docs`. Payloads follow the assessment's examples (mode, query, language_detected,
results / answer, citations with doc_id, chunk_id, page, snippet, relevance; latency_ms per stage;
interactive adds session_id, turn, rewritten_query, suggested_followups).

---

## 7. Results

All figures below are produced by the scripts from files in `results/` (this section is regenerated by
`scripts/make_report.py`); nothing is typed in by hand. Hardware: RTX 3050 Laptop 4 GB (see top).

**How the generation metrics are defined** (`hamsrag/gen_metrics.py`, `scripts/benchmark_generation.py`):
* *False "not found"* — share of answerable test questions where the system abstained (retrieval gate or LLM).
* *Correct "not found"* — share of the 26 unanswerable questions where it abstained.
* *Hallucination (answerable)* — an answer containing a number absent from its sources, or (only when the
  answer and the sources share a language) with < 40 % of its content words found in the cited sources.
  Unanswerable: any non-abstaining answer counts as a hallucination.
* *Numeric faithfulness* — share of numbers in the answer that appear in the cited sources.
* *Citation precision / recall* — cited chunks vs. the labelled gold chunk(s).
* *Answer relevance* — cosine similarity (multilingual-e5-base) between the answer and the reference answer.
* *Local-LLM judge* — Qwen3.5-2B answering YES/NO to "is every claim supported by the sources?"; weak and
  self-graded, reported only as a secondary signal.
* Abstentions phrased in the model's own words ("the sources do not contain…") are detected and replaced by
  the canonical not-found sentence, both at serving time and in the benchmark.

A categorised error analysis with verbatim examples is in [`results/error_analysis.md`](results/error_analysis.md).

<!-- RESULTS:BEGIN -->
### Before vs after fine-tuning (held-out test set, hybrid + rerank)

Before = base e5-base + base gte reranker; after = fine-tuned e5-base + fine-tuned reranker. Same LLM, prompts, index and data in both columns.

| Metric | Before | After |
|---|---|---|
| Recall@5 (AR) | 0.862 | 0.828 |
| Recall@5 (EN) | 1.000 | 1.000 |
| Recall@5 (mixed) | 1.000 | 0.969 |
| Recall@5 (cross-lingual) | 0.679 | 0.906 |
| MRR (AR) | 0.849 | 0.833 |
| MRR (EN) | 0.958 | 0.958 |
| nDCG@10 (AR) | 0.884 | 0.861 |
| nDCG@10 (EN) | 0.969 | 0.969 |
| nDCG@10 (all) | 0.810 | 0.873 |
| nDCG@10 (cross-lingual) | 0.628 | 0.793 |
| Faithfulness (numeric) | 0.958 | 0.977 |
| Faithfulness (lexical support) | 0.785 | 0.802 |
| Answer relevance (sim. to reference) | 0.930 | 0.929 |
| Hallucination rate (answerable) | 0.090 | 0.061 |
| Hallucination rate (unanswerable) | 0.077 | 0.115 |
| Correct "not found" on unanswerable | 0.923 | 0.885 |
| False "not found" on answerable | 0.275 | 0.167 |
| Citation precision | 0.900 | 0.900 |
| Local-LLM judge: faithful | 0.720 | 0.739 |
| Answer language correct | 1.000 | 1.000 |
| Human meaning score (1–5) | — | — |
| Human fluency AR (1–5) | — | — |
| Human fluency EN (1–5) | — | — |
| Time to first token (Smart AI, avg) | 456 ms | 396 ms |
| Total latency (Smart AI, avg) | 959 ms | 869 ms |
| p95 latency (Smart AI) | 1,442 ms | 1,234 ms |
| Quick Search p95 | 1 ms | 1 ms |
| GPU memory (all processes, MiB) | 3869 | 3907 |
| System RAM in use when measured (GiB of 16) | 13.160 | 12.720 |
| CPU memory (API process RSS, GiB) | 0.290 | 0.140 |

### Retrieval — after fine-tuning (`e5-base-ft` + `reranker-ft`)

| Mode | Group | n | R@1 | R@5 | R@10 | R@20 | Hit@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | all | 138 | 0.435 | 0.573 | 0.609 | 0.630 | 0.573 | 0.498 | 0.523 |
| bm25 | ar | 29 | 0.690 | 0.931 | 1.000 | 1.000 | 0.931 | 0.805 | 0.853 |
| bm25 | en | 24 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 | 0.954 |
| bm25 | mixed | 32 | 0.594 | 0.812 | 0.844 | 0.938 | 0.812 | 0.691 | 0.723 |
| bm25 | cross | 53 | 0.000 | 0.038 | 0.075 | 0.075 | 0.038 | 0.014 | 0.028 |
| bm25 | cross ar->en | 26 | 0.000 | 0.077 | 0.154 | 0.154 | 0.077 | 0.028 | 0.057 |
| bm25 | cross en->ar | 27 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bm25 | dialect | 7 | 0.286 | 0.429 | 0.714 | 0.714 | 0.429 | 0.401 | 0.474 |
| dense | all | 138 | 0.638 | 0.927 | 0.971 | 0.993 | 0.927 | 0.769 | 0.818 |
| dense | ar | 29 | 0.586 | 0.897 | 1.000 | 1.000 | 0.897 | 0.728 | 0.793 |
| dense | en | 24 | 0.792 | 1.000 | 1.000 | 1.000 | 1.000 | 0.889 | 0.918 |
| dense | mixed | 32 | 0.750 | 0.938 | 0.969 | 1.000 | 0.938 | 0.841 | 0.871 |
| dense | cross | 53 | 0.528 | 0.906 | 0.943 | 0.981 | 0.906 | 0.694 | 0.753 |
| dense | cross ar->en | 26 | 0.462 | 0.885 | 0.885 | 0.962 | 0.885 | 0.635 | 0.692 |
| dense | cross en->ar | 27 | 0.593 | 0.926 | 1.000 | 1.000 | 0.926 | 0.750 | 0.812 |
| dense | dialect | 7 | 0.286 | 0.857 | 1.000 | 1.000 | 0.857 | 0.587 | 0.689 |
| hybrid_rrf | all | 138 | 0.638 | 0.927 | 0.971 | 0.993 | 0.927 | 0.768 | 0.817 |
| hybrid_rrf | ar | 29 | 0.586 | 0.897 | 1.000 | 1.000 | 0.897 | 0.736 | 0.801 |
| hybrid_rrf | en | 24 | 0.792 | 1.000 | 1.000 | 1.000 | 1.000 | 0.889 | 0.918 |
| hybrid_rrf | mixed | 32 | 0.812 | 0.969 | 0.969 | 1.000 | 0.969 | 0.878 | 0.899 |
| hybrid_rrf | cross | 53 | 0.491 | 0.887 | 0.943 | 0.981 | 0.887 | 0.665 | 0.731 |
| hybrid_rrf | cross ar->en | 26 | 0.423 | 0.846 | 0.885 | 0.962 | 0.846 | 0.595 | 0.661 |
| hybrid_rrf | cross en->ar | 27 | 0.556 | 0.926 | 1.000 | 1.000 | 0.926 | 0.732 | 0.798 |
| hybrid_rrf | dialect | 7 | 0.286 | 0.857 | 1.000 | 1.000 | 0.857 | 0.589 | 0.691 |
| hybrid_union | all | 138 | 0.638 | 0.927 | 0.971 | 0.986 | 0.927 | 0.768 | 0.817 |
| hybrid_union | ar | 29 | 0.586 | 0.897 | 1.000 | 1.000 | 0.897 | 0.736 | 0.801 |
| hybrid_union | en | 24 | 0.792 | 1.000 | 1.000 | 1.000 | 1.000 | 0.889 | 0.918 |
| hybrid_union | mixed | 32 | 0.812 | 0.969 | 0.969 | 0.969 | 0.969 | 0.877 | 0.899 |
| hybrid_union | cross | 53 | 0.491 | 0.887 | 0.943 | 0.981 | 0.887 | 0.664 | 0.731 |
| hybrid_union | cross ar->en | 26 | 0.423 | 0.846 | 0.885 | 0.962 | 0.846 | 0.595 | 0.661 |
| hybrid_union | cross en->ar | 27 | 0.556 | 0.926 | 1.000 | 1.000 | 0.926 | 0.732 | 0.798 |
| hybrid_union | dialect | 7 | 0.286 | 0.857 | 1.000 | 1.000 | 0.857 | 0.589 | 0.691 |
| hybrid+rerank | all | 138 | 0.783 | 0.920 | 0.957 | 0.993 | 0.920 | 0.849 | 0.873 |
| hybrid+rerank | ar | 29 | 0.793 | 0.828 | 0.966 | 1.000 | 0.828 | 0.833 | 0.861 |
| hybrid+rerank | en | 24 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.969 |
| hybrid+rerank | mixed | 32 | 0.875 | 0.969 | 1.000 | 1.000 | 0.969 | 0.927 | 0.945 |
| hybrid+rerank | cross | 53 | 0.660 | 0.906 | 0.906 | 0.981 | 0.906 | 0.761 | 0.793 |
| hybrid+rerank | cross ar->en | 26 | 0.577 | 0.846 | 0.846 | 0.962 | 0.846 | 0.691 | 0.724 |
| hybrid+rerank | cross en->ar | 27 | 0.741 | 0.963 | 0.963 | 1.000 | 0.963 | 0.828 | 0.860 |
| hybrid+rerank | dialect | 7 | 0.571 | 0.714 | 1.000 | 1.000 | 0.714 | 0.659 | 0.737 |
| interactive(gold-rewrite) | all | 44 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 0.862 | 0.897 |

### Retrieval — before fine-tuning (`e5-base` + `gte-rerank`)

| Mode | Group | n | R@1 | R@5 | R@10 | R@20 | Hit@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | all | 138 | 0.435 | 0.573 | 0.609 | 0.630 | 0.573 | 0.498 | 0.523 |
| bm25 | ar | 29 | 0.690 | 0.931 | 1.000 | 1.000 | 0.931 | 0.805 | 0.853 |
| bm25 | en | 24 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 | 0.954 |
| bm25 | mixed | 32 | 0.594 | 0.812 | 0.844 | 0.938 | 0.812 | 0.691 | 0.723 |
| bm25 | cross | 53 | 0.000 | 0.038 | 0.075 | 0.075 | 0.038 | 0.014 | 0.028 |
| bm25 | cross ar->en | 26 | 0.000 | 0.077 | 0.154 | 0.154 | 0.077 | 0.028 | 0.057 |
| bm25 | cross en->ar | 27 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bm25 | dialect | 7 | 0.286 | 0.429 | 0.714 | 0.714 | 0.429 | 0.401 | 0.474 |
| dense | all | 138 | 0.485 | 0.725 | 0.833 | 0.891 | 0.725 | 0.600 | 0.651 |
| dense | ar | 29 | 0.586 | 0.862 | 1.000 | 1.000 | 0.862 | 0.732 | 0.797 |
| dense | en | 24 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.969 |
| dense | mixed | 32 | 0.656 | 0.875 | 0.875 | 0.938 | 0.875 | 0.745 | 0.772 |
| dense | cross | 53 | 0.132 | 0.434 | 0.641 | 0.755 | 0.434 | 0.278 | 0.353 |
| dense | cross ar->en | 26 | 0.077 | 0.269 | 0.500 | 0.577 | 0.269 | 0.186 | 0.246 |
| dense | cross en->ar | 27 | 0.185 | 0.593 | 0.778 | 0.926 | 0.593 | 0.366 | 0.455 |
| dense | dialect | 7 | 0.286 | 0.571 | 0.857 | 0.857 | 0.571 | 0.426 | 0.524 |
| hybrid_rrf | all | 138 | 0.471 | 0.717 | 0.826 | 0.891 | 0.717 | 0.587 | 0.638 |
| hybrid_rrf | ar | 29 | 0.586 | 0.897 | 1.000 | 1.000 | 0.897 | 0.735 | 0.800 |
| hybrid_rrf | en | 24 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.969 |
| hybrid_rrf | mixed | 32 | 0.625 | 0.875 | 0.938 | 1.000 | 0.875 | 0.734 | 0.780 |
| hybrid_rrf | cross | 53 | 0.113 | 0.396 | 0.585 | 0.717 | 0.396 | 0.249 | 0.314 |
| hybrid_rrf | cross ar->en | 26 | 0.077 | 0.231 | 0.500 | 0.577 | 0.231 | 0.173 | 0.237 |
| hybrid_rrf | cross en->ar | 27 | 0.148 | 0.556 | 0.667 | 0.852 | 0.556 | 0.321 | 0.389 |
| hybrid_rrf | dialect | 7 | 0.286 | 0.714 | 0.857 | 0.857 | 0.714 | 0.431 | 0.529 |
| hybrid_union | all | 138 | 0.471 | 0.717 | 0.826 | 0.891 | 0.717 | 0.587 | 0.638 |
| hybrid_union | ar | 29 | 0.586 | 0.897 | 1.000 | 1.000 | 0.897 | 0.735 | 0.800 |
| hybrid_union | en | 24 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.969 |
| hybrid_union | mixed | 32 | 0.625 | 0.875 | 0.938 | 1.000 | 0.875 | 0.734 | 0.780 |
| hybrid_union | cross | 53 | 0.113 | 0.396 | 0.585 | 0.717 | 0.396 | 0.249 | 0.315 |
| hybrid_union | cross ar->en | 26 | 0.077 | 0.231 | 0.500 | 0.577 | 0.231 | 0.174 | 0.237 |
| hybrid_union | cross en->ar | 27 | 0.148 | 0.556 | 0.667 | 0.852 | 0.556 | 0.322 | 0.390 |
| hybrid_union | dialect | 7 | 0.286 | 0.714 | 0.857 | 0.857 | 0.714 | 0.431 | 0.529 |
| hybrid+rerank | all | 138 | 0.717 | 0.848 | 0.891 | 0.891 | 0.848 | 0.786 | 0.810 |
| hybrid+rerank | ar | 29 | 0.793 | 0.862 | 1.000 | 1.000 | 0.862 | 0.849 | 0.884 |
| hybrid+rerank | en | 24 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.969 |
| hybrid+rerank | mixed | 32 | 0.812 | 1.000 | 1.000 | 1.000 | 1.000 | 0.896 | 0.923 |
| hybrid+rerank | cross | 53 | 0.528 | 0.679 | 0.717 | 0.717 | 0.679 | 0.606 | 0.628 |
| hybrid+rerank | cross ar->en | 26 | 0.423 | 0.577 | 0.577 | 0.577 | 0.577 | 0.509 | 0.520 |
| hybrid+rerank | cross en->ar | 27 | 0.630 | 0.778 | 0.852 | 0.852 | 0.778 | 0.700 | 0.732 |
| hybrid+rerank | dialect | 7 | 0.571 | 0.571 | 0.857 | 0.857 | 0.571 | 0.623 | 0.673 |
| interactive(gold-rewrite) | all | 44 | 0.727 | 0.955 | 0.977 | 0.977 | 0.955 | 0.838 | 0.873 |

### Generation — after fine-tuning

| Set | n | language_correctness | abstain_rate | hallucination_rate | context_recall | context_precision | citation_precision | citation_recall | numeric_faithfulness | lexical_support | answer_relevance | judge_faithful | bleu | chrf | rouge_l |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 138 | 1.000 | 0.167 | 0.061 | 0.884 | 0.221 | 0.900 | 0.913 | 0.977 | 0.802 | 0.929 | 0.739 | 17.750 | 42.860 | 0.369 |
| test:ar | 29 | 1.000 | 0.138 | 0.000 | 0.828 | 0.207 | 0.880 | 0.880 | 1.000 | 0.801 | 0.944 | 0.720 | 23.490 | 48.870 | 0.442 |
| test:cross | 53 | 1.000 | 0.264 | 0.103 | 0.830 | 0.207 | 0.808 | 0.821 | 0.963 | 0.594 | 0.912 | 0.590 | 7.310 | 33.390 | 0.269 |
| test:mixed | 32 | 1.000 | 0.125 | 0.036 | 0.938 | 0.234 | 0.982 | 1.000 | 0.982 | 0.830 | 0.932 | 0.857 | 20.010 | 42.730 | 0.368 |
| test:en | 24 | 1.000 | 0.042 | 0.087 | 1.000 | 0.250 | 0.978 | 1.000 | 0.972 | 0.830 | 0.935 | 0.870 | 27.090 | 51.780 | 0.461 |
| unanswerable | 26 | 1.000 | 0.885 | 0.115 | — | — | — | — | — | — | — | — | — | — | — |
| conversation | 45 | 1.000 | 0.178 | 0.158 | 0.795 | 0.199 | 0.649 | 0.811 | 0.974 | 0.674 | — | 0.486 | — | — | — |

### Generation — before fine-tuning

| Set | n | language_correctness | abstain_rate | hallucination_rate | context_recall | context_precision | citation_precision | citation_recall | numeric_faithfulness | lexical_support | answer_relevance | judge_faithful | bleu | chrf | rouge_l |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 138 | 1.000 | 0.275 | 0.090 | 0.812 | 0.203 | 0.900 | 0.910 | 0.958 | 0.785 | 0.930 | 0.720 | 19.200 | 44.150 | 0.388 |
| test:ar | 29 | 1.000 | 0.379 | 0.056 | 0.828 | 0.207 | 0.944 | 0.944 | 1.000 | 0.768 | 0.947 | 0.833 | 30.180 | 53.810 | 0.490 |
| test:cross | 53 | 1.000 | 0.415 | 0.097 | 0.641 | 0.160 | 0.758 | 0.774 | 0.957 | 0.636 | 0.920 | 0.581 | 9.750 | 36.510 | 0.314 |
| test:mixed | 32 | 1.000 | 0.094 | 0.103 | 0.938 | 0.234 | 0.948 | 0.966 | 0.921 | 0.810 | 0.926 | 0.690 | 16.350 | 39.990 | 0.339 |
| test:en | 24 | 1.000 | 0.083 | 0.091 | 1.000 | 0.250 | 1.000 | 1.000 | 0.972 | 0.822 | 0.936 | 0.864 | 28.970 | 52.080 | 0.471 |
| unanswerable | 26 | 1.000 | 0.923 | 0.077 | — | — | — | — | — | — | — | — | — | — | — |
| conversation | 45 | 1.000 | 0.222 | 0.111 | 0.750 | 0.188 | 0.664 | 0.800 | 0.976 | 0.732 | — | 0.457 | — | — | — |

### Latency (sequential, NVIDIA GeForce RTX 3050 Laptop GPU, 16 CPU threads)

| Mode | Stage | n | avg | p50 | p95 | max |
|---|---|---|---|---|---|---|
| quick_search | total | 60 | 1 ms | 1 ms | 1 ms | 1 ms |
| quick_search | search | 60 | 0 ms | 0 ms | 0 ms | 1 ms |
| quick_search | preprocessing | 60 | 0 ms | 0 ms | 0 ms | 0 ms |
| retrieval_only(hybrid+rerank) | retrieval_stage | 60 | 14 ms | 14 ms | 17 ms | 18 ms |
| retrieval_only(hybrid+rerank) | rerank | 60 | 143 ms | 149 ms | 158 ms | 162 ms |
| retrieval_only(hybrid+rerank) | first_result | 60 | 14 ms | 14 ms | 17 ms | 18 ms |
| smart_ai_search | retrieval | 60 | 17 ms | 16 ms | 23 ms | 26 ms |
| smart_ai_search | rerank | 60 | 148 ms | 151 ms | 168 ms | 182 ms |
| smart_ai_search | time_to_first_token | 60 | 396 ms | 378 ms | 569 ms | 628 ms |
| smart_ai_search | generation_total | 60 | 704 ms | 691 ms | 1,065 ms | 1,473 ms |
| smart_ai_search | total | 60 | 869 ms | 840 ms | 1,234 ms | 1,644 ms |
| interactive | query_rewrite | 45 | 150 ms | 196 ms | 372 ms | 386 ms |
| interactive | time_to_first_token | 45 | 539 ms | 515 ms | 825 ms | 930 ms |
| interactive | total | 45 | 1,383 ms | 1,331 ms | 2,197 ms | 4,325 ms |
| interactive | query_rewrite_when_triggered | 29 | 233 ms | 207 ms | 380 ms | 386 ms |

**Latency vs corpus size** (p95 per query)

| Chunks | BM25 | FAISS flat | FAISS HNSW |
|---|---|---|---|
| 1000 | 1 ms | 0 ms | 0 ms |
| 10000 | 9 ms | 3 ms | 0 ms |
| 100000 | 75 ms | 22 ms | 1 ms |

**Rerank top-k** (latency and quality)

| rerank_k | avg | p95 | R@5 | MRR |
|---|---|---|---|---|
| 5 | 43 ms | 50 ms | 0.950 | 0.876 |
| 10 | 77 ms | 88 ms | 0.950 | 0.870 |
| 20 | 142 ms | 160 ms | 0.933 | 0.863 |
| 30 | 194 ms | 210 ms | 0.917 | 0.852 |
| 50 | 326 ms | 353 ms | 0.917 | 0.850 |

**Context size vs time to first token**

| context chunks | prompt tokens | TTFT avg | TTFT p95 |
|---|---|---|---|
| 2 | 665.6 | 342 ms | 413 ms |
| 4 | 981.7 | 395 ms | 484 ms |
| 6 | 1241.4 | 436 ms | 475 ms |

**Longest input tested:** 164 words (≈1032 prompt tokens incl. context), total 1,486 ms, answered: True.

### Concurrency (through the HTTP API)

| Mode | Users | Requests | avg | p95 | QPS | Failures | GPU MiB (max) | RAM GiB (max) |
|---|---|---|---|---|---|---|---|---|
| quick_search | 1 | 64 | 4 ms | 5 ms | 250.29 | 0.000 | 3820.000 | 15.240 |
| quick_search | 2 | 64 | 4 ms | 5 ms | 492.27 | 0.000 | 3820.000 | 15.250 |
| quick_search | 4 | 64 | 9 ms | 13 ms | 404.33 | 0.000 | 3820.000 | 15.250 |
| quick_search | 8 | 64 | 28 ms | 60 ms | 264.26 | 0.000 | 3820.000 | 15.260 |
| quick_search | 16 | 128 | 64 ms | 144 ms | 231.29 | 0.000 | 3820.000 | 15.260 |
| smart_ai_search | 1 | 16 | 2,227 ms | 2,568 ms | 0.45 | 0.000 | 3843.000 | 15.620 |
| smart_ai_search | 2 | 16 | 2,506 ms | 3,966 ms | 0.76 | 0.000 | 3851.000 | 15.570 |
| smart_ai_search | 4 | 16 | 3,576 ms | 4,478 ms | 1.06 | 0.000 | 3876.000 | 15.540 |
| smart_ai_search | 8 | 16 | 5,528 ms | 8,344 ms | 1.33 | 0.000 | 3912.000 | 15.570 |
| smart_ai_search | 16 | 32 | 9,364 ms | 12,827 ms | 1.43 | 0.000 | 3967.000 | 15.720 |
| interactive | 1 | 16 | 2,502 ms | 2,901 ms | 0.4 | 0.000 | 3967.000 | 15.720 |
| interactive | 2 | 16 | 3,967 ms | 6,033 ms | 0.5 | 0.000 | 3969.000 | 15.710 |
| interactive | 4 | 16 | 5,582 ms | 8,574 ms | 0.67 | 0.000 | 3969.000 | 15.720 |
| interactive | 8 | 16 | 8,672 ms | 12,258 ms | 0.74 | 0.000 | 3969.000 | 14.810 |
| interactive | 16 | 32 | 17,808 ms | 31,170 ms | 0.68 | 0.000 | 3968.000 | 15.130 |
<!-- RESULTS:END -->

---

## 8. Known limitations

* **Hardware.** Benchmarks ran on a 4 GB laptop GPU that also drives the display; latency on an L4 is
  expected to be lower but was not measured.
* **Data size.** The knowledge bank and most queries are synthetic (authored with an LLM's help, then
  reviewed); real enterprise documents are longer and noisier. The test set is 138 questions on 10 held-out
  documents, so differences of 1–3 points are within noise.
* **Arabic in-language retrieval did not improve.** After fine-tuning, Arabic R@5 dropped 0.862 → 0.828
  (one question out of 29) while cross-lingual R@5 rose 0.679 → 0.906. The training data is dominated by
  cross-lingual pairs; more Arabic–Arabic hard negatives would be the next thing to add.
* **Over-cautious generator.** 16.7 % of answerable questions (23 of 138) still get "not found". 8 are
  blocked by the retrieval gate; in 13 of the other 15 the right chunk was in the context and the 2B model
  refused anyway (see error analysis). A larger generator (Qwen3.5-4B, the L4 profile) or LoRA tuning on
  answer/abstain examples should reduce this.
* **Run-to-run variation.** The LLM output is not fully deterministic; repeating the generation benchmark
  moved the generation metrics by about 1–2 points, so smaller differences should not be read as real.
* **Conversations.** In multi-turn tests the gold passage reaches the LLM in 79.5 % of turns (75.0 % before
  fine-tuning); most misses are follow-ups whose rewrite loses the referent. Adding a worked example to the
  rewrite prompt was tried and reverted: the example leaked into rewrites and recall fell to 68 %.
* **Number confusion.** Some answers attach a correct number to the wrong attribute (e.g. a maximum load
  reported as a weight). Numeric faithfulness catches numbers that are absent from the sources, not
  numbers that are present but misattributed.
* **Conversation citations.** Multi-turn citation precision is 0.57: the labels list one gold chunk per
  turn while the model often cites several relevant chunks, and some follow-ups lose their referent.
* **Repetition loops.** The 2B model occasionally repeats the same sentence until it hits the token limit
  (seen while recording the demo, on the first turn of the leave-policy conversation). A repetition guard
  (`_repeat_cut` in `hamsrag/rag.py`) now stops generation at the first repeated sentence and drops it from
  the answer. It was added after the benchmarks above, so those numbers were measured without it.
* **Saudi dialect** coverage is limited to 7 dialect test queries.
* **Faithfulness is measured with proxies** (numeric faithfulness, same-language lexical support,
  citation validity) plus a weak local judge, not with a strong external LLM judge (no paid API allowed).

## 9. Possible improvements

Larger and real document set; distillation of the reranker from bge-reranker-v2-m3; LoRA fine-tuning of
the generator for citation style; ONNX/INT8 encoders for CPU-only Smart AI Search; speculative
decoding (Qwen3.5 MTP heads); per-chunk translations to strengthen BM25 on cross-lingual queries.
